# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Archived Dense/Router/MoE arithmetic: FP64 derivatives, rounding points, routing and expert dispatch."""
# The Transformers config signature is generated dynamically; keyword checks are false positives here.
# pylint: disable=unexpected-keyword-arg
from collections.abc import Callable
import unittest
from unittest.mock import patch

import torch
import torch_npu
from torch.nn import functional as F

from hyper_parallel.models.jt_deepseek_v3.archived_moe import (
    ArchivedMLP,
    ArchivedMoE,
    ArithmeticSigmoid,
    RouterRoutedInputPair,
    RoutingNormalization,
    _RouterLinear,
    _SwiGLU,
    _WeightedProduct,
    _expert_order,
)
from hyper_parallel.models.jt_deepseek_v3.archived_reductions import weighted_sum
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3Config
from tests.common.mark_utils import arg_mark

PADDING = 4  # One pad token per routed expert.


def moe_config(hidden_size: int = 5120) -> JTDeepseekV3Config:
    """Four routed experts, top-2, a 2.5 routing scale and the archived hidden width."""
    return JTDeepseekV3Config(
        hidden_size=hidden_size, moe_intermediate_size=4, n_routed_experts=4, n_shared_experts=1,
        num_experts_per_tok=2, n_group=1, topk_group=1, norm_topk_prob=True, routed_scaling_factor=2.5,
        moe_aux_loss_coeff=0.01, use_pad_tokens=True)


def cpu_swiglu(pair: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """CPU stand-in for ``torch_npu.npu_swiglu``: ``silu(first half) * second half`` in FP32, rounded once."""
    gate, up = pair.float().chunk(2, dim=dim)
    return (F.silu(gate) * up).to(pair.dtype)


def biased_moe() -> tuple[ArchivedMoE, torch.Tensor, torch.Tensor]:
    """Build an MoE whose logits are the first four hidden features and whose bias favours expert 3.

    Returns:
        The module, the pad-prefixed FP32 hidden states of three tokens and their router logits.
    """
    moe = ArchivedMoE(moe_config())
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0], [-1.0, 0.5, 1.5, 0.0], [0.25, -0.5, 1.0, 2.0]])
    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.weight[:, :4] = torch.eye(4)
        moe.gate.e_score_correction_bias.copy_(torch.tensor([0.0, 0.0, 0.0, 1.0]))
    hidden = torch.zeros(1, PADDING + 3, 5120)
    hidden[0, PADDING:, :4] = logits
    return moe, hidden, logits


def scaled_moe(seed: int) -> tuple[ArchivedMoE, torch.Tensor]:
    """Build an MoE with random FP32 routing, BF16 shared experts and distinct BF16 expert weights.

    Returns:
        The module and one BF16 weight vector per routed expert, ``[4, 5120]``.
    """
    torch.manual_seed(seed)
    moe = ArchivedMoE(moe_config())
    moe.shared_experts.to(torch.bfloat16)
    with torch.no_grad():
        moe.gate.weight.normal_(std=0.01)
    return moe, (torch.rand(4, 5120) + 0.5).bfloat16()


def scaled_experts(scales: torch.Tensor) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return a CPU stand-in for ``forward_expert_major`` in which expert ``e`` scales its rows by ``scales[e]``.

    Elementwise experts make each row's result independent of batching, so per-token references match bitwise.
    """

    def forward_expert_major(inputs: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Scale each expert's contiguous block of expert-major rows by that expert's weights."""
        return torch.cat([rows * scale for rows, scale in zip(inputs.split(counts.tolist()), scales)])

    return forward_expert_major


class TestArchivedMoE(unittest.TestCase):
    """Check the archived router, dense and MoE arithmetic against independent references."""

    def setUp(self) -> None:
        """Serve the fused NPU SwiGLU from its CPU stand-in."""
        self.enterContext(patch.object(torch_npu, "npu_swiglu", new=cpu_swiglu, create=True))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_routing_derivative_matches_double_precision_objective(self):
        """Sigmoid and both normalizations match FP64 autograd of the plain routing math, scale included."""
        torch.manual_seed(47)
        inputs = torch.randn(3, 8, dtype=torch.float64)
        indices = inputs.topk(2, dim=-1).indices
        first, second = torch.randn(3, 2, dtype=torch.float64), torch.randn(3, 8, dtype=torch.float64)
        actual_input = inputs.clone().requires_grad_()
        scores = ArithmeticSigmoid.apply(actual_input)
        selected, normalized = RoutingNormalization.apply(scores, indices, 2.827)
        (selected.mul(first).sum() + normalized.mul(second).sum()).backward()
        expected_input = inputs.clone().requires_grad_()
        scores = expected_input.sigmoid()
        expected_selected = scores.gather(1, indices) / scores.gather(1, indices).sum(-1, keepdim=True) * 2.827
        expected_normalized = scores / scores.sum(-1, keepdim=True)
        (expected_selected.mul(first).sum() + expected_normalized.mul(second).sum()).backward()
        pairs = ((selected, expected_selected), (normalized, expected_normalized),
                 (actual_input.grad, expected_input.grad))
        for actual, expected in pairs:
            torch.testing.assert_close(actual.detach(), expected.detach(), rtol=1e-12, atol=1e-12)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_sigmoid_derivative_multiplies_the_gradient_first(self):
        """The FP32 sigmoid derivative rounds as (gradient * sigmoid) * (1 - sigmoid), the archived order."""
        torch.manual_seed(48)
        values, gradient = torch.randn(4096) * 4, torch.randn(4096)
        actual = values.clone().requires_grad_()
        ArithmeticSigmoid.apply(actual).backward(gradient)
        result = values.sigmoid()
        expected = (gradient * result) * (1 - result)
        self.assertFalse(torch.equal(expected, (result * (1 - result)) * gradient))  # the order is observable
        torch.testing.assert_close(actual.grad, expected, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_router_and_expert_input_derivatives_join_in_fp32_then_round_once(self):
        """The BF16 input gradient rounds the FP32 sum once instead of adding a pre-rounded router term."""
        hidden = torch.zeros(2, dtype=torch.bfloat16, requires_grad=True)
        expert_input, router_input = RouterRoutedInputPair.apply(hidden)
        # BF16 rounds 2**-8 + 2**-20 to 2**-8, and 1 + 2**-8 ties to even (1.0); the FP32 sum lies above the tie.
        router_gradient = torch.tensor([2.0 ** -8 + 2.0 ** -20, 3.0])
        expert_gradient = torch.tensor([1.0, 0.0], dtype=torch.bfloat16)
        torch.autograd.backward((expert_input, router_input), (expert_gradient, router_gradient))
        expected = torch.tensor([1.0 + 2.0 ** -7, 3.0], dtype=torch.bfloat16)
        torch.testing.assert_close(hidden.grad, expected, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_weighted_product_rounds_at_the_archived_points(self):
        """FP32 forward product; main/MTP output gradients and the weight gradient round where the archive did."""
        torch.manual_seed(5120)
        outputs, gradient = torch.randn(3, 5120).bfloat16(), torch.randn(3, 5120)
        weights = torch.rand(3, 1) / 3  # FP32 weights that BF16 cannot represent
        # Products of BF16/FP32 factors are exact in FP64, so each cast below is exactly one rounding.
        expected_product = (outputs.double() * weights.double()).float()
        main_gradient = (gradient.double() * weights.double()).float().bfloat16()
        mtp_gradient = (gradient.bfloat16().double() * weights.bfloat16().double()).bfloat16()
        self.assertFalse(torch.equal(main_gradient, mtp_gradient))
        rounded_products = (gradient.bfloat16().double() * outputs.double()).bfloat16()
        weight_gradient = weighted_sum(rounded_products).bfloat16().float()
        for mtp, output_gradient in ((False, main_gradient), (True, mtp_gradient)):
            actual_outputs, actual_weights = outputs.clone().requires_grad_(), weights.clone().requires_grad_()
            product = _WeightedProduct.apply(actual_outputs, actual_weights, mtp)
            product.backward(gradient)
            torch.testing.assert_close(product.detach(), expected_product, rtol=0, atol=0)
            torch.testing.assert_close(actual_outputs.grad, output_gradient, rtol=0, atol=0)
            torch.testing.assert_close(actual_weights.grad, weight_gradient, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_swiglu_matches_double_precision_with_main_and_mtp_up_rounding(self):
        """SwiGLU matches FP64 autograd; MTP rounds SiLU before the up-gradient product, main rounds once."""
        torch.manual_seed(203)
        gate, up = (torch.randn(64, 48) * 2).bfloat16(), torch.randn(64, 48).bfloat16()
        upstream = torch.randn(64, 48).bfloat16()
        reference_gate, reference_up = gate.double().requires_grad_(), up.double().requires_grad_()
        reference = F.silu(reference_gate) * reference_up
        reference.backward(upstream.double())
        silu = F.silu(gate.float())
        main_up = (upstream.float() * silu).bfloat16()
        mtp_up = (upstream.float() * silu.bfloat16().float()).bfloat16()
        self.assertFalse(torch.equal(main_up, mtp_up))
        for rounded_up, expected_up in ((False, main_up), (True, mtp_up)):
            actual_gate, actual_up = gate.clone().requires_grad_(), up.clone().requires_grad_()
            output = _SwiGLU.apply(actual_gate, actual_up, rounded_up)
            output.backward(upstream)
            # Twice the BF16 unit roundoff per rounding (one; two for the MTP up gradient) absorbs FP32 error.
            torch.testing.assert_close(output.double(), reference.detach(), rtol=2 ** -7, atol=0)
            torch.testing.assert_close(actual_gate.grad.double(), reference_gate.grad, rtol=2 ** -7, atol=1e-5)
            torch.testing.assert_close(actual_up.grad.double(), reference_up.grad, rtol=2 ** -6, atol=0)
            torch.testing.assert_close(actual_up.grad, expected_up, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_mlp_interleaves_gate_and_up_rows_like_the_product_mlp(self):
        """The fused gate/up GEMM and its gradients match the product down(silu(gate(x)) * up(x)) math."""
        torch.manual_seed(204)
        mlp = ArchivedMLP(moe_config(hidden_size=16), intermediate_size=8)
        parameters = (mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight)
        inputs, upstream = torch.randn(5, 16), torch.randn(5, 16)
        results = []
        for archived in (True, False):
            mlp.zero_grad(set_to_none=True)
            values = inputs.clone().requires_grad_()
            output = mlp(values) if archived else mlp.down_proj(F.silu(mlp.gate_proj(values)) * mlp.up_proj(values))
            output.backward(upstream)
            results.append([output.detach(), values.grad] + [parameter.grad for parameter in parameters])
        for actual, expected in zip(*results):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_route_selects_with_expert_bias_and_weights_with_raw_scores(self):
        """The bias picks the experts; weights, auxiliary loss and accumulated counts follow the hand computation."""
        moe, hidden, logits = biased_moe()
        indices, weights = moe.route(hidden)
        # The unbiased top-2 sets are {0, 1}, {1, 2} and {2, 3}; the +1 bias makes expert 3 win everywhere.
        chosen = torch.tensor([[0, 3], [2, 3], [2, 3]])
        scores = logits.double().sigmoid()
        chosen_scores = scores.gather(1, chosen)
        expected_weights = torch.zeros(3, 4, dtype=torch.float64).scatter(
            1, chosen, chosen_scores / chosen_scores.sum(-1, keepdim=True) * 2.5)
        real_indices, real_weights = indices[PADDING:], weights[PADDING:]
        torch.testing.assert_close(real_indices.sort(-1).values, chosen, rtol=0, atol=0)
        torch.testing.assert_close(torch.zeros(3, 4).scatter(1, real_indices, real_weights).double(),
                                   expected_weights, rtol=1e-6, atol=0)
        counts = torch.tensor([1.0, 0.0, 2.0, 3.0])
        load = counts.double() / 6
        expected_loss = ((scores / scores.sum(-1, keepdim=True)).mean(0) * load).sum() * 4 * 0.01
        torch.testing.assert_close(moe.pop_auxiliary_loss().double(), expected_loss, rtol=1e-6, atol=0)
        torch.testing.assert_close(moe.tokens_per_expert, counts, rtol=0, atol=0)
        moe.route(hidden)
        torch.testing.assert_close(moe.tokens_per_expert, 2 * counts, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_route_spreads_pad_tokens_over_every_expert_with_zero_weight(self):
        """Pad token p uses experts (2p, 2p + 1) % 4 with zero weight ahead of the real tokens."""
        moe, hidden, _ = biased_moe()
        indices, weights = moe.route(hidden)
        self.assertEqual(indices.shape, (PADDING + 3, 2))
        torch.testing.assert_close(indices[:PADDING], torch.tensor([[0, 1], [2, 3], [0, 1], [2, 3]]),
                                   rtol=0, atol=0)
        torch.testing.assert_close(weights[:PADDING], torch.zeros(PADDING, 2), rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_moe_forward_matches_a_per_token_reference(self):
        """Each token sums w_k * expert_k(x) in FP32, rounds once, and joins the shared experts in FP32."""
        moe, scales = scaled_moe(255)
        hidden = torch.randn(1, 5, 5120).bfloat16()
        with patch.object(moe.experts, "forward_expert_major", new=scaled_experts(scales)):
            actual = moe(hidden)
        indices, weights = moe.route(torch.cat((hidden.new_zeros(1, PADDING, 5120), hidden), dim=1))
        indices, weights = indices[PADDING:], weights[PADDING:]
        per_token = (hidden[0].unsqueeze(1) * scales[indices]).float() * weights.unsqueeze(-1)
        expected = per_token.sum(1).bfloat16().float() + moe.shared_experts(hidden)[0].float()
        torch.testing.assert_close(actual, expected.unsqueeze(0), rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_moe_input_gradient_rounds_the_fp32_sum_of_expert_and_router_derivatives(self):
        """The BF16 input gradient is the expert-path derivative plus the FP32 router derivative, rounded once."""
        moe, scales = scaled_moe(256)
        logit_gradients = []

        def record_logit_gradient(_module: torch.nn.Module, _args: tuple, logits: torch.Tensor) -> None:
            """Keep the derivative that reaches the FP32 router logits."""
            logits.register_hook(logit_gradients.append)

        moe.gate.register_forward_hook(record_logit_gradient)
        hidden = torch.cat((torch.zeros(1, PADDING, 5120), torch.randn(1, 5, 5120)), dim=1).bfloat16()
        upstream = torch.randn(PADDING + 5, 5120).bfloat16()
        actual = hidden.clone().requires_grad_()
        with patch.object(moe.experts, "forward_expert_major", new=scaled_experts(scales)):
            moe.local_routed_forward(actual).backward(upstream.unsqueeze(0))
        indices, weights = moe.route(hidden)
        # Per slot: the main output gradient rounds the FP32 product, the BF16 expert scales it; slots add in BF16.
        slots = (upstream.float().unsqueeze(1) * weights.unsqueeze(-1)).bfloat16() * scales[indices]
        expert = slots.float().sum(1).bfloat16()
        router = torch.cat((torch.zeros(PADDING, 5120), logit_gradients[0] @ moe.gate.weight.data))
        expected = (expert.float() + router).bfloat16()
        self.assertFalse(torch.equal(expected, expert + router.bfloat16()))  # a BF16 join is observable
        torch.testing.assert_close(actual.grad[0], expected, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_expert_order_groups_a_realistic_batch_by_expert(self):
        """4104 top-2 tokens over 8 experts: a permutation listing each expert's entries contiguously in id order."""
        torch.manual_seed(4104)
        indices = torch.rand(4104, 8).topk(2, dim=-1).indices
        order = _expert_order(indices)
        torch.testing.assert_close(order.sort().values, torch.arange(indices.numel()), rtol=0, atol=0)
        torch.testing.assert_close(indices.reshape(-1)[order], indices.reshape(-1).sort().values, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_expert_order_ranks_the_top_k_major_sequence(self):
        """Entries come out in the order the top-k-major id sequence (slot 0 of every token, then slot 1) sorts."""
        torch.manual_seed(4105)
        tokens = 4104
        indices = torch.rand(tokens, 8).topk(2, dim=-1).indices
        order = _expert_order(indices)
        ranked = _expert_order(indices.transpose(0, 1).reshape(-1, 1))
        # Entry (token t, slot k) is position 2t + k of ``indices`` and position k * tokens + t of the sequence.
        torch.testing.assert_close(torch.stack((order // 2, order % 2)),
                                   torch.stack((ranked % tokens, ranked // tokens)), rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_expert_order_breaks_ties_by_the_original_top_k_major_position(self):
        """Equal expert ids keep their top-k-major relative order now that both sorts are stable (#252 R6)."""
        # token-major (token, slot) -> expert id: tokens 0 and 2 route (1, 0), token 1 routes (0, 1).
        indices = torch.tensor([[1, 0], [0, 1], [1, 0]])
        order = _expert_order(indices)
        # Expert 0 sits at top-k-major positions 1, 3, 5 (token 1 slot 0, token 0 slot 1, token 2 slot 1);
        # expert 1 sits at 0, 2, 4. A stable tie-break keeps each group in that top-k-major order.
        torch.testing.assert_close(order, torch.tensor([2, 1, 5, 0, 4, 3]), rtol=0, atol=0)
        torch.testing.assert_close(indices.reshape(-1)[order], torch.tensor([0, 0, 0, 1, 1, 1]), rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_router_linear_gradients_match_double_precision_autograd(self):
        """Router logits and both input and weight gradients match FP64 autograd of F.linear."""
        torch.manual_seed(245)
        values, weight = torch.randn(7, 16, dtype=torch.float64), torch.randn(4, 16, dtype=torch.float64)
        upstream = torch.randn(7, 4, dtype=torch.float64)
        expected_values, expected_weight = values.clone().requires_grad_(), weight.clone().requires_grad_()
        expected = F.linear(expected_values, expected_weight)  # pylint: disable=not-callable
        expected.backward(upstream)
        for dtype, tolerance in ((torch.float64, 1e-12), (torch.float32, 1e-5)):
            actual_values = values.to(dtype, copy=True).requires_grad_()
            actual_weight = weight.to(dtype, copy=True).requires_grad_()
            logits = _RouterLinear.apply(actual_values, actual_weight)
            logits.backward(upstream.to(dtype))
            pairs = ((logits, expected), (actual_values.grad, expected_values.grad),
                     (actual_weight.grad, expected_weight.grad))
            for actual, reference in pairs:
                torch.testing.assert_close(actual.detach().double(), reference.detach(),
                                           rtol=tolerance, atol=tolerance)


if __name__ == "__main__":
    unittest.main()
