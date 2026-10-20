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
"""Complete standalone model construction and optional acceleration boundaries."""
# pylint: disable=protected-access

import copy
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (
    JTDeepseekV3Config,
    JTDeepseekV3ForCausalLM, JTDeepseekV3Decoder, JTDeepseekV3Gate, JTDeepseekV3MoE,
    JTDeepseekV3Attention, JTDeepseekV3MLAAttention, JTDeepseekV3RotaryEmbedding,
)
from hyper_parallel.components.modules.mtp import MultiTokenPredictionLayer
from hyper_parallel.core.activation_memory import checkpoint_wrapper
from hyper_parallel.data.batching import TextParallelBatch
from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.models.build_options import FSDP2Config
from hyper_parallel.models.replacement import compile_module_replacements, apply_module_replacements
from hyper_parallel.models.jt_deepseek_v3.adapter.jt_builder import _load_reference_state
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (
    ArithmeticSigmoid, JTDeepseekV3MLP, RouterRoutedInputPair, RoutingNormalization, _RouterLinear, _SwiGLU,
    _WeightedProduct, _expert_order, weighted_sum,
)
from hyper_parallel.models.registry import get_model_adapter
from hyper_parallel.trainer.config import entries_to_module_replacements
from tests.common.jt_config import jt_test_config
from tests.common.mark_utils import arg_mark


def cpu_swiglu(pair: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """CPU stand-in for ``torch_npu.npu_swiglu``: ``silu(first half) * second half`` in FP32, rounded once."""
    gate, up = pair.float().chunk(2, dim=dim)
    return (F.silu(gate) * up).to(pair.dtype)


def native_weighted_sum(product: torch.Tensor) -> torch.Tensor:
    """Per-row FP32 sum for CPU fixtures narrower than the 80 x 64 hidden layout of ``weighted_sum``."""
    return product.float().sum(-1, keepdim=True)


PADDING = 4  # One pad token per routed expert of ``moe_config``.
# 1 + EPS rounds back to 1 in FP32 (ties to even), while EPS + EPS = 2 * EPS survives;
# whether two EPS terms meet before the 1 therefore exposes the pairing order.
EPS = 2.0 ** -24
ONE_PLUS_TWO_EPS = 1.0 + 2.0 ** -23


def moe_config(hidden_size: int = 5120) -> JTDeepseekV3Config:
    """Four routed experts, top-2, a 2.5 routing scale and the 5120 hidden width of ``weighted_sum``."""
    # The Transformers config signature is generated dynamically; keyword checks are false positives here.
    return JTDeepseekV3Config(  # pylint: disable=unexpected-keyword-arg
        hidden_size=hidden_size, moe_intermediate_size=4, n_routed_experts=4, n_shared_experts=1,
        num_experts_per_tok=2, n_group=1, topk_group=1, norm_topk_prob=True, routed_scaling_factor=2.5,
        moe_aux_loss_coeff=0.01, use_pad_tokens=True)


def biased_moe() -> tuple[JTDeepseekV3MoE, torch.Tensor, torch.Tensor]:
    """Build an MoE whose logits are the first four hidden features and whose bias favours expert 3.

    Returns:
        The module, the pad-prefixed FP32 hidden states of three tokens and their router logits.
    """
    moe = JTDeepseekV3MoE(moe_config())
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0], [-1.0, 0.5, 1.5, 0.0], [0.25, -0.5, 1.0, 2.0]])
    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.weight[:, :4] = torch.eye(4)
        moe.gate.e_score_correction_bias.copy_(torch.tensor([0.0, 0.0, 0.0, 1.0]))
    hidden = torch.zeros(1, PADDING + 3, 5120)
    hidden[0, PADDING:, :4] = logits
    return moe, hidden, logits


def scaled_moe(seed: int) -> tuple[JTDeepseekV3MoE, torch.Tensor]:
    """Build an MoE with random FP32 routing, BF16 shared experts and distinct BF16 expert weights.

    Returns:
        The module and one BF16 weight vector per routed expert, ``[4, 5120]``.
    """
    torch.manual_seed(seed)
    moe = JTDeepseekV3MoE(moe_config())
    moe.shared_experts.to(torch.bfloat16)
    with torch.no_grad():
        moe.gate.weight.normal_(std=0.01)
    return moe, (torch.rand(4, 5120) + 0.5).bfloat16()


def scaled_experts(scales: torch.Tensor) -> Any:
    """Return a CPU stand-in for ``forward_expert_major`` in which expert ``e`` scales its rows by ``scales[e]``.

    Elementwise experts make each row's result independent of batching, so per-token references match bitwise.
    """

    def forward_expert_major(inputs: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Scale each expert's contiguous block of expert-major rows by that expert's weights."""
        return torch.cat([rows * scale for rows, scale in zip(inputs.split(counts.tolist()), scales)])

    return forward_expert_major


class TestMoEArithmetic(unittest.TestCase):
    """Check the router, dense MLP and MoE arithmetic against independent references."""

    def setUp(self) -> None:
        """Serve the fused NPU SwiGLU from its CPU stand-in."""
        self.enterContext(patch("hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3.torch_npu.npu_swiglu",
                                new=cpu_swiglu))

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
    def test_weighted_sum_follows_80_row_half_tree_then_native_64_lane_sum(self):
        """Hidden position r * 64 + l is row r of lane l: rows meet by halves (locked, archived order);
        lanes sum natively (issue #252 arm R10 found the archived adjacent-pair lane tree numerically
        inert for this layout). Native torch sum isn't naive left-to-right accumulation either, so this
        same two-EPS-before-the-1 construction still survives across lanes, not just across rows."""
        rows = torch.zeros(1, 5120, dtype=torch.bfloat16)
        rows[0, [0, 64, 192]] = torch.tensor([1.0, EPS, EPS], dtype=torch.bfloat16)  # lane 0, rows 0, 1, 3
        self.assertEqual(weighted_sum(rows).item(), ONE_PLUS_TWO_EPS)
        lanes_only = torch.zeros(1, 5120, dtype=torch.bfloat16)
        lanes_only[0, [0, 1, 3]] = torch.tensor([1.0, EPS, EPS], dtype=torch.bfloat16)  # row 0, lanes 0, 1, 3
        self.assertEqual(weighted_sum(lanes_only).item(), 1.0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_weighted_sum_matches_double_precision_and_includes_every_position(self):
        """Random rows stay within 13 FP32 additions of FP64; every hidden position contributes once."""
        torch.manual_seed(255)
        product = torch.randn(6, 5120).bfloat16()
        actual = weighted_sum(product)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.shape, (6, 1))
        gamma = 13 * 2.0 ** -24 / (1 - 13 * 2.0 ** -24)
        bound = gamma * product.double().abs().sum(-1).max().item()
        torch.testing.assert_close(actual.double(), product.double().sum(-1, keepdim=True), rtol=0, atol=bound)
        torch.testing.assert_close(weighted_sum(torch.eye(5120, dtype=torch.bfloat16)), torch.ones(5120, 1),
                                   rtol=0, atol=0)

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
    def test_mlp_interleaves_gate_and_up_rows_like_the_hf_mlp(self):
        """The fused gate/up GEMM and its gradients match the HF down(silu(gate(x)) * up(x)) math."""
        torch.manual_seed(204)
        mlp = JTDeepseekV3MLP(moe_config(hidden_size=16), intermediate_size=8)
        parameters = (mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight)
        inputs, upstream = torch.randn(5, 16), torch.randn(5, 16)
        results = []
        for fused in (True, False):
            mlp.zero_grad(set_to_none=True)
            values = inputs.clone().requires_grad_()
            output = mlp(values) if fused else mlp.down_proj(F.silu(mlp.gate_proj(values)) * mlp.up_proj(values))
            output.backward(upstream)
            results.append([output.detach(), values.grad] + [parameter.grad for parameter in parameters])
        for actual, expected in zip(*results):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_mtp_layers_select_the_mtp_rounding_points(self):
        """Only the MTP MoE and its shared experts round like MTP; the trunk MoE and dense MLP use the main path."""
        model = JTDeepseekV3ForCausalLM(small_config())
        dense, trunk = model.model.layers[0].mlp, model.model.layers[1].mlp
        prediction = model.mtp.layers[0].transformer_layer.mlp
        self.assertIsInstance(dense, JTDeepseekV3MLP)
        self.assertEqual((dense.mtp, trunk.mtp, trunk.shared_experts.mtp), (False, False, False))
        self.assertEqual((prediction.mtp, prediction.shared_experts.mtp), (True, True))

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


def small_config() -> JTDeepseekV3Config:
    """Build a CPU-sized fixture without changing the production validation recipe."""
    config = JTDeepseekV3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, moe_intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2, n_group=1, topk_group=1,
        q_lora_rank=8, kv_lora_rank=8, qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=4,
        mlp_layer_types=["dense", "sparse"], max_position_embeddings=32, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        architectures=["JTDeepseekV3ForCausalLM"],
        num_nextn_predict_layers=1, use_pad_tokens=True, norm_topk_prob=True,
        routed_scaling_factor=1.0, moe_aux_loss_coeff=0.01, mtp_loss_factor=0.3)
    config.rope_interleave = True
    return config


class TestCompleteModel(unittest.TestCase):
    """Exercise model semantics without a builder, EP adapter or replacement pass."""

    def test_auxiliary_loss_pop_preserves_router_and_input_gradients(self):
        """Consuming the loss clears its owner but preserves the differentiable objective."""
        torch.manual_seed(17)
        model = JTDeepseekV3ForCausalLM(small_config())
        moe = model.model.layers[1].mlp
        hidden = torch.randn(1, moe.padding + 8, 16, requires_grad=True)
        moe.route(hidden)
        reference = moe.auxiliary_loss
        loss = moe.pop_auxiliary_loss()
        self.assertIs(loss, reference)
        self.assertIsNone(moe.auxiliary_loss)
        parameters = (hidden, moe.gate.weight)
        expected = torch.autograd.grad(reference, parameters, retain_graph=True)
        actual = torch.autograd.grad(loss, parameters)
        for gradient, baseline in zip(actual, expected, strict=True):
            torch.testing.assert_close(gradient, baseline, rtol=0, atol=0)
            self.assertGreater(gradient.abs().sum().item(), 0)
        with self.assertRaisesRegex(RuntimeError, "No auxiliary loss"):
            moe.pop_auxiliary_loss()

    @patch("hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3.weighted_sum", new=native_weighted_sum)
    @patch("hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3.torch_npu.npu_swiglu", new=cpu_swiglu)
    @patch("hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3.npu_grouped_swiglu")
    def test_checkpoint_auxiliary_loss_consumption_matches_eager_gradients(self, grouped_swiglu):
        """Trunk and MTP replay preserve accumulated gradients without leaving loss references."""
        # Use differentiable CPU experts to exercise graph lifetime without the NPU-only kernel.
        def cpu_experts(inputs, gate_up_weight, down_weight, counts):
            outputs = []
            for index, tokens in enumerate(inputs.split(counts.tolist())):
                gate, up = F.linear(tokens, gate_up_weight[index]).chunk(2, dim=-1)
                outputs.append(F.linear(F.silu(gate) * up, down_weight[index]))
            return torch.cat(outputs)

        grouped_swiglu.side_effect = cpu_experts
        torch.manual_seed(19)
        reference = JTDeepseekV3ForCausalLM(small_config())
        candidate = copy.deepcopy(reference)
        for index, layer in enumerate(candidate.model.layers):
            candidate.model.layers[index] = checkpoint_wrapper(layer)
        for layer in candidate.mtp.layers:
            layer.transformer_layer = checkpoint_wrapper(layer.transformer_layer)
        owners = [module for module in candidate.modules() if isinstance(module, JTDeepseekV3MoE)]
        self.assertEqual(len(owners), 2)
        for objective in ("aux", "total"):
            with self.subTest(objective=objective):
                reference.zero_grad(set_to_none=True)
                candidate.zero_grad(set_to_none=True)
                for micro_step in range(3):
                    tokens = (torch.arange(8).unsqueeze(0) + micro_step) % 32
                    expected = reference(tokens, (tokens + 1) % 32).loss
                    actual = candidate(tokens, (tokens + 1) % 32).loss
                    for name, value in expected.items():
                        torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
                    self.assertTrue(all(module.auxiliary_loss is None for module in owners))
                    expected_loss = expected["foundation_loss/aux"] if objective == "aux" else sum(expected.values())
                    actual_loss = actual["foundation_loss/aux"] if objective == "aux" else sum(actual.values())
                    (expected_loss / 3).backward()
                    (actual_loss / 3).backward()
                    self.assertTrue(all(module.auxiliary_loss is None for module in owners))
                for original, replayed in zip(reference.parameters(), candidate.parameters(), strict=True):
                    if original.grad is None:
                        self.assertIsNone(replayed.grad)
                    else:
                        self.assertIsNotNone(replayed.grad)
                        torch.testing.assert_close(replayed.grad, original.grad, rtol=1e-5, atol=1e-7)
                self.assertGreater(candidate.model.embed_tokens.weight.grad.abs().sum().item(), 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_standalone_forward_backward(self):
        """Feature: Complete model.

        Description: Construct directly while replacement compilation is forbidden.
        Expectation: Trunk, MTP and router execute and backpropagate without an EP adapter.
        """
        torch.manual_seed(11)
        with patch("hyper_parallel.models.replacement.compile_module_replacements", side_effect=AssertionError):
            model = JTDeepseekV3ForCausalLM(small_config())
        self.assertIsInstance(model.model.layers[0], JTDeepseekV3Decoder)
        self.assertIsInstance(model.model.layers[1].mlp, JTDeepseekV3MoE)
        self.assertEqual(model.model.layers[1].mlp.ep_compute, model.model.layers[1].mlp.local_routed_forward)
        self.assertIsInstance(model.mtp.layers[0], MultiTokenPredictionLayer)
        tokens = torch.arange(8).unsqueeze(0)
        output = model(tokens, (tokens + 1) % 32)
        self.assertEqual(
            set(output.loss),
            {"foundation_loss/lm", "foundation_loss/mtp", "foundation_loss/aux"},
        )
        total_loss = sum(output.loss.values())
        self.assertTrue(torch.isfinite(total_loss))
        total_loss.backward()
        for name in ["model.embed_tokens.weight", "model.layers.1.mlp.gate.weight", "mtp.layers.0.eh_proj.weight"]:
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_standard_policy_bfloat16_forward_closes_activations(self):
        """Feature: Framework BF16 policy.

        Description: Emulate FSDP param_dtype=bfloat16 compute copies over FP32 storage; buffers stay as stored.
        Expectation: Activations stay BF16 without FSDP input casts; losses and the expert bias stay FP32. Trunk
            and MTP router logits and auxiliary-loss scores are FP32, computed from the upcast BF16 router input
            and BF16 weight copy.
        """
        model = JTDeepseekV3ForCausalLM(small_config())
        for parameter in model.parameters():
            parameter.data = parameter.data.to(torch.bfloat16)
        biases = [
            module.e_score_correction_bias
            for module in model.modules()
            if hasattr(module, "e_score_correction_bias")
        ]
        self.assertTrue(biases)
        self.assertTrue(all(bias.dtype == torch.float32 for bias in biases))

        activations = {}
        routers = {}

        def capture(name: str) -> Any:
            """Record a selected module output by name."""

            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                """Store the tensor output after tuple unwrapping."""
                output = output[0] if isinstance(output, tuple) else output
                activations[name] = output

            return hook

        def capture_router(name: str) -> Any:
            """Record a router input and its logits by router name."""

            def hook(_module: Any, inputs: Any, output: Any) -> None:
                """Store the router input and the returned logits."""
                routers[name] = (inputs[0], output)

            return hook

        gates = {name: module for name, module in model.named_modules() if isinstance(module, JTDeepseekV3Gate)}
        handles = [
            model.model.layers[0].self_attn.register_forward_hook(capture("attention")),
            model.model.layers[1].mlp.register_forward_hook(capture("moe")),
            model.model.layers[1].register_forward_hook(capture("decoder")),
            *(gate.register_forward_hook(capture_router(name)) for name, gate in gates.items()),
        ]
        try:
            tokens = torch.arange(8).unsqueeze(0)
            with torch.no_grad(), patch.object(RoutingNormalization, "apply",
                                               wraps=RoutingNormalization.apply) as normalization:
                output = model(tokens, (tokens + 1) % 32)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual({name: value.dtype for name, value in activations.items()},
                         {"attention": torch.bfloat16, "moe": torch.float32, "decoder": torch.float32})
        self.assertTrue(all(value.dtype == torch.float32 for value in output.loss.values()))
        self.assertEqual(list(routers), ["model.layers.1.mlp.gate", "mtp.layers.0.transformer_layer.mlp.gate"])
        scores = [call.args[0] for call in normalization.call_args_list]
        for (name, (router_input, logits)), router_scores in zip(routers.items(), scores, strict=True):
            with self.subTest(router=name):
                weight = gates[name].weight
                self.assertEqual((router_input.dtype, weight.dtype), (torch.float32, torch.bfloat16))
                self.assertEqual(logits.dtype, torch.float32)
                expected = F.linear(  # pylint: disable=not-callable
                    router_input.reshape(-1, router_input.shape[-1]).float(), weight.float())
                self.assertTrue(torch.equal(logits, expected), f"{name} logits differ from the FP32 product")
                self.assertEqual(router_scores.dtype, torch.float32)
                self.assertTrue(torch.equal(router_scores, logits.sigmoid()),
                                f"{name} auxiliary-loss scores differ from the router logits")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_standard_policy_bfloat16_rotates_rope_in_fp32(self):
        """Feature: RoPE precision under the framework BF16 policy.

        Description: Emulate FSDP param_dtype=bfloat16 compute copies; capture every trunk and MTP rotary call.
        Expectation: The position tables are FP32, and every rotated output equals the FP32 rotation of the
            BF16 channel pairs rounded to BF16 once, as MindFormers rotates with rotary_dtype=float32.
        """
        model = JTDeepseekV3ForCausalLM(small_config())
        for parameter in model.parameters():
            parameter.data = parameter.data.to(torch.bfloat16)
        calls = []

        def capture(_module: Any, inputs: Any, output: torch.Tensor) -> None:
            """Store the rotary channels, the position tables and the rotated output."""
            calls.append((*inputs, output))

        handles = [module.register_forward_hook(capture) for module in model.modules()
                   if isinstance(module, JTDeepseekV3RotaryEmbedding)]
        try:
            tokens = torch.arange(8).unsqueeze(0)
            with torch.no_grad():
                model(tokens, (tokens + 1) % 32)
        finally:
            for handle in handles:
                handle.remove()
        # Query and key rotations in both trunk layers and in the MTP layer.
        self.assertEqual(len(calls), 6)
        for values, cos, sin, output in calls:
            self.assertEqual((values.dtype, cos.dtype, sin.dtype, output.dtype),
                             (torch.bfloat16, torch.float32, torch.float32, torch.bfloat16))
            even, odd = values[..., ::2].float(), values[..., 1::2].float()
            half = even.shape[-1]
            cos, sin = cos[..., :half].unsqueeze(1), sin[..., :half].unsqueeze(1)
            expected = torch.cat((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
            self.assertTrue(torch.equal(output, expected.to(torch.bfloat16)))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_replacement_selects_attention(self):
        """Feature: Acceleration boundary.

        Description: Apply the optional high-performance attention replacement.
        Expectation: Attention changes while decoder, MLP, norm and MTP weights survive.
        """
        model = JTDeepseekV3ForCausalLM(small_config())
        previous = dict(model.named_modules())
        self.assertEqual(sum(isinstance(m, JTDeepseekV3Attention) for m in previous.values()), 3)
        original_q = model.model.layers[0].self_attn.q_a_proj.weight.detach().clone()
        original_kv = model.model.layers[0].self_attn.kv_a_proj_with_mqa.weight.detach().clone()
        recipe = jt_test_config()
        rules = entries_to_module_replacements(recipe.plan_overrides)
        self.assertEqual(len(rules), 1)
        plan = compile_module_replacements(model, rules)
        apply_module_replacements(model, plan, weights_mapping=[])
        self.assertTrue(torch.equal(model.model.layers[0].self_attn.linear_qkv.weight,
                                    torch.cat((original_q, original_kv))))
        current = dict(model.named_modules())
        self.assertEqual(sum(isinstance(m, JTDeepseekV3MLAAttention) for m in current.values()), 3)
        for name in ["model.layers.0", "model.layers.1.mlp", "model.norm", "mtp.layers.0"]:
            self.assertIs(current[name], previous[name])
        self.assertIs(model.model.layers[0].self_attn.q_a_layernorm,
                      previous["model.layers.0.self_attn.q_a_layernorm"])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_meta_replacement_loads_converted_reference_state(self):
        """Feature: Offline checkpoint loading.

        Description: Load arrays that already use the model's final parameter layout and differ from its state.
        Expectation: Every materialized parameter and buffer holds the loaded values exactly.
        """
        config = small_config()
        rules = entries_to_module_replacements(jt_test_config().plan_overrides)
        with torch.device("meta"):
            candidate = JTDeepseekV3ForCausalLM(config)
            plan = compile_module_replacements(candidate, rules)
            apply_module_replacements(candidate, plan, weights_mapping=[])
        candidate.to_empty(device="cpu")
        generator = torch.Generator().manual_seed(0)
        arrays = {name: torch.randn(value.shape, generator=generator).to(value.dtype).numpy()
                  for name, value in candidate.state_dict().items()}
        _load_reference_state(candidate, arrays)
        for name, value in candidate.state_dict().items():
            self.assertEqual(value.detach().numpy().tobytes(), arrays[name].tobytes(), name)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_family_registration_does_not_replace_standard_deepseek(self):
        """Feature: Independent family registration.

        Description: Resolve both families and the custom architecture identity.
        Expectation: JT owns its spec; the original V3/V2 identities stay registered independently.
        """
        standard = get_model_adapter("deepseek_v3")
        custom = get_model_adapter("jt_deepseek_v3")
        self.assertIs(get_model_adapter("JTDeepseekV3ForCausalLM"), custom)
        self.assertIsNot(standard, custom)
        self.assertEqual(standard.architecture, "DeepseekV3ForCausalLM")
        self.assertEqual(custom.model_type, "jt_deepseek_v3")
        self.assertEqual(small_config().model_type, "jt_deepseek_v3")
        self.assertIs(get_model_adapter(small_config().architectures[0]), custom)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_fsdp_discovery_uses_jt_identity_and_declared_mtp_unit(self):
        """Feature: FSDP unit discovery.

        Description: Resolve FSDP units for the JT model through the framework manager.
        Expectation: The JT adapter applies; each MTP transformer layer is one unit, not split leaves.
        """
        model = JTDeepseekV3ForCausalLM(small_config())
        self.assertEqual(FSDP2Manager._get_model_adapter_spec(model).model_type, "jt_deepseek_v3")
        manager = FSDP2Manager(FSDP2Config(), SimpleNamespace(fsdp_moe_mesh=None))
        self.assertEqual(
            sorted(unit.fqn for unit in manager._find_wrap_modules(model)),
            ["model.layers.0", "model.layers.1", "mtp.layers.0.transformer_layer"],
        )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_native_config_roundtrip(self):
        """Feature: Native model configuration.

        Description: Build and serialize native JT configuration without an example YAML.
        Expectation: Model dimensions, JT options and independent adapter identity survive.
        """
        recipe = jt_test_config()
        config = JTDeepseekV3Config(**recipe.model.config)
        self.assertIs(type(config), JTDeepseekV3Config)
        self.assertFalse(hasattr(recipe.model, "reference_yaml"))
        self.assertFalse(hasattr(config, "jt_config"))
        restored = JTDeepseekV3Config.from_dict(config.to_dict())
        self.assertEqual(restored.to_dict(), config.to_dict())
        self.assertEqual(restored.mlp_layer_types, ["dense", "sparse"])
        self.assertEqual(restored.num_nextn_predict_layers, 1)
        self.assertEqual(restored.mtp_loss_factor, 0.3)
        restored.n_group = 2
        with self.assertRaisesRegex(ValueError, "n_group=topk_group=1"):
            JTDeepseekV3ForCausalLM(restored)
        self.assertEqual(restored.moe_aux_loss_coeff, 0.0001)
        self.assertEqual(restored.rope_parameters["rope_theta"], 5000000)
        self.assertIs(get_model_adapter(restored.architectures[0]), get_model_adapter("jt_deepseek_v3"))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_shared_batch_inputs_define_lm_and_mtp_objectives(self):
        """Feature: Native Trainer input and objective contract.

        Description: Forward the shared text batch's model inputs unchanged; the final target is the valid ID 0.
        Expectation: LM and MTP losses equal dense CE over the batch mask; MTP has no target past the end.
        """
        torch.manual_seed(5)
        model = JTDeepseekV3ForCausalLM(small_config())
        tokens = torch.arange(8).unsqueeze(0)
        labels = torch.roll(tokens, -1, 1)
        labels[0, [2, 5]] = -100
        mesh = SimpleNamespace(cp_size=1, pp_size=1, tp_size=1, dp_size=1, dp_rank=0, device_mesh=None)
        batch = TextParallelBatch(mesh, torch.device("cpu"), None, {}, False,
                                  source_type="indexed", attention_mode="compressed")
        model_inputs, loss_inputs = batch(iter([{"tokens": tokens, "labels": labels}]))
        head_outputs = []
        handle = model.lm_head.register_forward_hook(lambda _module, _inputs, output: head_outputs.append(output))
        try:
            output = model(**model_inputs, use_cache=False)
        finally:
            handle.remove()
        lm_logits, mtp_logits = (logits[0].detach() for logits in head_outputs)
        targets, mask = loss_inputs["shift_labels"][0], loss_inputs["loss_mask"][0].bool()
        torch.testing.assert_close(output.loss["foundation_loss/lm"], F.cross_entropy(lm_logits[mask], targets[mask]))
        # The single MTP depth predicts each next target, scaled by the MTP loss factor.
        expected_mtp = F.cross_entropy(mtp_logits[:-1][mask[1:]], targets[1:][mask[1:]]) * model.config.mtp_loss_factor
        torch.testing.assert_close(output.loss["foundation_loss/mtp"], expected_mtp)
        with self.assertRaisesRegex(ValueError, "attention_mask=None"):
            model(**(model_inputs | {"attention_mask": torch.ones_like(tokens)}))
        with self.assertRaisesRegex(ValueError, "explicit shift_labels"):
            model(input_ids=tokens, labels=labels)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_row_without_targets_has_zero_lm_and_mtp_losses(self):
        """Feature: Native Trainer objective contract.

        Description: Forward a row whose shifted labels are all ignored, with two MTP depths.
        Expectation: The LM and MTP losses are zero instead of NaN, and every gradient stays finite.
        """
        torch.manual_seed(5)
        config = small_config()
        config.num_nextn_predict_layers = 2
        model = JTDeepseekV3ForCausalLM(config)
        tokens = torch.arange(8).unsqueeze(0)
        losses = model(tokens, torch.full_like(tokens, -100)).loss
        self.assertEqual(losses["foundation_loss/lm"].item(), 0.0)
        self.assertEqual(losses["foundation_loss/mtp"].item(), 0.0)
        sum(losses.values()).backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                with self.subTest(parameter=name):
                    self.assertTrue(torch.isfinite(parameter.grad).all())
