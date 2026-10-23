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

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
import copy
from types import SimpleNamespace
from typing import Any, Optional
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (
    JTDeepseekV3Config, JTDeepseekV3LMHead, JTDeepseekV3RMSNorm, head_weight_gradient,
    JTDeepseekV3ForCausalLM, JTDeepseekV3Decoder, JTDeepseekV3Gate, JTDeepseekV3MoE,
    JTDeepseekV3Attention, JTDeepseekV3MLAAttention, JTDeepseekV3RotaryEmbedding,
    SharedKeyHalfTree,
)
from hyper_parallel.components.losses import calculate_seq_aux_loss
from hyper_parallel.components.modules.mtp import MultiTokenPredictionLayer
from hyper_parallel.core.activation_memory import checkpoint_wrapper
from hyper_parallel.data.batching import TextParallelBatch
from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.models.build_options import FSDP2Config
from hyper_parallel.models.replacement import compile_module_replacements, apply_module_replacements
from hyper_parallel.models.jt_deepseek_v3.adapter.jt_builder import _load_reference_state
from hyper_parallel.models.registry import get_model_adapter
from hyper_parallel.trainer.config import entries_to_module_replacements
from tests.common.jt_config import cpu_npu_kernels, cpu_rms_norm, jt_test_config
from tests.common.mark_utils import arg_mark


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


def mla_rms_norm(values: torch.Tensor, weight: torch.Tensor, *,
                 epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU stand-in for ``torch_npu.npu_rms_norm`` with the HF RMSNorm's FP32 formula."""
    rstd = torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + epsilon)
    return values * rstd * weight, rstd


def mla_source_attention(seed: int) -> JTDeepseekV3Attention:
    """Build a materialized four-head source MLA whose latent norm weights differ from their initial ones."""
    config = small_config()
    config.num_attention_heads = config.num_key_value_heads = 4
    torch.manual_seed(seed)
    source = JTDeepseekV3Attention(config, 0)
    with torch.no_grad():
        for name in ("q_a_layernorm.weight", "kv_a_layernorm.weight"):
            source.get_parameter(name).uniform_(0.5, 1.5)
    return source


class TestSharedKeyHalfTree(unittest.TestCase):
    """The shared rope key expands over heads and reduces their gradients by an FP32 half tree."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_shared_key_expands_heads_not_tokens(self):
        """The MLA projection contract is token-major, including its backward."""
        values = torch.arange(30, dtype=torch.float32).reshape(2, 5, 1, 3).requires_grad_()
        actual = SharedKeyHalfTree.apply(values, 8)
        expected = values.expand(2, 5, 8, 3)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        weights = torch.arange(actual.numel(), dtype=torch.float32).reshape_as(actual)
        (actual * weights).sum().backward()
        torch.testing.assert_close(values.grad, weights.sum(2, keepdim=True), rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_shared_key_sums_head_gradients_by_fp32_half_tree(self):
        """Head j meets head j + n / 2 first, in FP32, and the sum is rounded once to the key dtype."""
        # (1e8 - 1e8) + (1 + 1) = 2, whereas a left-to-right FP32 sum loses each 1 against 1e8 (spacing 8) and
        # gives 1, and pairing neighbours gives 0.
        values = torch.zeros(1, 1, 1, 1, requires_grad=True)
        SharedKeyHalfTree.apply(values, 4).backward(torch.tensor([1e8, 1.0, -1e8, 1.0]).reshape(1, 1, 4, 1))
        self.assertEqual(values.grad.item(), 2.0)
        # BF16 heads: (1 + 2**-8) + (2**-8 + 0) = 1 + 2**-7 in FP32, whereas BF16 partial sums tie 1 + 2**-8
        # back to 1 and then 1 + 2**-8 back to 1 again.
        values = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16, requires_grad=True)
        gradient = torch.tensor([1.0, 2.0 ** -8, 2.0 ** -8, 0.0], dtype=torch.bfloat16).reshape(1, 1, 4, 1)
        SharedKeyHalfTree.apply(values, 4).backward(gradient)
        self.assertEqual(values.grad.item(), 1.0 + 2.0 ** -7)


class TestMLAAttentionNumerics(unittest.TestCase):
    """The MLA replacement keeps the source FP32 values and changes only rounding and summation order."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_fp32_projection_matches_source_attention_bitwise(self):
        """From the same materialized source, FP32 query, key and value equal the source attention's bitwise."""
        source = mla_source_attention(5)
        attention = JTDeepseekV3MLAAttention(module=copy.deepcopy(source))
        hidden = torch.randn(2, 3, 16)
        angle = torch.randn(2, 3, 4)
        position_embeddings = (angle.cos(), angle.sin())
        with patch("torch_npu.npu_rms_norm", mla_rms_norm):
            actual = attention._project_attention_inputs(hidden, position_embeddings, None)
        # On CPU the source attention passes its query, key and value to SDPA.
        with patch.object(F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention) as kernel:
            source(hidden, position_embeddings=position_embeddings)
        for name, expected, value in zip(("query", "key", "value"), kernel.call_args.args, actual, strict=True):
            with self.subTest(name):
                torch.testing.assert_close(value, expected, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_latent_projection_runs_the_two_source_gemms(self):
        """The fused latent weight runs as one GEMM over the Q rows and one over the KV-latent and rope rows."""
        attention = JTDeepseekV3MLAAttention(module=mla_source_attention(9))
        hidden = torch.randn(2, 3, 16)
        with patch("torch_npu.npu_rms_norm", mla_rms_norm), patch.object(F, "linear", wraps=F.linear) as gemm:
            attention.project_latent_inputs(hidden)
        weights = [call.args[1] for call in gemm.call_args_list]
        self.assertEqual(len(weights), 2)
        for rows, weight in zip((slice(0, 8), slice(8, 20)), weights, strict=True):
            torch.testing.assert_close(weight, attention.linear_qkv.weight[rows], rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_latent_norms_round_fp32_rms_with_source_weights_once(self):
        """BF16 latents are the FP32 RMSNorm with the source weights, BF16 or FP32, rounded once to BF16."""
        source = mla_source_attention(7)
        hidden = torch.randn(2, 8, 16, dtype=torch.bfloat16)
        names = ("q_a_layernorm", "kv_a_layernorm")
        for weight_dtype in (torch.bfloat16, torch.float32):
            attention = JTDeepseekV3MLAAttention(module=copy.deepcopy(source))
            for name, parameter in attention.named_parameters():
                parameter.data = parameter.data.to(weight_dtype if "layernorm" in name else torch.bfloat16)
            inputs = {}
            for name in names:
                attention.get_submodule(name).register_forward_hook(
                    lambda _module, args, _output, name=name: inputs.__setitem__(name, args[0]))
            with patch("torch_npu.npu_rms_norm", mla_rms_norm):
                latents = attention.project_latent_inputs(hidden)[:2]
            for name, latent in zip(names, latents, strict=True):
                with self.subTest(name=name, weight_dtype=weight_dtype):
                    weight = source.get_parameter(f"{name}.weight").to(weight_dtype).double()
                    expected = mla_rms_norm(inputs[name].double(), weight,
                                            epsilon=source.config.rms_norm_eps)[0].to(torch.bfloat16)
                    torch.testing.assert_close(latent, expected, rtol=0, atol=0)
                    if weight_dtype == torch.bfloat16:
                        # The source HF norm rounds the normalized value to BF16 before the weight multiply.
                        source_norm = copy.deepcopy(source.get_submodule(name)).to(torch.bfloat16)
                        self.assertFalse(torch.equal(source_norm(inputs[name]), expected))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_shared_rope_key_gradient_reaches_projection_as_half_tree_sum(self):
        """Per-head rope-key gradients [1e8, 1, -1e8, 1] reach the latent projection as their half-tree sum 2."""
        attention = JTDeepseekV3MLAAttention(module=mla_source_attention(3))
        hidden = torch.ones(1, 1, 16)
        no_rotation = (torch.ones(1, 1, 4), torch.zeros(1, 1, 4))
        with patch("torch_npu.npu_rms_norm", mla_rms_norm):
            _, key, _ = attention._project_attention_inputs(hidden, no_rotation, None)
        gradient = torch.zeros_like(key)
        gradient[..., 4:] = torch.tensor([1e8, 1.0, -1e8, 1.0]).reshape(1, 4, 1, 1)
        key.backward(gradient)
        # The rope rows are the last four of the fused latent projection; a plain head sum would give 1.
        torch.testing.assert_close(attention.linear_qkv.weight.grad[-4:], torch.full((4, 16), 2.0), rtol=0, atol=0)


class TestCompleteModel(unittest.TestCase):
    """Exercise model semantics without a builder, EP adapter or replacement pass."""

    def setUp(self) -> None:
        """Run the NPU-only RMSNorm and output-head weight-gradient kernels as their CPU stand-ins."""
        self.enterContext(cpu_npu_kernels())

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

    def test_sequence_auxiliary_loss_ignores_dispatch_bias_and_group_filter(self) -> None:
        """The sequence objective uses raw affinities while bias counts track actual dispatch."""
        logits = torch.tensor([[2, 1, -1, -2], [1.8, .9, -.8, -1.5], [-.5, 1.3, 2, -1]])
        scores = logits.double().sigmoid()
        # These fixed logits select {0, 1}, {0, 1}, {1, 2} without dispatch corrections.
        load = torch.tensor([2, 3, 1, 0], dtype=torch.float64) / 6
        denominator = scores.sum(-1, keepdim=True)
        projection = (scores * load).sum(-1, keepdim=True)
        coefficient = 1e-4
        expected_loss = coefficient * 4 * (load * (scores / denominator).mean(0)).sum()
        expected_gradient = coefficient * 4 / 3 * scores * (1 - scores) * (
            load * denominator - projection) / denominator.square()

        for groups in (1, 2):
            for use_padding in (False, True):
                with self.subTest(groups=groups, use_padding=use_padding):
                    config = small_config()
                    config.n_group = groups
                    config.topk_group = 1
                    config.use_pad_tokens = use_padding
                    config.moe_aux_loss_coeff = coefficient
                    module = JTDeepseekV3MoE(config)
                    with torch.no_grad():
                        module.gate.weight.zero_()
                        module.gate.weight[:, :4].copy_(torch.eye(4))
                    dispatches = []
                    loads = []
                    for bias in (torch.zeros(4), torch.tensor([-1., -1., 1., 1.])):
                        module.tokens_per_expert = None
                        with torch.no_grad():
                            module.gate.e_score_correction_bias.copy_(bias)
                        hidden = torch.zeros(1, module.padding + 3, 16, requires_grad=True)
                        with torch.no_grad():
                            hidden[0, module.padding:, :4].copy_(logits)
                        indices, weights = module.route(hidden)
                        loss = module.pop_auxiliary_loss()
                        hidden_gradient, weight_gradient = torch.autograd.grad(loss, (hidden, module.gate.weight))
                        torch.testing.assert_close(loss.double(), expected_loss, rtol=0, atol=1e-10)
                        torch.testing.assert_close(hidden_gradient[0, module.padding:, :4].double(),
                                                   expected_gradient, rtol=0, atol=1e-11)
                        expected_weight = expected_gradient.T @ hidden.detach()[0, module.padding:].double()
                        torch.testing.assert_close(weight_gradient.double(), expected_weight, rtol=0, atol=5e-11)
                        padding_gradient = hidden_gradient[0, :module.padding]
                        torch.testing.assert_close(padding_gradient, torch.zeros_like(padding_gradient), rtol=0, atol=0)
                        selected = indices[module.padding:]
                        counts = torch.bincount(selected.flatten(), minlength=4).float()
                        torch.testing.assert_close(module.tokens_per_expert, counts, rtol=0, atol=0)
                        torch.testing.assert_close(weights[:module.padding],
                                                   torch.zeros_like(weights[:module.padding]), rtol=0, atol=0)
                        uncorrected = logits.sigmoid().gather(-1, selected)
                        expected_weights = uncorrected / uncorrected.sum(-1, keepdim=True)
                        torch.testing.assert_close(weights[module.padding:], expected_weights, rtol=1e-6, atol=1e-7)
                        module.update_expert_bias(lr=.001)
                        torch.testing.assert_close(module.gate.e_score_correction_bias,
                                                   bias + (counts.mean() - counts).sign() * .001, rtol=0, atol=0)
                        dispatches.append(selected.sort(-1).values)
                        loads.append(counts)
                    self.assertFalse(torch.equal(*dispatches),
                                     f"Dispatch must respond to bias: first={dispatches[0]}, second={dispatches[1]}")
                    self.assertFalse(torch.equal(*loads),
                                     f"Bias counts must track dispatch: first={loads[0]}, second={loads[1]}")

    @patch("hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3.npu_grouped_swiglu")
    def test_checkpoint_auxiliary_loss_consumption_matches_eager_gradients(self, grouped_swiglu):
        """Trunk and MTP replay preserve accumulated gradients without leaving loss references."""
        # Use differentiable CPU experts to exercise graph lifetime without the NPU-only kernel.
        def cpu_experts(inputs: torch.Tensor, gate_up_weight: torch.Tensor,
                        down_weight: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
            """Evaluate the expert branches with differentiable CPU operations."""
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
        Expectation: Attention activations stay BF16 without FSDP input casts; the last trunk layer returns FP32;
            losses and the expert bias stay FP32. Trunk and MTP router logits and auxiliary-loss scores are FP32,
            computed from the upcast BF16 router input and BF16 weight copy.
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
            with torch.no_grad(), patch(f"{JTDeepseekV3Gate.__module__}.calculate_seq_aux_loss",
                                        wraps=calculate_seq_aux_loss) as auxiliary_loss:
                output = model(tokens, (tokens + 1) % 32)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual({name: value.dtype for name, value in activations.items()},
                         {"attention": torch.bfloat16, "moe": torch.bfloat16, "decoder": torch.float32})
        self.assertTrue(all(value.dtype == torch.float32 for value in output.loss.values()))
        self.assertEqual(list(routers), ["model.layers.1.mlp.gate", "mtp.layers.0.transformer_layer.mlp.gate"])
        scores = [call.args[0] for call in auxiliary_loss.call_args_list]
        for (name, (router_input, logits)), router_scores in zip(routers.items(), scores, strict=True):
            with self.subTest(router=name):
                weight = gates[name].weight
                self.assertEqual((router_input.dtype, weight.dtype), (torch.bfloat16, torch.bfloat16))
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
        for name in ("q_a_layernorm", "kv_a_layernorm"):
            norm = model.model.layers[0].self_attn.get_submodule(name)
            self.assertIsInstance(norm, JTDeepseekV3RMSNorm)
            self.assertTrue(norm.keep_compute_in_fp32)
            self.assertIs(norm.weight, previous[f"model.layers.0.self_attn.{name}"].weight)

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
        # The final norm rounds the head input to BF16; the loss upcasts the BF16 logits.
        torch.testing.assert_close(output.loss["foundation_loss/lm"],
                                   F.cross_entropy(lm_logits[mask].float(), targets[mask]))
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


class SetterOnlyConfig:
    """TorchNPU 2.12 ``npu.config``: ``allow_internal_format`` can be written but not read back."""

    def __init__(self, runtime: "FakeNpuRuntime") -> None:
        """Forward every write to ``runtime``."""
        self._runtime = runtime

    def _write(self, enabled: bool) -> None:
        self._runtime.set_internal_format(enabled)

    allow_internal_format = property(fset=_write)


class FakeNpuRuntime:
    """Process-global TorchNPU switches read and written around the head weight-gradient GEMM."""

    def __init__(self, *, jit_compile: bool = False, internal_option: Optional[bytes] = b"disable") -> None:
        """Start from the caller's compile mode and internal-format option."""
        self.jit_compile = jit_compile
        self.internal_option = internal_option
        self.npu = SimpleNamespace(is_jit_compile_false=lambda: not self.jit_compile,
                                   set_compile_mode=self.set_compile_mode, config=SetterOnlyConfig(self),
                                   device=lambda _: nullcontext())
        self._C = SimpleNamespace(_npu_getOption=self.get_option)

    def state(self) -> tuple[bool, Optional[bytes]]:
        """Return the compile mode and the backing internal-format option."""
        return self.jit_compile, self.internal_option

    def get_option(self, name: str) -> Optional[bytes]:
        """Read a backing runtime option; only the internal-format option is modelled."""
        if name != "ALLOW_INTERNAL_FORMAT":
            raise KeyError(name)
        return self.internal_option

    def set_compile_mode(self, *, jit_compile: bool) -> None:
        """Switch JIT compilation for the whole process."""
        self.jit_compile = jit_compile

    def set_internal_format(self, enabled: bool) -> None:
        """Back the setter-only ``npu.config.allow_internal_format`` attribute."""
        self.internal_option = b"enable" if enabled else b"disable"


class RecordingKernel:
    """NZ GEMM stand-in returning BF16 ``gradient.T @ inputs`` and recording the switches it ran under."""

    def __init__(self, runtime: FakeNpuRuntime, error: Optional[Exception] = None) -> None:
        """Observe ``runtime``; raise ``error`` instead of computing when it is given."""
        self.runtime = runtime
        self.error = error
        self.seen = []

    def __call__(self, gradient: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Record the runtime switches, then fail or return the product rounded once to BF16."""
        self.seen.append(self.runtime.state())
        if self.error is not None:
            raise self.error
        return (gradient.double().T @ inputs.double()).to(torch.bfloat16)


@contextmanager
def npu_head_gemm(runtime: FakeNpuRuntime, kernel: RecordingKernel) -> Iterator[None]:
    """Install ``runtime`` as TorchNPU (``torch.npu`` is ``torch_npu.npu``) and ``kernel`` as the NZ GEMM."""
    module = JTDeepseekV3LMHead.__module__
    with patch(f"{module}.torch_npu", runtime), patch.object(torch, "npu", runtime.npu, create=True), \
            patch(f"{module}._nz_matmul", kernel):
        yield


class TestRMSNormNumerics(unittest.TestCase):
    """RMSNorm arithmetic stays FP32 until its declared boundary; decoder layers join and round once."""

    def setUp(self) -> None:
        """Run the NPU-only RMSNorm kernels as their CPU stand-ins."""
        self.enterContext(cpu_npu_kernels())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_residual_norm_rounds_once_after_joining_both_input_gradients(self):
        """The FP32 RMSNorm and residual input derivatives are added before the single BF16 rounding."""
        residual_gradient = 2.0 ** -8
        values = torch.tensor([[1.0, -1.0, 1.0, -1.0]], dtype=torch.bfloat16, requires_grad=True)
        weight = torch.full((4,), 1.0 + 2.0 ** -8, dtype=torch.float64, requires_grad=True)
        norm = JTDeepseekV3RMSNorm(4, 0.0, residual=True)
        with torch.no_grad():
            norm.weight.copy_(weight)
        normalized, residual = norm(values)
        (normalized.float().sum() + residual.float().mul(residual_gradient).sum()).backward()
        # Unit RMS makes the norm derivative exactly 1 + 2**-8, so the joined derivative 1 + 2**-7 is a BF16
        # value; rounding the norm derivative first ties it to 1, and adding 2**-8 in BF16 ties back to 1.
        reference = values.detach().double().requires_grad_()
        norm_part, weight_gradient = torch.autograd.grad(
            cpu_rms_norm(reference, weight, epsilon=0.0)[0].sum(), (reference, weight))
        residual_part = torch.full_like(norm_part, residual_gradient)
        joined = (norm_part + residual_part).to(torch.bfloat16)
        rounded_first = norm_part.to(torch.bfloat16) + residual_part.to(torch.bfloat16)
        self.assertFalse(torch.equal(joined, rounded_first))
        torch.testing.assert_close(values.grad, joined, rtol=0, atol=0)
        torch.testing.assert_close(norm.weight.grad, weight_gradient.float(), rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_norm_casts_fp32_result_once_to_the_selected_output_dtype(self):
        """Without a residual the FP32 result is cast to ``output_dtype`` when given, else to the input dtype."""
        # RMS 2 normalizes to +-1, so the result is +-weight: 1 + 2**-8 + 2**-10 is exact in FP32 and
        # rounds up to 1 + 2**-7 in BF16.
        values = torch.tensor([[2.0, -2.0, 2.0, -2.0]])
        weight = torch.full((4,), 1.0 + 2.0 ** -8 + 2.0 ** -10)
        cases = ((values, torch.bfloat16, torch.bfloat16), (values, None, torch.float32),
                 (values.bfloat16(), None, torch.bfloat16))
        for inputs, output_dtype, dtype in cases:
            with self.subTest(input_dtype=inputs.dtype, output_dtype=output_dtype):
                norm = JTDeepseekV3RMSNorm(4, 0.0, output_dtype=output_dtype)
                with torch.no_grad():
                    norm.weight.copy_(weight)
                expected = cpu_rms_norm(inputs.double(), weight.double(), epsilon=0.0)[0].to(dtype)
                torch.testing.assert_close(norm(inputs), expected, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_decoder_rounds_mlp_join_once_and_shares_last_output_in_fp32(self):
        """The MLP branch joins the residual before one cast; a shared-output layer returns the result in FP32."""
        # Unit inputs and a zero attention branch leave a unit residual: joined in FP32, 1 + 2**-8 + 2**-20
        # rounds up to 1 + 2**-7, whereas rounding the branch to BF16 first ties 1 + 2**-8 back to 1.
        branch = 2.0 ** -8 + 2.0 ** -20
        hidden = torch.ones(1, 3, 16, dtype=torch.bfloat16)
        self.assertTrue(torch.equal(hidden + torch.full(hidden.shape, branch).bfloat16(), hidden))
        for shared_output, dtype in ((False, torch.bfloat16), (True, torch.float32)):
            with self.subTest(shared_output=shared_output):
                decoder = JTDeepseekV3Decoder(small_config(), 0, shared_output=shared_output)
                with patch.object(decoder.self_attn, "forward", lambda states, **_: (torch.zeros_like(states), None)), \
                        patch.object(decoder.mlp, "forward", lambda states: torch.full(states.shape, branch)):
                    output = decoder(hidden)
                torch.testing.assert_close(output, torch.full(hidden.shape, 1.0 + 2.0 ** -7, dtype=dtype),
                                           rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_model_declares_norm_boundaries_and_fp32_units(self):
        """Decoder norms join residuals; the final norm and MTP hidden norms round to BF16; shared layers are FP32."""
        config = small_config()
        config.num_nextn_predict_layers = 2
        with torch.device("meta"):
            model = JTDeepseekV3ForCausalLM(config)
        decoders = [*model.model.layers, *(layer.transformer_layer for layer in model.mtp.layers)]
        # Only the last trunk layer feeds both the final norm and the MTP hidden norm.
        self.assertEqual([decoder.shared_output for decoder in decoders], [False, True, False, False])
        norms = {name: (True, None) for name, _ in model.named_modules()
                 if name.endswith(("input_layernorm", "post_attention_layernorm"))}
        norms["model.norm"] = (False, torch.bfloat16)
        for index in range(2):
            norms |= {f"mtp.layers.{index}.enorm": (False, None), f"mtp.layers.{index}.hnorm": (False, torch.bfloat16),
                      f"mtp.layers.{index}.final_layernorm": (False, None)}
        self.assertEqual(len(norms), 2 * len(decoders) + 7)
        for name, (residual, output_dtype) in norms.items():
            with self.subTest(norm=name):
                norm = model.get_submodule(name)
                self.assertIsInstance(norm, JTDeepseekV3RMSNorm)
                self.assertEqual((norm.residual, norm.output_dtype), (residual, output_dtype))
        self.assertIsInstance(model.model.embed_tokens, torch.nn.Embedding)
        self.assertIsInstance(model.lm_head, JTDeepseekV3LMHead)
        for module in (model.model.embed_tokens, model.lm_head, *map(model.get_submodule, norms)):
            self.assertTrue(module.keep_compute_in_fp32)


class TestLMHeadNumerics(unittest.TestCase):
    """The output head keeps ordinary derivatives except for its archived weight-gradient GEMM."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_head_sums_per_use_weight_gradients_in_fp32(self):
        """Each use's BF16 weight derivative comes from the NZ GEMM and is summed in the FP32 parameter."""
        head = JTDeepseekV3LMHead(16, 32, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.arange(512.0).reshape(32, 16) % 9 - 4)
        main = torch.eye(16, dtype=torch.bfloat16).unsqueeze(0).requires_grad_()
        mtp = (torch.eye(16, dtype=torch.bfloat16) * 2.0 ** -8).unsqueeze(0).requires_grad_()
        runtime = FakeNpuRuntime()
        kernel = RecordingKernel(runtime)
        with npu_head_gemm(runtime, kernel):
            (head(main).float().sum() + head(mtp).float().sum()).backward()
        # Unit upstream gradients make each use's weight derivative ones.T @ (s * I) = s: summed in FP32 the
        # MTP share 2**-8 survives next to the main share 1, whereas a BF16 sum ties back to 1.
        upstream = torch.ones(16, 32, dtype=torch.float64)
        main_part, mtp_part = ((upstream.T @ values.detach().double()[0]).to(torch.bfloat16) for values in (main, mtp))
        self.assertEqual(len(kernel.seen), 2)
        self.assertFalse(torch.equal((main_part + mtp_part).float(), main_part.float() + mtp_part.float()))
        torch.testing.assert_close(head.weight.grad, main_part.float() + mtp_part.float(), rtol=0, atol=0)
        input_gradient = (upstream @ head.weight.detach().bfloat16().double()).to(torch.bfloat16).unsqueeze(0)
        torch.testing.assert_close(main.grad, input_gradient, rtol=0, atol=0)
        torch.testing.assert_close(mtp.grad, input_gradient, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_head_gradient_restores_compile_and_format_switches_after_success_and_failure(self):
        """The kernel runs with JIT compilation and internal formats; the caller's switches return either way."""
        gradient = torch.ones(32, 16, dtype=torch.bfloat16)
        inputs = torch.ones(32, 48, dtype=torch.bfloat16)
        for jit_compile, option in ((False, b"disable"), (True, b"enable")):
            for error in (None, RuntimeError("MatMulV2/NZ: CANN status 507011")):
                with self.subTest(jit_compile=jit_compile, option=option, kernel_fails=error is not None):
                    runtime = FakeNpuRuntime(jit_compile=jit_compile, internal_option=option)
                    kernel = RecordingKernel(runtime, error)
                    with npu_head_gemm(runtime, kernel):
                        if error is None:
                            head_weight_gradient(gradient, inputs)
                        else:
                            with self.assertRaisesRegex(RuntimeError, "507011"):
                                head_weight_gradient(gradient, inputs)
                    self.assertEqual(kernel.seen, [(True, b"enable")])
                    self.assertEqual(runtime.state(), (jit_compile, option))
