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
"""JT optimizer-step numerics: QK clipping after the update and the MF-archive L2 gradient norm."""
# The Transformers config signature is generated dynamically; keyword checks are false positives here.
# pylint: disable=unexpected-keyword-arg
from collections.abc import Callable
import math
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist

from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.core.utils.clip_grad import (
    MF_GRADIENT_LAYOUT, clip_grad_norm_, core_sum, gradient_square_sum, half_tree, mf_gradient, sequential_sum,
    tiled_square_sum,
)
from hyper_parallel.models.jt_deepseek_v3.adapter.runtime import jt_optimizer
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (
    JTDeepseekV3Config, JTDeepseekV3ForCausalLM, JTDeepseekV3MLAAttention,
)
from hyper_parallel.models.replacement import apply_module_replacements, compile_module_replacements
from hyper_parallel.trainer.config import entries_to_module_replacements
from tests.common.jt_config import jt_test_config
from tests.common.mark_utils import arg_mark

THRESHOLD = 100.0


def replaced_model(dtype: torch.dtype) -> JTDeepseekV3ForCausalLM:
    """Build a CPU-sized JT model whose attention uses the MLA replacement.

    Two heads with 4 nope, 4 rope and 4 value channels give 8 projection rows per head.
    """
    torch.manual_seed(7)
    config = JTDeepseekV3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, moe_intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2, n_group=1, topk_group=1,
        q_lora_rank=8, kv_lora_rank=8, qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=4,
        mlp_layer_types=["dense", "sparse"], max_position_embeddings=32, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        architectures=["JTDeepseekV3ForCausalLM"], num_nextn_predict_layers=1, use_pad_tokens=True,
        norm_topk_prob=True, routed_scaling_factor=1.0, moe_aux_loss_coeff=0.01, mtp_loss_factor=0.3,
        moe_router_enable_expert_bias=False)
    config.rope_interleave = True
    model = JTDeepseekV3ForCausalLM(config)
    rules = entries_to_module_replacements(jt_test_config().plan_overrides)
    apply_module_replacements(model, compile_module_replacements(model, rules), weights_mapping=[])
    return model.to(dtype)


def attention_modules(model: torch.nn.Module) -> list[JTDeepseekV3MLAAttention]:
    """Return the replaced trunk and MTP attention modules in model order."""
    return [module for module in model.modules() if isinstance(module, JTDeepseekV3MLAAttention)]


def row_factors(head_scales: list[float], rope_scaled: bool) -> torch.Tensor:
    """Expected per-row multipliers of a q_b_proj (``rope_scaled``) or kv_b_proj weight.

    QK clipping scales a head's query and key nope rows by sqrt(scale) and its query rope
    rows by scale; key value rows are unchanged.
    """
    rows = []
    for scale in head_scales:
        rows += [scale ** 0.5] * 4 + [scale if rope_scaled else 1.0] * 4
    return torch.tensor(rows)[:, None]


def two_rank_all_reduce(peer: torch.Tensor) -> Callable[..., None]:
    """Emulate an all-reduce whose only other rank contributes ``peer``."""
    def all_reduce(tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM, group: Any = None) -> None:
        """Reduce ``tensor`` in place with the peer contribution under ``op``."""
        del group
        tensor.copy_(torch.maximum(tensor, peer) if op == dist.ReduceOp.MAX else tensor + peer)
    return all_reduce


class TestQKClip(unittest.TestCase):
    """QK clipping applied after the optimizer step must agree across replicas."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_optimizer_step_clips_projections(self):
        """Feature: QK clipping in the optimizer post-update hook.

        Description: Step the recipe optimizer behind the mixed-precision wrapper with FP32 parameters
            while head 0 exceeds the threshold 4x and head 1 stays below it.
        Expectation: Head 0 is scaled exactly once; head 1 is unchanged.
        """
        model = replaced_model(torch.float32)
        builder = jt_optimizer.build_optimizer(
            model=model, qk_clip_threshold=THRESHOLD,
            muon_config={"lr": 0.0}, adamw_config={"adamw_lr": 0.0})
        optimizer = Float16OptimizerWithFloat16Params(builder.get_optimizer(), model)
        modules = attention_modules(model)
        originals = []
        for module in modules:
            module.max_logits_val = torch.tensor([4 * THRESHOLD, THRESHOLD / 2])
            originals.append((module.q_b_proj.weight.detach().clone(), module.kv_b_proj.weight.detach().clone()))
        optimizer.step()
        for module, (query, key_value) in zip(modules, originals):
            self.assertTrue(torch.equal(module.q_b_proj.weight, query * row_factors([0.25, 1.0], rope_scaled=True)))
            self.assertTrue(torch.equal(module.kv_b_proj.weight,
                                        key_value * row_factors([0.25, 1.0], rope_scaled=False)))
            self.assertTrue(torch.equal(module.max_logits_val, torch.zeros(2)))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_replicas_clip_with_shared_maximum(self):
        """Feature: QK clipping across data-parallel replicas.

        Description: Two replicas of the same heads see different heads exceed the threshold;
            each clips under an all-reduce emulating the other replica.
        Expectation: Both replicas clip both heads by the shared maximum and stay identical.
        """
        replicas = [replaced_model(torch.float32), replaced_model(torch.float32)]
        originals = [(module.q_b_proj.weight.detach().clone(), module.kv_b_proj.weight.detach().clone())
                     for module in attention_modules(replicas[0])]
        local_maxima = ([4 * THRESHOLD, THRESHOLD / 2], [THRESHOLD / 2, 4 * THRESHOLD])
        contributions = []
        for model, maxima in zip(replicas, local_maxima):
            model.qk_clip_group = object()
            for module in attention_modules(model):
                module.max_logits_val = torch.tensor(maxima)
            contributions.append(torch.cat([module.max_logits_val for module in attention_modules(model)]))
        for model, peer in zip(replicas, reversed(contributions)):
            with patch.object(dist, "all_reduce", side_effect=two_rank_all_reduce(peer)):
                jt_optimizer.clip_qk(model, THRESHOLD)
        for replica in replicas:
            for module, (query, key_value) in zip(attention_modules(replica), originals):
                self.assertTrue(torch.equal(module.q_b_proj.weight,
                                            query * row_factors([0.25, 0.25], rope_scaled=True)),
                                "every replica must clip q_b_proj with the shared maximum")
                self.assertTrue(torch.equal(module.kv_b_proj.weight,
                                            key_value * row_factors([0.25, 0.25], rope_scaled=False)),
                                "every replica must clip kv_b_proj with the shared maximum")


# Native JT storage behind every MF_GRADIENT_LAYOUT entry. The MF Q/KV down projections are
# fixed row ranges of ``linear_qkv`` (0:1536, 1536:2112); every other dimension is small.
# Logical sizes cover each archived SumX tiling: <= 5120 elements, exactly 40960, and
# > 5120 with a partial final tile.
ATTENTION_SHAPES = {
    'self_attn.linear_qkv.weight': (2112, 4),
    'self_attn.q_a_layernorm.weight': (64,),
    'self_attn.kv_a_layernorm.weight': (64,),
    'self_attn.q_b_proj.weight': (16, 4),
    'self_attn.kv_b_proj.weight': (16, 8),
    'self_attn.o_proj.weight': (8, 16),
    'input_layernorm.weight': (64,),
    'post_attention_layernorm.weight': (64,),
}
MOE_SHAPES = {
    'mlp.gate.weight': (2, 32),
    'mlp.experts.gate_up_proj': (2, 8, 16),
    'mlp.experts.down_proj': (2, 16, 4),
    'mlp.shared_experts.gate_proj.weight': (4, 16),
    'mlp.shared_experts.up_proj.weight': (4, 16),
    'mlp.shared_experts.down_proj.weight': (16, 4),
}
NATIVE_SHAPES = {
    'model.embed_tokens.weight': (1024, 17),
    **{'model.layers.0.' + name: shape for name, shape in ATTENTION_SHAPES.items()},
    'model.layers.0.mlp.gate_proj.weight': (16, 8),
    'model.layers.0.mlp.up_proj.weight': (16, 8),
    'model.layers.0.mlp.down_proj.weight': (8, 16),
    **{'model.layers.1.' + name: shape for name, shape in {**ATTENTION_SHAPES, **MOE_SHAPES}.items()},
    'model.norm.weight': (64,),
    'lm_head.weight': (80, 512),
    'mtp.layers.0.enorm.weight': (64,),
    'mtp.layers.0.hnorm.weight': (64,),
    'mtp.layers.0.eh_proj.weight': (8, 16),
    **{'mtp.layers.0.transformer_layer.' + name: shape for name, shape in {**ATTENTION_SHAPES, **MOE_SHAPES}.items()},
    'mtp.layers.0.final_layernorm.weight': (64,),
}

# MF names whose gradient is one native gradient, unchanged.
RENAMED = {
    'embedding.word_embeddings.weight': 'model.embed_tokens.weight',
    'output_layer.weight': 'lm_head.weight',
    'decoder.final_layernorm.weight': 'model.norm.weight',
    'decoder.layers.0.self_attention.linear_proj.weight': 'model.layers.0.self_attn.o_proj.weight',
    'decoder.layers.0.self_attention.linear_q_up_proj.weight': 'model.layers.0.self_attn.q_b_proj.weight',
    'decoder.layers.0.self_attention.linear_kv_up_proj.weight': 'model.layers.0.self_attn.kv_b_proj.weight',
    'decoder.layers.0.self_attention.q_layernorm.weight': 'model.layers.0.self_attn.q_a_layernorm.weight',
    'decoder.layers.0.self_attention.kv_layernorm.weight': 'model.layers.0.self_attn.kv_a_layernorm.weight',
    'decoder.layers.0.input_layernorm.weight': 'model.layers.0.input_layernorm.weight',
    'decoder.layers.0.pre_mlp_layernorm.weight': 'model.layers.0.post_attention_layernorm.weight',
    'decoder.layers.0.mlp.linear_fc2.weight': 'model.layers.0.mlp.down_proj.weight',
    'decoder.layers.1.mlp.router.weight': 'model.layers.1.mlp.gate.weight',
    'decoder.layers.1.mlp.shared_experts.linear_fc2.weight': 'model.layers.1.mlp.shared_experts.down_proj.weight',
    'mtp.layers.0.eh_proj.weight': 'mtp.layers.0.eh_proj.weight',
    'mtp.layers.0.transformer_layer.mlp.router.weight': 'mtp.layers.0.transformer_layer.mlp.gate.weight',
    'mtp.layers.0.final_layernorm.weight': 'mtp.layers.0.final_layernorm.weight',
}


def register(root: torch.nn.Module, name: str, parameter: torch.nn.Parameter) -> None:
    """Register ``parameter`` under its dotted name, creating parent modules."""
    *path, leaf = name.split('.')
    owner = root
    for part in path:
        if not hasattr(owner, part):
            owner.add_module(part, torch.nn.Module())
        owner = getattr(owner, part)
    owner.register_parameter(leaf, parameter)


def jt_storage(seed: int) -> tuple[torch.nn.Module, dict[str, torch.nn.Parameter]]:
    """Build a module with native JT parameter names holding random FP32 gradients."""
    generator = torch.Generator().manual_seed(seed)
    root = torch.nn.Module()
    parameters = {}
    for name, shape in NATIVE_SHAPES.items():
        parameter = torch.nn.Parameter(torch.zeros(shape))
        parameter.grad = torch.randn(shape, generator=generator)
        register(root, name, parameter)
        parameters[name] = parameter
    return root, parameters


def gradients_of(parameters: dict[str, torch.nn.Parameter]) -> dict[str, torch.Tensor]:
    """Map each native name to its live gradient, as ``clip_grad_norm_`` reads it."""
    return {name: parameter.grad for name, parameter in parameters.items()}


def double_norm(parameters: dict[str, torch.nn.Parameter]) -> float:
    """FP64 L2 norm over every native gradient."""
    return math.sqrt(sum(parameter.grad.double().square().sum().item() for parameter in parameters.values()))


# 1 + EPS rounds back to 1 in FP32 (ties to even), while EPS + EPS = 2 * EPS survives;
# whether two EPS terms meet before the 1 therefore exposes the pairing order.
EPS = 2.0 ** -24
ONE_PLUS_TWO_EPS = 1.0 + 2.0 ** -23


def lanes(*values: float) -> torch.Tensor:
    """Build one FP32 row from explicit lane values."""
    return torch.tensor(values, dtype=torch.float32)


class TestArchivedReductions(unittest.TestCase):
    """Lock the archived pairing order and check that every element contributes once."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_reduction_includes_odd_tail_and_partial_tiles(self):
        """Every element contributes, including non-power-of-two widths and final partial tiles."""
        values = torch.arange(1, 6, dtype=torch.float32)
        self.assertEqual(half_tree(values).item(), 15)
        ones = torch.ones(8144 * 2 + 17, dtype=torch.float32)
        self.assertEqual(tiled_square_sum(ones, 8144, 16), np.float32(ones.numel()))
        ones[-1] = 2
        self.assertEqual(tiled_square_sum(ones, 8144, 16), np.float32(ones.numel() + 3))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_half_tree_pairs_halves_and_folds_the_odd_tail_into_lane_zero(self):
        """Entry j meets entry j + n // 2; an odd tail joins the first partial sum."""
        self.assertEqual(half_tree(lanes(1.0, EPS, 0.0, EPS)).item(), ONE_PLUS_TWO_EPS)
        # Folding the tail into lane 1 instead would give (1) + (EPS + EPS) = 1 + 2 * EPS.
        self.assertEqual(half_tree(lanes(1.0, 0.0, 0.0, EPS, EPS)).item(), 1.0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_host_sums_round_after_every_addition_in_core_order(self):
        """Scalar sums round after each addition; core sums stride tiles over the cores first."""
        partials = np.asarray([1.0, EPS, 0.0, EPS], dtype=np.float32)
        self.assertEqual(sequential_sum(partials), np.float32(1.0))
        # Tile i accumulates on core i % cores: core 0 holds 1 + 0, core 1 holds EPS + EPS.
        self.assertEqual(core_sum(partials, cores=2), np.float32(ONE_PLUS_TWO_EPS))
        self.assertEqual(core_sum(partials, cores=1), np.float32(1.0))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_square_sum_tile_reduces_rows_by_halves_then_lanes_natively(self):
        """Element r * width + l is row r of lane l; rows meet by halves, then lanes sum natively."""
        rows = torch.zeros(64, dtype=torch.float32)
        rows[[0, 16, 48]] = torch.tensor([1.0, 2.0 ** -12, 2.0 ** -12])  # squares 1, EPS, EPS in lane 0
        self.assertEqual(tiled_square_sum(rows, 64, 16), np.float32(ONE_PLUS_TWO_EPS))
        lanes_only = torch.zeros(16, dtype=torch.float32)
        lanes_only[[0, 1, 3]] = torch.tensor([1.0, 2.0 ** -12, 2.0 ** -12])  # squares 1, EPS, EPS in row 0
        self.assertEqual(tiled_square_sum(lanes_only, 16, 16), np.float32(1.0))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_gradient_square_sum_selects_the_archived_tiling_by_element_count(self):
        """Up to 5120 elements form one 64-lane tile; exactly 40960 use 1024/64 tiles, not the 8144/16 default."""
        single_tile = torch.zeros(5120, dtype=torch.float32)
        single_tile[[0, 64, 192]] = torch.tensor([1.0, 2.0 ** -12, 2.0 ** -12])  # lane 0, rows 0, 1, 3
        self.assertEqual(gradient_square_sum(single_tile), np.float32(ONE_PLUS_TWO_EPS))
        separate_tiles = torch.zeros(40960, dtype=torch.float32)
        separate_tiles[[0, 1024, 2048]] = torch.tensor([1.0, 2.0 ** -12, 2.0 ** -12])  # tiles 0, 1, 2 of 1024
        self.assertEqual(gradient_square_sum(separate_tiles), np.float32(1.0))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_square_sums_match_double_precision(self):
        """Every SumX tiling stays within the FP32 rounding bound of the FP64 sum of squares."""
        torch.manual_seed(253)
        # The longest path (8144/16 tiles, 601 tiles over 40 cores) rounds at most 76 times, squaring included.
        bound = 128 * 2.0 ** -24
        for count in (64, 5120, 40960, 8144 * 600 + 5):
            value = torch.randn(count, dtype=torch.float32)
            expected = value.double().square().sum().item()
            actual = float(gradient_square_sum(value))
            self.assertLess(abs(actual - expected), bound * expected, f"count={count}")


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
class TestArchivedGradientClipping(unittest.TestCase):
    """MF logical gradient layout and the FP32 L2 norm and clipping of ``clip_grad_norm_``."""

    def test_renamed_gradients_are_native_values(self):
        """Renamed MF tensors expose the live native gradient values unchanged."""
        _, parameters = jt_storage(seed=1)
        gradients = gradients_of(parameters)
        for archived, native in RENAMED.items():
            with self.subTest(archived=archived):
                torch.testing.assert_close(mf_gradient(gradients, archived), parameters[native].grad, rtol=0, atol=0)

    def test_fused_storage_is_split_into_archived_matrices(self):
        """Q/KV rows, gate/up row interleave and expert transposes match per-row references."""
        _, parameters = jt_storage(seed=2)
        grad = gradients_of(parameters)
        qkv = grad['model.layers.0.self_attn.linear_qkv.weight']
        attention = 'decoder.layers.0.self_attention.'
        torch.testing.assert_close(mf_gradient(grad, attention + 'linear_q_down_proj.weight'),
                                   qkv[:1536], rtol=0, atol=0)
        torch.testing.assert_close(mf_gradient(grad, attention + 'linear_kv_down_proj.weight'),
                                   qkv[1536:], rtol=0, atol=0)
        for archived, prefix in [('decoder.layers.0.mlp.linear_fc1.weight', 'model.layers.0.mlp.'),
                                 ('decoder.layers.1.mlp.shared_experts.linear_fc1.weight',
                                  'model.layers.1.mlp.shared_experts.')]:
            gate, up = grad[prefix + 'gate_proj.weight'], grad[prefix + 'up_proj.weight']
            expected = torch.stack([row for pair in zip(gate, up) for row in pair])
            torch.testing.assert_close(mf_gradient(grad, archived), expected, rtol=0, atol=0)
        # gate_up_proj is [expert, 2 * intermediate, hidden]; weight1 row e * hidden + h is its column h.
        gate_up = grad['model.layers.1.mlp.experts.gate_up_proj']
        experts, _, hidden = gate_up.shape
        expected = torch.stack([gate_up[e, :, h] for e in range(experts) for h in range(hidden)])
        torch.testing.assert_close(mf_gradient(grad, 'decoder.layers.1.mlp.experts.weight1'),
                                   expected, rtol=0, atol=0)
        # down_proj is [expert, hidden, intermediate]; weight2 row e * intermediate + i is its column i.
        down = grad['model.layers.1.mlp.experts.down_proj']
        intermediate = down.shape[2]
        expected = torch.stack([down[e, :, i] for e in range(experts) for i in range(intermediate)])
        torch.testing.assert_close(mf_gradient(grad, 'decoder.layers.1.mlp.experts.weight2'),
                                   expected, rtol=0, atol=0)

    def test_norm_and_clipped_gradients_match_double_precision(self):
        """The archived FP32 schedule computes the full gradient norm and clips every gradient."""
        for ratio in (0.25, 4.0):
            with self.subTest(max_norm_ratio=ratio):
                model, parameters = jt_storage(seed=3)
                originals = {name: parameter.grad.clone() for name, parameter in parameters.items()}
                expected_norm = double_norm(parameters)
                max_norm = expected_norm * ratio
                norm = clip_grad_norm_(model, max_norm)
                self.assertEqual(norm.dtype, torch.float32)
                torch.testing.assert_close(norm.double(), torch.tensor(expected_norm, dtype=torch.float64),
                                           rtol=1e-6, atol=0)
                coefficient = min(max_norm / (expected_norm + 1e-6), 1.0)
                for name, parameter in parameters.items():
                    if coefficient == 1.0:
                        # A clamped coefficient multiplies by exactly one.
                        torch.testing.assert_close(parameter.grad, originals[name], rtol=0, atol=0)
                    else:
                        torch.testing.assert_close(parameter.grad.double(), originals[name].double() * coefficient,
                                                   rtol=1e-6, atol=0)

    def test_norm_adds_entries_in_layout_order_and_clips_unlisted_gradients(self):
        """Entries add in MF layout order with one FP32 rounding each; an unlisted gradient is only clipped.

        Layout entry 8 (layer 1 ``linear_proj``) squares to 1 and every other entry to 2 ** -26. The eight
        entries before it sum exactly to 2 ** -23, so the running sum becomes 1 + 2 ** -23; every later
        entry is an eighth of an ulp and rounds away, and the square root rounds back to 1. Parameter
        order, a reversed layout, a pairwise or an FP64 sum all end above 1.
        """
        model, parameters = jt_storage(seed=4)
        for name, parameter in parameters.items():
            parameter.grad.zero_()
            if not name.endswith('up_proj.weight'):  # gate_proj alone opens each fused linear_fc1 entry
                parameter.grad.view(-1)[0] = 2.0 ** -13
            if name.endswith('linear_qkv.weight'):  # row 1536 opens linear_kv_down_proj
                parameter.grad[1536, 0] = 2.0 ** -13
        parameters['model.layers.1.self_attn.o_proj.weight'].grad.view(-1)[0] = 1.0
        unlisted = torch.nn.Parameter(torch.zeros(3))
        unlisted.grad = torch.full((3,), 3.0)
        register(model, 'extra.weight', unlisted)
        self.assertEqual(MF_GRADIENT_LAYOUT[8], 'decoder.layers.1.self_attention.linear_proj.weight')
        norm = clip_grad_norm_(model, 0.5)
        self.assertEqual(norm.item(), 1.0, "entries must add in layout order, rounding after each")
        coefficient = torch.clamp(0.5 / (norm + 1e-6), max=1.0)
        self.assertTrue(torch.equal(unlisted.grad, torch.full((3,), 3.0) * coefficient),
                        "a gradient outside the layout must be clipped but not counted")

    def test_shipped_layout_reads_every_jt_parameter(self):
        """Every MF layout name resolves on the JT model, and together they read all its gradients."""
        model = replaced_model(torch.float32)
        gradients = {name: torch.full_like(parameter, float(index))
                     for index, (name, parameter) in enumerate(model.named_parameters(), start=1)}
        seen = set()
        for name in MF_GRADIENT_LAYOUT:
            seen.update(mf_gradient(gradients, name).unique().tolist())
        self.assertEqual(seen, set(range(1, len(gradients) + 1)))


if __name__ == "__main__":
    unittest.main()
