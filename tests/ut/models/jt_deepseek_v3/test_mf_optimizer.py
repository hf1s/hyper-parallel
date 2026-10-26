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
"""MindFormers-aligned JT Muon: Newton-Schulz blocks, FP32 update scaling and the recipe builder."""
import inspect
import unittest
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import patch

import numpy as np
import torch
import yaml
from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32MLP

from hyper_parallel.components.modules.swiglu_mlp import SwiGLUMLP
from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.muon import Muon, zeropower_via_newtonschulz5
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.models.jt_deepseek_v3.adapter.runtime import jt_optimizer
from hyper_parallel.models.jt_deepseek_v3.adapter.runtime.mf_optimizer import (
    MFMuon, MFOptimizerBuilder, build_mf_optimizer, make_mf_ns_transform_fn,
)
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3ForCausalLM
from hyper_parallel.trainer.config.optimization import OptimizerConfig
from hyper_parallel.trainer.config.resolver import resolve_component
from tests.common.jt_config import jt_test_config
from tests.common.mark_utils import arg_mark
from tests.ut.models.jt_deepseek_v3.test_jt_optimizer import attention_modules, replaced_model, row_factors

THRESHOLD = 100.0
LR, WEIGHT_DECAY, RMS = 1e-4, 0.1, 0.2
NEWTON_SCHULZ = {"ns_steps": 5, "ns_variant": "legacy", "ns_epsilon": 1e-7}
MUON_CONFIG = {"lr": LR, "weight_decay": WEIGHT_DECAY, "momentum": 0.95, "nesterov": True,
               "matched_adamw_rms": RMS, **NEWTON_SCHULZ}
ADAMW_CONFIG = {"adamw_lr": LR, "adamw_weight_decay": WEIGHT_DECAY, "adamw_betas": [0.9, 0.95], "adamw_eps": 1e-8}
# The recipe's structure at CPU size: nope rows = 2 x rope rows and every attention block taller than wide.
SMALL = SimpleNamespace(q_lora_rank=12, kv_lora_rank=6, qk_rope_head_dim=4, qk_nope_head_dim=8, v_head_dim=8,
                        moe_intermediate_size=6)
HEADS, HIDDEN, EXPERTS, FFN = 4, 40, 3, 10
# Model dimensions of examples/training_demo/jt_deepseek_v3/hp_4k_1_card.yaml.
RECIPE = SimpleNamespace(q_lora_rank=1536, kv_lora_rank=512, qk_rope_head_dim=64, qk_nope_head_dim=128,
                         v_head_dim=128, moe_intermediate_size=1536)
RECIPE_HEADS, RECIPE_HIDDEN, RECIPE_EXPERTS, RECIPE_FFN, RECIPE_SHARED_FFN = 128, 5120, 8, 12288, 2 * 1536
LEGACY_NS = partial(zeropower_via_newtonschulz5, steps=5, ns_variant="legacy", epsilon=1e-7)


def fused_shapes(config: Any, heads: int, hidden: int, experts: int, ffn: int, shared_ffn: int) -> dict:
    """Shapes of every fused parameter class, keyed by a model parameter name."""
    return {
        "model.layers.0.self_attn.linear_qkv.weight":
            (config.q_lora_rank + config.kv_lora_rank + config.qk_rope_head_dim, hidden),
        "model.layers.1.self_attn.q_b_proj.weight":
            (heads * (config.qk_nope_head_dim + config.qk_rope_head_dim), config.q_lora_rank),
        "mtp.layers.0.transformer_layer.self_attn.kv_b_proj.weight":
            (heads * (config.qk_nope_head_dim + config.v_head_dim), config.kv_lora_rank),
        "model.layers.1.mlp.experts.gate_up_proj": (experts, 2 * config.moe_intermediate_size, hidden),
        "model.layers.0.mlp.linear_fc1.weight": (2 * ffn, hidden),
        "mtp.layers.0.transformer_layer.mlp.shared_experts.linear_fc1.weight": (2 * shared_ffn, hidden),
    }


SMALL_SHAPES = fused_shapes(SMALL, HEADS, HIDDEN, EXPERTS, FFN, 2 * SMALL.moe_intermediate_size)


def mf_rows(fqn: str, rows: int, config: Any) -> Optional[List[torch.Tensor]]:
    """Row ids of MindFormers' Newton-Schulz blocks of one HP parameter, from the ``muon_split_fn`` schema.

    ``None`` means MindFormers orthogonalizes the parameter whole.
    """
    def per_head(first: int, second: int) -> List[torch.Tensor]:
        """Rows of a ``periodic`` split: the first and the second part of every head."""
        starts = torch.arange(rows // (first + second))[:, None] * (first + second)
        return [(starts + torch.arange(first)).flatten(), (starts + first + torch.arange(second)).flatten()]

    if fqn.endswith("linear_qkv.weight"):  # linear_q_down_proj | linear_kv_down_proj: (kv_lora_rank, rope, 1)
        latent_start = config.q_lora_rank
        latent_stop = latent_start + config.kv_lora_rank
        return [torch.arange(latent_start), torch.arange(latent_start, latent_stop), torch.arange(latent_stop, rows)]
    if fqn.endswith("q_b_proj.weight"):  # linear_q_up_proj: (qk_head_dim, qk_pos_emb_head_dim, heads)
        return per_head(config.qk_nope_head_dim, config.qk_rope_head_dim)
    if fqn.endswith("kv_b_proj.weight"):  # linear_kv_up_proj: (qk_head_dim, v_head_dim, heads)
        return per_head(config.qk_nope_head_dim, config.v_head_dim)
    if fqn.endswith(("experts.gate_up_proj", "linear_fc1.weight")):
        # experts.weight1 reshape_concat; interleaved linear_fc1 alt_pair_periodic. Both are gate | up here.
        return [torch.arange(rows // 2), torch.arange(rows // 2, rows)]
    return None


def mf_update(value: torch.Tensor, ns_input: torch.Tensor, blocks: Optional[List[torch.Tensor]],
              newton_schulz: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """MindFormers' ``_apply_muon_update`` from its BF16 Newton-Schulz input, each block scaled by its own dims.

    ``lr * (sqrt(max(dims)) * rms)`` and ``1 - lr * weight_decay`` are FP32 scalars rounded once per
    operation; every BF16 block output is multiplied out of place into FP32 and the FP32 update is
    subtracted from the decayed FP32 parameter.
    """
    blocks = [torch.arange(ns_input.shape[-2])] if blocks is None else blocks
    update = torch.empty(value.shape, dtype=torch.float32)
    for rows in blocks:
        block = ns_input[..., rows, :]
        ratio = np.sqrt(np.float32(max(block.shape[-2:]))) * np.float32(RMS)
        update[..., rows, :] = newton_schulz(block).float() * float(np.float32(LR) * ratio)
    decay = np.float32(1.0) - np.float32(LR) * np.float32(WEIGHT_DECAY)
    return value * float(decay) - update


def max_ulp(actual: torch.Tensor, expected: torch.Tensor) -> int:
    """Largest distance between two FP32 tensors in units in the last place."""
    def ordered(tensor: torch.Tensor) -> torch.Tensor:
        """Map FP32 bit patterns onto integers ordered like the values."""
        bits = tensor.float().contiguous().view(torch.int32).long()
        return torch.where(bits < 0, -(bits & 0x7FFFFFFF), bits)
    return int((ordered(actual) - ordered(expected)).abs().max())


def identity_ns(value: torch.Tensor, steps: int) -> torch.Tensor:
    """Newton-Schulz stand-in returning its input, so the scaled update is known exactly."""
    del steps
    return value


def normalize_ns(value: torch.Tensor, steps: int) -> torch.Tensor:
    """Newton-Schulz stand-in normalizing every matrix by its FP64 Frobenius norm.

    Like Newton-Schulz the result depends on each block as a whole, so the block partition shows in
    the update; unlike BF16 matmuls it does not depend on how the optimizer batches blocks.
    """
    del steps
    return (value.double() / value.double().norm(dim=(-2, -1), keepdim=True)).to(value.dtype)


def muon_with_parameter(optimizer_cls: type, value: torch.Tensor, gradient: torch.Tensor, name: str,
                        **options: Any) -> tuple:
    """Build a Muon optimizer with momentum 0 for one named parameter, so its NS input is BF16(gradient)."""
    parameter = torch.nn.Parameter(value.clone())
    parameter.model_name = name
    parameter.grad = gradient.clone()
    optimizer = optimizer_cls([parameter], lr=LR, weight_decay=WEIGHT_DECAY, matched_adamw_rms=RMS, momentum=0.0,
                              nesterov=True, **NEWTON_SCHULZ, **options)
    return optimizer, parameter


def one_step(optimizer_cls: type, value: torch.Tensor, gradient: torch.Tensor, name: str,
             **options: Any) -> torch.Tensor:
    """Apply one step to a single parameter and return its new value."""
    optimizer, parameter = muon_with_parameter(optimizer_cls, value, gradient, name, **options)
    optimizer.step()
    return parameter.data.clone()


def mf_model(fused: bool) -> JTDeepseekV3ForCausalLM:
    """Small FP32 JT model with QK statistics below the clip threshold.

    ``fused`` packs the dense and shared-expert MLPs into SwiGLU ``linear_fc1`` as hp_4k_1_card.yaml does.
    """
    model = replaced_model(torch.float32)
    if fused:
        for name, module in list(model.named_modules()):
            if type(module) is DeepseekV32MLP:  # pylint: disable=unidiomatic-typecheck
                parent, _, child = name.rpartition(".")
                setattr(model.get_submodule(parent), child, SwiGLUMLP(module=module))
    for module in attention_modules(model):
        module.max_logits_val = torch.zeros(2)
    return model


def build(model: torch.nn.Module, **muon_overrides: Any) -> MFOptimizerBuilder:
    """Build the MindFormers-aligned optimizer for ``model`` with the recipe's settings."""
    return build_mf_optimizer(model=model, qk_clip_threshold=THRESHOLD, muon_config={**MUON_CONFIG, **muon_overrides},
                              adamw_config=ADAMW_CONFIG)


def set_gradients(model: torch.nn.Module, seed: int) -> None:
    """Give every parameter a reproducible gradient."""
    generator = torch.Generator().manual_seed(seed)
    for parameter in model.parameters():
        parameter.grad = torch.randn(parameter.shape, generator=generator) * 1e-2


class TestMFNewtonSchulzBlocks(unittest.TestCase):
    """make_mf_ns_transform_fn cuts fused parameters into MindFormers' logical matrices."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_blocks_match_mindformers_rows(self):
        """Feature: MindFormers Newton-Schulz partition (difference 1).

        Description: Split BF16 inputs of every fused parameter class (linear_qkv, q_b_proj, kv_b_proj, expert
            gate_up_proj, fused dense and shared-expert linear_fc1) and restore one marker per block.
        Expectation: Every block equals MindFormers' logical matrix row for row, and restore fills every row
            from its own block.
        """
        transform_fn = make_mf_ns_transform_fn(SMALL)
        for fqn, shape in SMALL_SHAPES.items():
            with self.subTest(fqn=fqn):
                value = torch.randn(shape).bfloat16()
                transform = transform_fn(fqn, value)
                rows = mf_rows(fqn, shape[-2], SMALL)
                self.assertEqual(len(transform.tensors), len(rows))
                for tensor, block_rows in zip(transform.tensors, rows):
                    torch.testing.assert_close(tensor, value[..., block_rows, :], rtol=0, atol=0)
                restored = torch.full(shape, float("nan"))
                transform.restore([torch.full(tensor.shape, float(index))
                                   for index, tensor in enumerate(transform.tensors)], restored)
                for index, block_rows in enumerate(rows):
                    block = restored[..., block_rows, :]
                    self.assertTrue(torch.equal(block, torch.full_like(block, index)),
                                    f"block {index} must own exactly MindFormers' rows")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_recipe_sizes_match_mindformers_blocks(self):
        """Feature: MindFormers Newton-Schulz partition at hp_4k_1_card.yaml sizes.

        Description: Split one-column inputs holding their row ids at the recipe's row counts, and meta inputs
            with the recipe's full parameter shapes.
        Expectation: Block rows equal MindFormers' row lists and block shapes equal MindFormers' logical
            matrices, e.g. q_b_proj into [16384, 1536] nope and [8192, 1536] rope rows.
        """
        expected_shapes = {
            "linear_qkv": [(1536, 5120), (512, 5120), (64, 5120)],
            "q_b_proj": [(16384, 1536), (8192, 1536)],
            "kv_b_proj": [(16384, 512), (16384, 512)],
            "gate_up_proj": [(8, 1536, 5120)] * 2,
            "mlp.linear_fc1": [(12288, 5120)] * 2,
            "shared_experts.linear_fc1": [(3072, 5120)] * 2,
        }
        transform_fn = make_mf_ns_transform_fn(RECIPE)
        shapes = fused_shapes(RECIPE, RECIPE_HEADS, RECIPE_HIDDEN, RECIPE_EXPERTS, RECIPE_FFN, RECIPE_SHARED_FFN)
        for (fqn, shape), block_shapes in zip(shapes.items(), expected_shapes.values()):
            with self.subTest(fqn=fqn):
                meta = transform_fn(fqn, torch.empty(shape, dtype=torch.bfloat16, device="meta"))
                self.assertEqual([tuple(tensor.shape) for tensor in meta.tensors], block_shapes)
                row_ids = torch.arange(shape[-2], dtype=torch.float32)[:, None].expand(*shape[:-1], 1).contiguous()
                blocks = transform_fn(fqn, row_ids).tensors
                for tensor, block_rows in zip(blocks, mf_rows(fqn, shape[-2], RECIPE)):
                    expected = block_rows.float()[:, None].expand(*tensor.shape[:-1], 1)
                    torch.testing.assert_close(tensor, expected, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_whole_parameters_and_wrong_layouts(self):
        """Feature: Parameters MindFormers keeps whole, and fused parameters whose layout does not fit.

        Description: Ask for blocks of parameters MindFormers orthogonalizes whole (including unfused gate/up
            projections) and of fused parameters whose shape does not fit the configuration.
        Expectation: Both return None, so these parameters keep Muon's whole-matrix Newton-Schulz.
        """
        transform_fn = make_mf_ns_transform_fn(SMALL)
        for fqn in ("model.layers.0.self_attn.o_proj.weight", "model.layers.1.mlp.experts.down_proj",
                    "model.layers.1.mlp.gate.weight", "mtp.layers.0.eh_proj.weight",
                    "model.layers.0.mlp.gate_proj.weight", "model.layers.1.mlp.shared_experts.up_proj.weight"):
            self.assertIsNone(transform_fn(fqn, torch.zeros(8, 8)), fqn)
        for fqn, shape in (("model.layers.0.self_attn.linear_qkv.weight", (21, HIDDEN)),
                           ("model.layers.0.self_attn.q_b_proj.weight", (13, SMALL.q_lora_rank)),
                           ("model.layers.1.mlp.experts.gate_up_proj", (12, HIDDEN)),
                           ("model.layers.0.mlp.linear_fc1.weight", (21, HIDDEN))):
            self.assertIsNone(transform_fn(fqn, torch.zeros(shape)), fqn)


class TestMFUpdatePrecision(unittest.TestCase):
    """MFMuon keeps the scaled update FP32 as MindFormers does; native Muon rounds it to BF16."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_scaled_update_stays_fp32(self):
        """Feature: FP32 lr * sqrt(max(dims)) * rms application (difference 2).

        Description: With an identity Newton-Schulz, step a [16, 16] parameter from zero and from non-zero
            values. sqrt(16) * 0.2 needs more than BF16's 8 significant bits, so the BF16 update times it is not
            representable in BF16.
        Expectation: Both native Muon paths reproduce the BF16-rounded update and miss MindFormers' FP32 formula
            by many ULPs; MFMuon reproduces the FP32 formula, bit for bit from zero.
        """
        torch.manual_seed(0)
        gradient = torch.randn(16, 16).bfloat16().float()
        for value in (torch.zeros(16, 16), torch.randn(16, 16) * 0.02):
            expected = mf_update(value, gradient.bfloat16(), None, lambda block: block)
            scale = 16 ** 0.5 * RMS
            rounded = value * (1 - LR * WEIGHT_DECAY) - LR * (gradient.bfloat16() * scale).float()
            self.assertGreater(max_ulp(rounded, expected), 16, "the BF16-rounded update must be observable")
            native_paths = {"fast": {}, "transform": {"ns_transform_fn": lambda fqn, tensor: None}}
            for path, options in native_paths.items():
                with self.subTest(zero=not value.any(), native=path):
                    native = one_step(Muon, value, gradient, "w", zeropower_fn=identity_ns, **options)
                    self.assertLessEqual(max_ulp(native, rounded), 1)
                    self.assertGreater(max_ulp(native, expected), 16)
            actual = one_step(MFMuon, value, gradient, "w", zeropower_fn=identity_ns)
            self.assertLessEqual(max_ulp(actual, expected), 0 if not value.any() else 1)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_split_update_matches_mindformers(self):
        """Feature: MindFormers blocks with FP32 scaling (both differences).

        Description: One legacy Newton-Schulz step of a q_b_proj-shaped parameter, whose nope and rope blocks
            have different sizes, from MindFormers' formula and from three optimizers.
        Expectation: MFMuon with the MindFormers split matches the formula within one FP32 ULP; native Muon
            with the same split (BF16 scaling) and native Muon on the whole matrix both miss it.
        """
        fqn = "model.layers.1.self_attn.q_b_proj.weight"
        shape = SMALL_SHAPES[fqn]
        torch.manual_seed(1)
        value, gradient = torch.randn(shape) * 0.02, torch.randn(shape).bfloat16().float()
        expected = mf_update(value, gradient.bfloat16(), mf_rows(fqn, shape[0], SMALL), LEGACY_NS)
        split = make_mf_ns_transform_fn(SMALL)
        self.assertLessEqual(max_ulp(one_step(MFMuon, value, gradient, fqn, ns_transform_fn=split), expected), 1)
        self.assertGreater(max_ulp(one_step(Muon, value, gradient, fqn, ns_transform_fn=split), expected), 4)
        whole = one_step(Muon, value, gradient, fqn)
        self.assertGreater(((whole - expected).abs().max() / (value - expected).abs().max()).item(), 0.1)


class TestMFShardPath(unittest.TestCase):
    """The sharded Muon path keeps MFMuon's FP32 updates through packing and broadcast."""

    @staticmethod
    def sharded_setup() -> tuple:
        """Prepare MFMuon, a linear_qkv parameter, its NS input and a singleton shard assignment."""
        fqn = "model.layers.0.self_attn.linear_qkv.weight"
        shape = SMALL_SHAPES[fqn]
        torch.manual_seed(2)
        value, gradient = torch.randn(shape) * 0.02, torch.randn(shape).bfloat16().float()
        optimizer, parameter = muon_with_parameter(MFMuon, value, gradient, fqn,
                                                   ns_transform_fn=make_mf_ns_transform_fn(SMALL))
        group = optimizer.param_groups[0]
        ns_inputs = optimizer._update_muon_momentum(group, [parameter])  # pylint: disable=protected-access
        assignment = SimpleNamespace(owned_records=[SimpleNamespace(param=parameter, index=0)], shard_pgs=(None,),
                                     layout_spec=SimpleNamespace(shard_axes=((0, 0),)))
        expected = mf_update(value, gradient.bfloat16(), mf_rows(fqn, shape[0], SMALL), LEGACY_NS)
        return optimizer, parameter, group, ns_inputs, assignment, expected

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_sharded_update_matches_mindformers(self):
        """Feature: FP32 update through the sharded Muon path.

        Description: Run _gather_and_compute_shard_updates and _fused_broadcast_and_apply for linear_qkv, split
            into its three MindFormers blocks, on a singleton shard group, and the framework's BF16 broadcast
            packing on the same updates.
        Expectation: The computed updates are FP32 and MFMuon's sharded step matches MindFormers' formula within
            one ULP; the framework packing rounds the same updates to BF16.
        """
        optimizer, parameter, group, ns_inputs, assignment, expected = self.sharded_setup()
        updates, coords = optimizer._gather_and_compute_shard_updates(  # pylint: disable=protected-access
            [parameter], ns_inputs, assignment, {}, group, {})
        self.assertEqual(updates[parameter].dtype, torch.float32)
        framework = torch.nn.Parameter(parameter.data.clone())
        with torch.no_grad():
            Muon._fused_broadcast_and_apply(  # pylint: disable=protected-access
                optimizer, [framework], {framework: updates[parameter]}, {framework: coords[parameter]}, group,
                assignment)
            optimizer._fused_broadcast_and_apply(  # pylint: disable=protected-access
                [parameter], updates, coords, group, assignment)
        self.assertLessEqual(max_ulp(parameter.data, expected), 1)
        self.assertGreater(max_ulp(framework.data, expected), 4)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_multi_rank_broadcast_packs_fp32(self):
        """Feature: FP32 relay broadcast of sharded Muon updates.

        Description: Pretend the shard group has two ranks while the local rank computes the update, and record
            the buffers handed to the relay broadcast by MFMuon and by the framework method.
        Expectation: MFMuon broadcasts an FP32 buffer holding the exact update and applies it unrounded; the
            framework method broadcasts BF16.
        """
        optimizer, parameter, group, ns_inputs, assignment, expected = self.sharded_setup()
        updates, coords = optimizer._gather_and_compute_shard_updates(  # pylint: disable=protected-access
            [parameter], ns_inputs, assignment, {}, group, {})
        framework = torch.nn.Parameter(parameter.data.clone())
        two_ranks = ((2,), (0,), (None,), 2)
        for method, target, dtype in (
                (Muon._fused_broadcast_and_apply, framework, torch.bfloat16),  # pylint: disable=protected-access
                (MFMuon._fused_broadcast_and_apply, parameter, torch.float32)):  # pylint: disable=protected-access
            with patch.object(optimizer, "_get_shard_info", return_value=two_ranks), \
                    patch.object(optimizer, "_batched_relay_broadcast") as relay, torch.no_grad():
                method(optimizer, [target], {target: updates[parameter]}, {target: coords[parameter]}, group,
                       assignment)
            buffer = relay.call_args.args[0][coords[parameter]]
            self.assertEqual(buffer.dtype, dtype)
            torch.testing.assert_close(buffer[:parameter.numel()], updates[parameter].reshape(-1).to(dtype),
                                       rtol=0, atol=0)
        self.assertLessEqual(max_ulp(parameter.data, expected), 1)
        self.assertGreater(max_ulp(framework.data, expected), 4)


class TestMFOptimizerBuilder(unittest.TestCase):
    """build_mf_optimizer assembles the JT optimizer exactly like jt_optimizer.build_optimizer."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_step_matches_mindformers(self):
        """Feature: One MindFormers-aligned optimizer step of the JT model.

        Description: Build with build_mf_optimizer for a model with fused SwiGLU MLPs, give every parameter a
            gradient and step once from fresh state with the recipe's Nesterov momentum 0.95. Newton-Schulz is
            replaced by a per-matrix FP64 normalization so the expectation does not depend on BF16 matmul batching.
        Expectation: Every Muon parameter equals MindFormers' update (muon_split_fn blocks, per-block
            normalization, FP32 step factor) within one FP32 ULP.
        """
        model = mf_model(fused=True)
        optimizer = build(model, zeropower_fn=normalize_ns).get_optimizer()
        muon_ids = {id(parameter) for group in optimizer.chained_optimizers[-1].param_groups
                    for parameter in group["params"]}
        set_gradients(model, seed=3)
        expected = {}
        for name, parameter in model.named_parameters():
            if id(parameter) in muon_ids:
                # First Nesterov step from zero momentum: g + 0.95 * g.
                ns_input = (parameter.grad * 0.95 + parameter.grad).bfloat16()
                expected[name] = mf_update(parameter.detach(), ns_input,
                                           mf_rows(name, parameter.shape[-2], model.config),
                                           partial(normalize_ns, steps=5))
        self.assertIn("model.layers.0.mlp.linear_fc1.weight", expected)
        optimizer.step()
        for name, parameter in model.named_parameters():
            if name in expected:
                self.assertLessEqual(max_ulp(parameter.detach(), expected[name]), 1, name)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_builder_registers_the_jt_post_step_hook(self):
        """Feature: build_mf_optimizer assembly and post-step hook.

        Description: Build with lr 0 and compare with jt_optimizer.build_optimizer, then step behind the FP32
            main-parameter wrapper while head 0 exceeds the QK-clip threshold 4x.
        Expectation: Native AdamW and MFMuon are chained, MFMuon splits and folds lr, its post-step hook is the
            native one with the same threshold, and the step clips head 0 once and updates the expert bias once.
        """
        model = mf_model(fused=False)
        builder = build(model, lr=0.0)
        adamw, muon = builder.get_optimizer().chained_optimizers
        self.assertIs(type(adamw), AdamW)
        self.assertIs(type(muon), MFMuon)
        self.assertIsNotNone(muon.ns_transform_fn)
        self.assertTrue(all(group["apply_lr_in_update"] for group in muon.param_groups))
        native = jt_optimizer.build_optimizer(model=mf_model(fused=False), qk_clip_threshold=THRESHOLD,
                                              muon_config=MUON_CONFIG, adamw_config=ADAMW_CONFIG)
        hooks = [list(leaf._optimizer_step_post_hooks.values())  # pylint: disable=protected-access
                 for leaf in (muon, native.get_optimizer().chained_optimizers[-1])]
        self.assertEqual([(hook.func, hook.args[1]) for hook in hooks[0]],
                         [(hook.func, hook.args[1]) for hook in hooks[1]])
        self.assertEqual([hook.args[0] for hook in hooks[0]], [model])

        optimizer = Float16OptimizerWithFloat16Params(builder.get_optimizer(), model)
        originals = []
        for module in attention_modules(model):
            module.max_logits_val = torch.tensor([4 * THRESHOLD, THRESHOLD / 2])
            originals.append((module.q_b_proj.weight.detach().clone(), module.kv_b_proj.weight.detach().clone()))
        with patch.object(jt_optimizer, "_update_expert_bias", wraps=jt_optimizer._update_expert_bias) as bias:
            optimizer.step()
        bias.assert_called_once_with(model)
        for module, (query, key_value) in zip(attention_modules(model), originals):
            self.assertTrue(torch.equal(module.q_b_proj.weight, query * row_factors([0.25, 1.0], rope_scaled=True)))
            self.assertTrue(torch.equal(module.kv_b_proj.weight,
                                        key_value * row_factors([0.25, 1.0], rope_scaled=False)))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_recipe_switches_by_target_only(self):
        """Feature: YAML switch from jt_optimizer.build_optimizer to build_mf_optimizer.

        Description: Take the JT recipe's optimizer section, replace only `_target_`, resolve it again and build it
            with the Trainer's runtime model argument.
        Expectation: Both builders have the same signature and the unchanged keys build an MFOptimizerBuilder.
        """
        signatures = [inspect.signature(function).parameters.values()
                      for function in (jt_optimizer.build_optimizer, build_mf_optimizer)]
        parameters = [[(item.name, item.kind, item.annotation) for item in signature] for signature in signatures]
        self.assertEqual(parameters[0], parameters[1])
        recipe = jt_test_config().optimizer.to_dict()
        recipe["_target_"] = "hyper_parallel.models.jt_deepseek_v3.adapter.runtime.mf_optimizer.build_mf_optimizer"
        switched = resolve_component(recipe, annotation=OptimizerConfig, path="$.optimizer")
        self.assertTrue(switched.fp32_main_params)
        self.assertIsInstance(switched.target.build(model=mf_model(fused=False)), MFOptimizerBuilder)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_example_recipe_resolves_to_mf_optimizer_builder(self):
        """Feature: hp_4k_1_card_mf_muon.yaml, the example single-card recipe for build_mf_optimizer.

        Description: Load the real recipe file from disk and resolve only its optimizer section.
        Expectation: It builds an MFOptimizerBuilder whose Muon leaf is MFMuon.
        """
        root = Path(__file__).resolve().parents[4]
        path = root / "examples/training_demo/jt_deepseek_v3/hp_4k_1_card_mf_muon.yaml"
        document = yaml.safe_load(path.read_text())
        self.assertEqual(document["optimizer"]["_target_"],
                         "hyper_parallel.models.jt_deepseek_v3.adapter.runtime.mf_optimizer.build_mf_optimizer")
        resolved = resolve_component(document["optimizer"], annotation=OptimizerConfig, path="$.optimizer")
        builder = resolved.target.build(model=mf_model(fused=False))
        self.assertIsInstance(builder, MFOptimizerBuilder)
        self.assertIsInstance(builder.get_optimizer().chained_optimizers[-1], MFMuon)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_builder_rejects_unsupported_settings(self):
        """Feature: build_mf_optimizer configuration checks.

        Description: Build with a caller-provided Newton-Schulz split or reshape, and with a zero or NaN QK-clip
            threshold.
        Expectation: Each raises ValueError naming the offending setting.
        """
        cases = [({"ns_transform_fn": lambda fqn, tensor: None}, "remove muon_config.ns_transform_fn"),
                 ({"reshape_fn": lambda fqn, tensor: [tensor]}, "does not accept reshape_fn")]
        for overrides, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                build(mf_model(fused=False), **overrides)
        for threshold in (0.0, float("nan")):
            with self.assertRaisesRegex(ValueError, "qk_clip_threshold"):
                build_mf_optimizer(model=mf_model(fused=False), qk_clip_threshold=threshold, muon_config=MUON_CONFIG,
                                   adamw_config=ADAMW_CONFIG)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_builder_accepts_any_newton_schulz_settings(self):
        """Feature: build_mf_optimizer leaves Newton-Schulz settings to the recipe and lr folding to MFMuon.

        Description: Build with asym5 coefficients, with six steps, with a muon_config leaving ns_variant and
            ns_epsilon to Muon's defaults (asym5, 1e-10) and with apply_lr_in_update=False, then step once.
        Expectation: Each builds an MFOptimizerBuilder whose ChainedOptimizer holds an MFMuon with the configured
            Newton-Schulz settings and lr folded into the update, and the step updates every Muon parameter.
        """
        muon_defaults = {key: value for key, value in MUON_CONFIG.items() if key not in ("ns_variant", "ns_epsilon")}
        cases = {
            "asym5": ({**MUON_CONFIG, "ns_variant": "asym5"}, {"ns_variant": "asym5", "ns_steps": 5}),
            "six steps": ({**MUON_CONFIG, "ns_steps": 6}, {"ns_variant": "legacy", "ns_steps": 6}),
            "Muon defaults": (muon_defaults, {"ns_variant": "asym5", "ns_epsilon": 1e-10}),
            "apply_lr_in_update=False": ({**MUON_CONFIG, "apply_lr_in_update": False}, {"ns_variant": "legacy"}),
        }
        for label, (muon_config, settings) in cases.items():
            with self.subTest(case=label):
                model = mf_model(fused=False)
                builder = build_mf_optimizer(model=model, qk_clip_threshold=THRESHOLD, muon_config=muon_config,
                                             adamw_config=ADAMW_CONFIG)
                self.assertIsInstance(builder, MFOptimizerBuilder)
                optimizer = builder.get_optimizer()
                self.assertIsInstance(optimizer, ChainedOptimizer)
                muon = optimizer.chained_optimizers[-1]
                self.assertIs(type(muon), MFMuon)
                for group in muon.param_groups:
                    self.assertEqual({key: group[key] for key in settings}, settings)
                    self.assertTrue(group["apply_lr_in_update"])
                parameters = [parameter for group in muon.param_groups for parameter in group["params"]]
                before = [parameter.data.clone() for parameter in parameters]
                set_gradients(model, seed=5)
                optimizer.step()
                for parameter, value in zip(parameters, before):
                    self.assertTrue(torch.isfinite(parameter.data).all())
                    self.assertFalse(torch.equal(parameter.data, value))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_state_dict_round_trip_keeps_momentum_step_and_lr(self):
        """Feature: Checkpoint round trip of the MindFormers-aligned optimizer.

        Description: Step once and load the chained state into a fresh MF optimizer; separately load a checkpoint
            written by jt_optimizer.build_optimizer, which records apply_lr_in_update=False.
        Expectation: Momentum buffers, step and lr load unchanged from both checkpoints, and the loaded MFMuon
            keeps lr folded into its FP32 update.
        """
        native_model = mf_model(fused=False)
        native = jt_optimizer.build_optimizer(model=native_model, qk_clip_threshold=THRESHOLD,
                                              muon_config=MUON_CONFIG, adamw_config=ADAMW_CONFIG)
        sources = {"mf": (mf_model(fused=False), None), "native": (native_model, native)}
        for label, (source_model, source_builder) in sources.items():
            with self.subTest(source=label):
                source = (source_builder or build(source_model)).get_optimizer()
                set_gradients(source_model, seed=4)
                source.step()
                state = source.state_dict()
                self.assertEqual(any(key.endswith(".apply_lr_in_update") and value is False
                                     for key, value in state.items()), label == "native")
                target_model = mf_model(fused=False)
                target = build(target_model).get_optimizer()
                target.load_state_dict(state)
                source_muon, target_muon = source.chained_optimizers[-1], target.chained_optimizers[-1]
                target_parameters = dict(target_model.named_parameters())
                for name, parameter in source_model.named_parameters():
                    if parameter in source_muon.state:
                        torch.testing.assert_close(target_muon.state[target_parameters[name]]["momentum_buffer"],
                                                   source_muon.state[parameter]["momentum_buffer"], rtol=0, atol=0)
                for group in target_muon.param_groups:
                    self.assertEqual((group["step"], group["lr"], group["apply_lr_in_update"]), (1, LR, True))


if __name__ == "__main__":
    unittest.main()
