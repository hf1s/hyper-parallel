# Copyright 2026 Huawei Technologies Co., Ltd.
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

"""Check the archived optimizer recipe independently of saved training outputs."""
# The Transformers config signature is generated dynamically; keyword checks are false positives here.
# pylint: disable=unexpected-keyword-arg

import math
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.models.jt_deepseek_v3.adapter.runtime import archived_optimizer
from hyper_parallel.models.jt_deepseek_v3.adapter.runtime.archived_optimizer import (
    ArchivedAdamW, ArchivedMuon, ArchivedOptimizerBuilder, _logical_matrices, archived_adamw, archived_ns,
    normalize_ns,
)
from tests.common.mark_utils import arg_mark
from tests.ut.models.jt_deepseek_v3.test_jt_optimizer import attention_modules, replaced_model, row_factors

THRESHOLD = 100.0
ARCHIVED_MUON = {"lr": 0.0, "ns_variant": "legacy", "ns_epsilon": 1e-7}


def single_rank() -> MagicMock:
    """Stand in for an initialized one-rank process group inside the archived optimizer module."""
    return MagicMock(is_initialized=MagicMock(return_value=True), get_world_size=MagicMock(return_value=1))


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
class TestArchivedOptimizer(unittest.TestCase):
    """Archived NS normalization, momentum restore and AdamW step contracts."""

    def test_adamw_matches_double_precision_mathematical_step(self):
        """A changed accumulation order must preserve AdamW's state and step meaning."""
        parameter = torch.tensor([.8, -1.2, .3], dtype=torch.float32)
        gradient = torch.tensor([.2, -.1, -.3], dtype=torch.float32)
        moment = torch.tensor([.1, -.2, .01], dtype=torch.float32)
        variance = torch.tensor([.02, .04, .001], dtype=torch.float32)
        group = {'betas': (.9, .95), 'lr': .0001, 'weight_decay': .1, 'eps': 1e-8, 'step': 201}
        m64 = moment.double() * .9 + gradient.double() * .1
        v64 = variance.double() * .95 + gradient.double().square() * .05
        expected = parameter.double() * (1 - .0001 * .1)
        expected -= (.0001 / (1 - .9 ** 201)) * m64 / ((v64 / (1 - .95 ** 201)).sqrt() + 1e-8)
        archived_adamw(parameter, gradient, moment, variance, group)
        torch.testing.assert_close(parameter.double(), expected, rtol=2e-7, atol=2e-8)
        torch.testing.assert_close(moment.double(), m64, rtol=3e-7, atol=2e-8)
        torch.testing.assert_close(variance.double(), v64, rtol=2e-7, atol=2e-8)

    def test_expert_denominator_rounds_norm_and_epsilon_together(self):
        """An independent FP64 reference detects premature BF16 norm rounding."""
        values = torch.tensor([1., .125, .375, 2.25], dtype=torch.bfloat16).reshape(1, 2, 2) * 2.**-18
        reference_norm = values.double().square().sum().sqrt()
        expected = (values / (reference_norm + 1e-7).bfloat16()).bfloat16()
        early_rounded = values / (values.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        self.assertFalse(torch.equal(expected, early_rounded))
        torch.testing.assert_close(normalize_ns(values), expected, rtol=0, atol=0)

    def test_restore_keeps_momentum_and_step_while_selecting_lr_representation(self):
        """Legacy execution options cannot reset or replace the optimizer anchor."""
        parameter = torch.nn.Parameter(torch.zeros(2, 3))
        optimizer = object.__new__(ArchivedMuon)
        defaults = {'apply_lr_in_update': False, 'step': 200, 'lr': .0001,
                    'momentum': (.95, .95), 'ns_variant': 'legacy', 'ns_epsilon': 1e-7, 'nesterov': True}
        # Bypass only the initialized-process-group constructor; exercise real state restore.
        torch.optim.Optimizer.__init__(optimizer, [parameter], defaults)  # pylint: disable=unnecessary-dunder-call
        buffer = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        optimizer.state[parameter]['momentum_buffer'] = buffer.clone()
        state = optimizer.state_dict()
        optimizer.load_state_dict(state)
        torch.testing.assert_close(optimizer.state[parameter]['momentum_buffer'], buffer, rtol=0, atol=0)
        self.assertEqual(optimizer.param_groups[0]['step'], 200)
        self.assertEqual(optimizer.param_groups[0]['lr'], .0001)
        self.assertEqual(optimizer.restored_lr_folding, [False])
        self.assertTrue(optimizer.param_groups[0]['apply_lr_in_update'])
        parameter.grad = torch.full((2, 3), .125)
        update = optimizer._update_muon_momentum(optimizer.param_groups[0], [parameter])  # pylint: disable=protected-access
        expected_buffer = buffer * .95 + parameter.grad
        torch.testing.assert_close(optimizer.state[parameter]['momentum_buffer'], expected_buffer, rtol=0, atol=0)
        torch.testing.assert_close(update[parameter], parameter.grad + expected_buffer * .95, rtol=0, atol=0)

    def test_fused_broadcast_and_apply_preserves_fp32_without_framework_change(self):
        """The singleton-shard write-back stays local to ArchivedMuon, not framework Muon.

        Framework ``Muon._fused_broadcast_and_apply`` always packs updates through a
        BF16 buffer before any possible broadcast, rounding away precision even when
        ``total_shard_size == 1`` and no communication happens. ArchivedMuon overrides
        only this method; the dtype-preserving behavior must not depend on any change
        to ``core/optimizer/muon.py``, which this recipe leaves untouched.
        """
        base_parameters = [torch.tensor([[0.01, 0.02], [0.03, 0.04]]), torch.tensor([[0.1, -0.2, 0.3]])]
        updates = [torch.tensor([[0.1234567, -0.2345678], [0.3456789, -0.456789]]),
                   torch.tensor([[0.1234567, 0.3456789, -0.5678912]])]
        originals = [update.clone() for update in updates]
        for weight_decay in (0.0, 0.1):
            with self.subTest(weight_decay=weight_decay):
                parameters = [torch.nn.Parameter(value.clone()) for value in base_parameters]
                group = {'lr': 1e-4, 'weight_decay': weight_decay, 'apply_lr_in_update': True}
                expected = [value.clone().mul_(1 - group['lr'] * weight_decay).add_(update)
                           for value, update in zip(base_parameters, updates)]
                # BF16-rounding the update first, as framework Muon would, must differ from the FP32 result.
                rounded = [value.clone().mul_(1 - group['lr'] * weight_decay).add_(update.bfloat16().float())
                          for value, update in zip(base_parameters, updates)]
                self.assertFalse(any(torch.equal(a, b) for a, b in zip(expected, rounded)))

                optimizer = object.__new__(ArchivedMuon)
                optimizer._param_shard_metadata = {}  # pylint: disable=protected-access
                with patch.object(optimizer, '_get_shard_info', return_value=((1,), (0,), (None,), 1)), \
                        torch.no_grad():
                    optimizer._fused_broadcast_and_apply(  # pylint: disable=protected-access
                        parameters, dict(zip(parameters, updates)), None, group, SimpleNamespace(layout_spec=None))
                for actual, reference in zip(parameters, expected):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                for update, original in zip(updates, originals):
                    torch.testing.assert_close(update, original, rtol=0, atol=0)

    def test_fused_broadcast_and_apply_rejects_a_multi_rank_shard_group(self):
        """ArchivedMuon is restricted to single-rank runs; it must not silently broadcast."""
        optimizer = object.__new__(ArchivedMuon)
        optimizer._param_shard_metadata = {}  # pylint: disable=protected-access
        parameter = torch.nn.Parameter(torch.zeros(2, 2))
        group = {'lr': 1e-4, 'weight_decay': 0.0, 'apply_lr_in_update': True}
        with patch.object(optimizer, '_get_shard_info', return_value=((2,), (0,), (MagicMock(),), 2)):
            with self.assertRaisesRegex(ValueError, 'singleton shard group'):
                optimizer._fused_broadcast_and_apply(  # pylint: disable=protected-access
                    [parameter], {parameter: torch.zeros(2, 2)}, None, group, SimpleNamespace(layout_spec=None))

    def test_builder_registers_the_native_qk_clip_hook_after_archived_updates(self):
        """The archived YAML target clips QK projections once per step, exactly like the default target."""
        model = replaced_model(torch.float32)
        with patch.object(archived_optimizer, "dist", single_rank()):
            builder = ArchivedOptimizerBuilder(model=model, muon_config=ARCHIVED_MUON,
                                               adamw_config={"adamw_lr": 0.0}, qk_clip_threshold=THRESHOLD)
        optimizer = Float16OptimizerWithFloat16Params(builder.get_optimizer(), model)
        self.assertEqual([type(leaf) for leaf in builder.get_optimizer().chained_optimizers],
                         [ArchivedAdamW, ArchivedMuon])
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.5)
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
        for invalid in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(qk_clip_threshold=invalid), self.assertRaisesRegex(ValueError, "qk_clip_threshold"):
                ArchivedOptimizerBuilder(model=model, muon_config=ARCHIVED_MUON, adamw_config={},
                                         qk_clip_threshold=invalid)

    def test_unsupported_options_fail_at_construction(self):
        """Options the archived arithmetic ignores or cannot honour are rejected before any step runs."""
        parameters = [torch.nn.Parameter(torch.zeros(4, 4))]
        invalid_options = {
            "ns_variant": {"ns_variant": "asym5", "ns_epsilon": 1e-7},
            "ns_epsilon": {"ns_variant": "legacy", "ns_epsilon": 1e-6},
            "momentum": {"ns_variant": "legacy", "ns_epsilon": 1e-7, "momentum": (0.9, 0.95)},
            "apply_lr_in_update": {"ns_variant": "legacy", "ns_epsilon": 1e-7, "apply_lr_in_update": False},
            "momentum_update_fn": {"ns_variant": "legacy", "ns_epsilon": 1e-7, "momentum_update_fn": print},
        }
        with patch.object(archived_optimizer, "dist", single_rank()):
            ArchivedMuon(parameters, model_config=None, **ARCHIVED_MUON)
            for name, options in invalid_options.items():
                with self.subTest(option=name), self.assertRaises(ValueError):
                    ArchivedMuon(parameters, model_config=None, **options)
        two_ranks = MagicMock(is_initialized=MagicMock(return_value=True), get_world_size=MagicMock(return_value=2))
        with patch.object(archived_optimizer, "dist", two_ranks), self.assertRaisesRegex(ValueError, "one initialized"):
            ArchivedMuon(parameters, model_config=None, **ARCHIVED_MUON)

    def test_logical_matrices_split_the_archived_schema_and_restore_storage(self):
        """Fused JT storage splits into the archived Q/KV/RoPE, per-head and gate/up blocks and restores exactly."""
        config = SimpleNamespace(q_lora_rank=3, kv_lora_rank=2, qk_rope_head_dim=1, qk_nope_head_dim=2,
                                 v_head_dim=3, num_attention_heads=2)
        cases = {
            "model.layers.0.self_attn.linear_qkv.weight": ((6, 5), [(3, 5), (2, 5), (1, 5)]),
            "model.layers.0.self_attn.q_b_proj.weight": ((6, 5), [(4, 5), (2, 5)]),
            "model.layers.0.self_attn.kv_b_proj.weight": ((10, 5), [(4, 5), (6, 5)]),
            "model.layers.1.mlp.experts.gate_up_proj": ((2, 8, 5), [(2, 4, 5), (2, 4, 5)]),
            "model.layers.0.self_attn.o_proj.weight": ((5, 4), [(5, 4)]),
        }
        for name, (shape, piece_shapes) in cases.items():
            with self.subTest(name=name):
                value = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)
                transform = _logical_matrices(name, value, config)
                self.assertEqual([tuple(piece.shape) for piece in transform.tensors], piece_shapes)
                restored = torch.zeros_like(value)
                transform.restore([piece.clone() for piece in transform.tensors], restored)
                torch.testing.assert_close(restored, value, rtol=0, atol=0)
        heads = _logical_matrices("model.layers.0.self_attn.q_b_proj.weight",
                                  torch.arange(30, dtype=torch.float32).reshape(6, 5), config).tensors
        # Head h owns rows [3h, 3h + 3): nope rows 3h, 3h + 1 go to the first block, rope row 3h + 2 to the second.
        torch.testing.assert_close(heads[1][:, 0], torch.tensor([10., 25.]), rtol=0, atol=0)
        torch.testing.assert_close(heads[0][:, 0], torch.tensor([0., 5., 15., 20.]), rtol=0, atol=0)

    def test_adamw_step_advances_groups_and_skips_parameters_without_gradients(self):
        """Each step increments the group step once, creates zero moments lazily and leaves gradient-free tensors."""
        updated = torch.nn.Parameter(torch.tensor([0.5, -0.25]))
        frozen = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        optimizer = ArchivedAdamW([updated, frozen], lr=0.1, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
        updated.grad = torch.tensor([0.2, -0.4])
        expected = updated.data.clone()
        moment, variance = torch.zeros(2), torch.zeros(2)
        for step in (1, 2):
            archived_adamw(expected, updated.grad, moment, variance, {**optimizer.param_groups[0], "step": step})
            optimizer.step()
            self.assertEqual(optimizer.param_groups[0]["step"], step)
        torch.testing.assert_close(updated.data, expected, rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state[updated]["exp_avg"], moment, rtol=0, atol=0)
        torch.testing.assert_close(frozen.data, torch.tensor([1.0, 2.0]), rtol=0, atol=0)
        self.assertNotIn(frozen, optimizer.state)

    def test_q_up_blocks_are_orthogonalized_separately_with_the_archived_last_block_scale(self):
        """q_b_proj nope/rope rows are separate NS problems, both scaled by the rope block shape, re-interleaved."""
        config = SimpleNamespace(q_lora_rank=3, kv_lora_rank=2, qk_rope_head_dim=2, qk_nope_head_dim=3,
                                 v_head_dim=3, num_attention_heads=2)
        parameter = torch.nn.Parameter(torch.zeros(10, 3))
        parameter.model_name = "model.layers.0.self_attn.q_b_proj.weight"
        torch.manual_seed(258)
        momentum = torch.randn(10, 3)
        group = {"apply_lr_in_update": True, "ns_coefficients": None, "ns_steps": 5, "lr": 0.01,
                 "matched_adamw_rms": 0.2}
        optimizer = object.__new__(ArchivedMuon)
        optimizer.model_config = config
        actual = optimizer._compute_batched_ns_updates(  # pylint: disable=protected-access
            [parameter], {parameter: momentum.clone()}, group)[parameter]
        heads = momentum.reshape(2, 5, 3)
        # Archived quirk: both blocks use sqrt(max(rope block shape)) = sqrt(4), not their own sqrt(6) / sqrt(4).
        scale = np.float32(np.float32(np.float32(math.sqrt(4.0)) * np.float32(0.2)) * np.float32(-0.01))
        nope, rope = (archived_ns(block.reshape(-1, 3).bfloat16().float(), 5) * float(scale)
                      for block in (heads[:, :3], heads[:, 3:]))
        expected = torch.cat((nope.reshape(2, 3, 3), rope.reshape(2, 2, 3)), dim=1).reshape(10, 3)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
