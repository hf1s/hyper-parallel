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

"""Check archive arithmetic contracts independently of saved training outputs."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hyper_parallel.models.jt_deepseek_v3.mf_archive import model as archive_model
from hyper_parallel.models.jt_deepseek_v3.mf_archive.layers import (
    RoutingNormalization, ArithmeticSigmoid, SharedKeyHalfTree,
)
from hyper_parallel.models.jt_deepseek_v3.mf_archive.optimizer import ArchivedMuon, archived_adamw
from hyper_parallel.models.jt_deepseek_v3.mf_archive.reductions import (
    half_tree, adjacent_tree, tiled_square_sum, normalize_ns,
)
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.text_trainer import TextTrainer
from tests.common.mark_utils import arg_mark


@arg_mark(['platform_ascend910b'], 'level0', 'dryrun', 'essential')
class TestArchivedArithmetic(unittest.TestCase):
    """Distinguish arithmetic compatibility from altered mathematical gradients."""

    def test_shared_key_expands_heads_not_tokens(self):
        """The current MLA projection contract is token-major, including its backward."""
        values = torch.arange(30, dtype=torch.float32).reshape(2, 5, 1, 3).requires_grad_()
        actual = SharedKeyHalfTree.apply(values, 8)
        expected = values.expand(2, 5, 8, 3)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        weights = torch.arange(actual.numel(), dtype=torch.float32).reshape_as(actual)
        (actual * weights).sum().backward()
        torch.testing.assert_close(values.grad, weights.sum(2, keepdim=True), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'singleton shared head'):
            SharedKeyHalfTree.apply(torch.zeros(1, 1, 5, 3), 8)

    def test_routing_derivative_matches_double_precision_objective(self):
        """The reordered router backward must still differentiate both consumers."""
        torch.manual_seed(47)
        inputs = torch.randn(3, 8, dtype=torch.float64)
        indices = inputs.topk(2, dim=-1).indices
        first, second = torch.randn(3, 2, dtype=torch.float64), torch.randn(3, 8, dtype=torch.float64)
        actual_input = inputs.clone().requires_grad_()
        scores = ArithmeticSigmoid.apply(actual_input, 'gradient-first')
        selected, normalized = RoutingNormalization.apply(scores, indices, 2.827)
        (selected.mul(first).sum() + normalized.mul(second).sum()).backward()
        expected_input = inputs.clone().requires_grad_()
        scores = expected_input.sigmoid()
        selected = scores.gather(1, indices)
        loss = (selected / selected.sum(-1, keepdim=True) * 2.827).mul(first).sum()
        loss = loss + (scores / scores.sum(-1, keepdim=True)).mul(second).sum()
        loss.backward()
        torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=1e-12, atol=1e-12)

    def test_reduction_includes_odd_tail_and_partial_tiles(self):
        """Every element contributes, including non-power-of-two and final-tile tails."""
        values = torch.arange(1, 6, dtype=torch.float32)
        self.assertEqual(half_tree(values).item(), 15)
        self.assertEqual(adjacent_tree(values).item(), 15)
        ones = torch.ones(8144 * 2 + 17, dtype=torch.float32)
        self.assertEqual(tiled_square_sum(ones, 8144, 16), np.float32(ones.numel()))
        ones[-1] = 2
        self.assertEqual(tiled_square_sum(ones, 8144, 16), np.float32(ones.numel() + 3))

    def test_expert_denominator_rounds_norm_and_epsilon_together(self):
        """An independent FP64 reference detects premature BF16 norm rounding."""
        values = torch.tensor([1., .125, .375, 2.25], dtype=torch.bfloat16).reshape(1, 2, 2) * 2.**-18
        reference_norm = values.double().square().sum().sqrt()
        expected = (values / (reference_norm + 1e-7).bfloat16()).bfloat16()
        early_rounded = values / (values.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        self.assertFalse(torch.equal(expected, early_rounded))
        torch.testing.assert_close(normalize_ns(values), expected, rtol=0, atol=0)

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

    def test_source_has_no_experiment_imports_or_runtime_code_generation(self):
        """A clean profile must stand alone and must not rewrite imported module functions."""
        for source in Path(archive_model.__file__).parent.glob('*.py'):
            tree = ast.parse(source.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, {'exec', 'eval', 'compile'})
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith(('capture_', 'muon_', 'hp_', 'one_factor')))
                if isinstance(node, ast.Attribute):
                    self.assertNotEqual(node.attr, '__class__')
            self.assertNotIn('P12_', source.read_text())


@arg_mark(['platform_ascend910b'], 'level0', 'dryrun', 'essential')
class TestExplicitTrainerBase(unittest.TestCase):
    """Verify the narrow constructor seam and the untouched default."""

    def test_default_and_specialized_base_are_selected_before_setup(self):
        """No global BaseTrainer substitution is needed to specialize one stage."""
        class StopSetup(RuntimeError):
            pass

        class Specialized(BaseTrainer):
            pass

        for selected in (BaseTrainer, Specialized):
            seen = []

            def setup(instance: BaseTrainer) -> None:
                """Observe the selected base before any accelerator setup occurs."""
                seen.append(type(instance))
                raise StopSetup

            with patch.object(BaseTrainer, '_setup', autospec=True, side_effect=setup):
                with self.assertRaises(StopSetup):
                    if selected is BaseTrainer:
                        TextTrainer(SimpleNamespace())
                    else:
                        TextTrainer(SimpleNamespace(), base_trainer_class=selected)
            self.assertEqual(seen, [selected])

    def test_invalid_base_is_rejected_before_setup(self):
        with self.assertRaisesRegex(ValueError, 'BaseTrainer subclass'):
            TextTrainer(SimpleNamespace(), base_trainer_class=object)
