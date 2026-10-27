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
"""AdamW counters must follow each FP32 master's effective updates."""

import copy
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.optimizer.adamw import AdamW, adamw
from hyper_parallel.core.optimizer.swap_optimizer import SwapOptimizerConfig, swap_optimizer


class TestAdamWParameterSteps(unittest.TestCase):
    """Compare dense CPU updates and checkpoint continuation with Torch AdamW."""

    @staticmethod
    def _make_pair(**options):
        values = (1.0, 2.0, -0.5)
        params = [torch.tensor([value]) for value in values]
        reference_params = [param.clone() for param in params]
        config = {"lr": 0.1, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.1, **options}
        optimizer = AdamW(params, **config)
        reference = torch.optim.AdamW(reference_params, foreach=False, fused=False, **config)
        return optimizer, reference

    @staticmethod
    def _step_pair(optimizer, reference, gradients):
        for target in (optimizer, reference):
            for param, value in zip(target.param_groups[0]["params"], gradients):
                param.grad = None if value is None else torch.full_like(param, value)
            with torch.no_grad():
                target.step()

    def _assert_matches(self, optimizer, reference):
        pairs = list(zip(optimizer.param_groups[0]["params"], reference.param_groups[0]["params"]))
        for param, reference_param in pairs:
            torch.testing.assert_close(param, reference_param, rtol=2e-6, atol=2e-7)
        for param, reference_param in pairs:
            state, expected = optimizer.state.get(param, {}), reference.state.get(reference_param, {})
            if not expected:
                self.assertFalse(state, f"expected an uninitialized state, got={state}")
                continue
            expected_step = int(expected["step"].item())
            self.assertEqual(state["step"], expected_step, f"expected={expected_step}, got={state['step']}")
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if name in expected:
                    torch.testing.assert_close(state[name], expected[name], rtol=2e-6, atol=2e-7)

    def test_none_and_zero_gradients_use_effective_parameter_steps(self) -> None:
        """Delayed activation and later gaps skip counters, while a zero gradient advances them."""
        for amsgrad, maximize in ((False, False), (True, True)):
            with self.subTest(amsgrad=amsgrad, maximize=maximize):
                optimizer, reference = self._make_pair(amsgrad=amsgrad, maximize=maximize)
                for gradients in ((1.0, None, 0.0), (-2.0, 1.0, None), (0.5, None, None), (None, -0.25, 1.0)):
                    self._step_pair(optimizer, reference, gradients)
                self._assert_matches(optimizer, reference)

    def test_checkpoint_roundtrip_preserves_different_parameter_steps(self) -> None:
        """Resume uneven counters without turning missing gradients into updates."""
        for tensor_steps in (False, True):
            with self.subTest(tensor_steps=tensor_steps):
                optimizer, reference = self._make_pair()
                for gradients in ((1.0, None, 1.0), (1.0, -0.5, None), (1.0, None, None)):
                    self._step_pair(optimizer, reference, gradients)
                saved = copy.deepcopy(optimizer.state_dict())
                if tensor_steps:
                    for state in saved["state"].values():
                        state["step"] = torch.tensor(float(state["step"]))
                restored, _ = self._make_pair()
                for target, source in zip(restored.param_groups[0]["params"], optimizer.param_groups[0]["params"]):
                    target.copy_(source)
                restored.load_state_dict(saved)
                self._step_pair(restored, reference, (None, 0.25, -1.0))
                self._assert_matches(restored, reference)

    def test_legacy_checkpoint_recovers_initialized_group_steps(self) -> None:
        """Legacy moments inherit the saved group count; an absent state starts at zero."""
        optimizer, reference = self._make_pair()
        for _ in range(3):
            self._step_pair(optimizer, reference, (1.0, None, -0.5))
        saved = copy.deepcopy(optimizer.state_dict())
        for state in saved["state"].values():
            state.pop("step", None)
        for group in saved['param_groups']:
            group.pop('_hp_adamw_parameter_steps', None)
        restored, _ = self._make_pair()
        for target, source in zip(restored.param_groups[0]["params"], optimizer.param_groups[0]["params"]):
            target.copy_(source)
        restored.load_state_dict(saved)
        self._step_pair(restored, reference, (0.5, 1.0, -0.5))
        self._assert_matches(restored, reference)

    def test_uniform_steps_keep_one_fused_call_per_group(self) -> None:
        """The usual all-active case keeps a single fused batch."""
        optimizer, reference = self._make_pair()
        with patch("hyper_parallel.core.optimizer.adamw.adamw", wraps=adamw) as fused:
            for _ in range(3):
                self._step_pair(optimizer, reference, (1.0, 0.5, -0.5))
        self.assertEqual(fused.call_count, 3, f"expected=3 fused calls, got={fused.call_count}")
        self._assert_matches(optimizer, reference)

    def test_swap_preserves_parameter_steps_and_checkpoint_continuation(self) -> None:
        """Both swap preparation modes keep counters independent within a batch."""
        for packed_swap in (False, True):
            with self.subTest(packed_swap=packed_swap):
                optimizer, reference = self._make_pair()
                config = SwapOptimizerConfig(swap_times=1, min_numel=0, packed_swap=packed_swap)
                wrapped = swap_optimizer(optimizer, config)
                for gradients in ((1.0, 1.0, 1.0), (1.0, None, None), (1.0, None, 0.0)):
                    self._step_pair(wrapped, reference, gradients)
                self._assert_matches(wrapped, reference)
                saved = copy.deepcopy(wrapped.state_dict())
                restored_optimizer, _ = self._make_pair()
                restored = swap_optimizer(restored_optimizer, config)
                for target, source in zip(restored.param_groups[0]["params"], wrapped.param_groups[0]["params"]):
                    target.copy_(source)
                restored.load_state_dict(saved)
                self._step_pair(restored, reference, (None, 0.5, 1.0))
                self._assert_matches(restored, reference)

    def test_swap_legacy_checkpoint_fixes_counts_before_later_activation(self) -> None:
        """Legacy moments inherit the load-time count even after more skipped steps."""
        for packed_swap, legacy_marker in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(packed_swap=packed_swap, legacy_marker=legacy_marker):
                optimizer, reference = self._make_pair()
                config = SwapOptimizerConfig(swap_times=1, min_numel=0, packed_swap=packed_swap)
                wrapped = swap_optimizer(optimizer, config)
                for _ in range(3):
                    self._step_pair(wrapped, reference, (1.0, 1.0, 1.0))
                saved = copy.deepcopy(wrapped.state_dict())
                for state in saved['state'].values():
                    if legacy_marker:
                        state['step'] = None
                    else:
                        state.pop('step', None)
                for group in saved['param_groups']:
                    group.pop('_hp_adamw_parameter_steps', None)
                restored_optimizer, _ = self._make_pair()
                restored = swap_optimizer(restored_optimizer, config)
                for target, source in zip(restored.param_groups[0]['params'], wrapped.param_groups[0]['params']):
                    target.copy_(source)
                restored.load_state_dict(saved)
                for _ in range(5):
                    self._step_pair(restored, reference, (1.0, None, 0.0))
                self._step_pair(restored, reference, (None, 0.5, 1.0))
                self._assert_matches(restored, reference)
