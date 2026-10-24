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
"""Offline recovery contract checks with real CPU optimizers and FP32 masters."""

import copy
import unittest
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from hyper_parallel import DTensor, DeviceMesh, Replicate
from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.core.optimizer.adamw import AdamW as HyperAdamW
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from scripts.checkpoint_conversion.optimizer_state import restore_optimizer_state


class _Model(torch.nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        """Build matrix and scalar parameters for recovery checks."""
        super().__init__()
        self.matrix = torch.nn.Parameter(torch.zeros((2, 3), dtype=dtype))
        self.scalar = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))


class TestCheckpointOptimizerState(unittest.TestCase):
    """Check required slots, failure atomicity, and the actual next AdamW update."""

    def setUp(self) -> None:
        """Create finite source model and AdamW states."""
        self.model = _Model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.003, betas=(.8, .95), eps=1e-8,
                                           weight_decay=.01)
        self.weights = {"matrix": np.full((2, 3), 1.0017, dtype=np.float32), "scalar": np.array(.913, np.float32)}
        self.moments = {f"{prefix}.{name}": np.full(value.shape, fill, np.float32)
                        for name, value in self.weights.items() for prefix, fill in (("adam_m", .2), ("adam_v", .3))}

    def assert_nested_equal(self, actual: Any, expected: Any) -> None:
        """Compare snapshots including scalar values and optimizer tensor state."""
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key, value in expected.items():
                self.assert_nested_equal(actual[key], value)
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_scalar_and_inactive_family_placeholder(self):
        self.moments["muon_m.matrix"] = np.array([0], np.float32)
        self.assertEqual(restore_optimizer_state(self.model, self.optimizer, self.weights, self.moments, 200), 4)
        self.assertEqual(self.optimizer.state[self.model.scalar]["exp_avg"].shape, torch.Size([]))
        for name, parameter in self.model.named_parameters():
            torch.testing.assert_close(parameter, torch.from_numpy(self.weights[name]), rtol=0, atol=0)
            for prefix, field in (("adam_m", "exp_avg"), ("adam_v", "exp_avg_sq")):
                torch.testing.assert_close(self.optimizer.state[parameter][field],
                                           torch.from_numpy(self.moments[f"{prefix}.{name}"]), rtol=0, atol=0)
            self.assertEqual(self.optimizer.state[parameter]["step"].item(), 200)

    def test_invalid_input_does_not_change_initialized_state(self):
        for parameter in self.model.parameters():
            parameter.grad = torch.full_like(parameter, .125)
        self.optimizer.step()
        original_model = copy.deepcopy(self.model.state_dict())
        original_optimizer = copy.deepcopy(self.optimizer.state_dict())
        cases = [
            ("missing_late_slot", {key: value for key, value in self.moments.items() if key != "adam_v.scalar"}, 200),
            ("wrong_shape", {**self.moments, "adam_m.matrix": np.ones((1,), np.float32)}, 200),
            ("wrong_dtype", {**self.moments, "adam_m.matrix": np.ones((2, 3), np.float64)}, 200),
            ("nonfinite", {**self.moments, "adam_v.scalar": np.array(np.nan, np.float32)}, 200),
            ("negative_variance", {**self.moments, "adam_v.scalar": np.array(-1, np.float32)}, 200),
            ("unknown_parameter", {**self.moments, "adam_m.unknown": np.array(0, np.float32)}, 200),
            ("unknown_family", {**self.moments, "sgd_m.scalar": np.array(0, np.float32)}, 200),
            ("inactive_real_state", {**self.moments, "muon_m.matrix": np.zeros((2, 3), np.float32)}, 200),
            ("bool_step", self.moments, True),
            ("negative_step", self.moments, -1),
            ("counter_cannot_advance", self.moments, 2**24),
        ]
        for name, moments, step in cases:
            with self.subTest(name=name), self.assertRaises(ValueError):
                restore_optimizer_state(self.model, self.optimizer, self.weights, moments, step)
            self.assert_nested_equal(self.model.state_dict(), original_model)
            self.assert_nested_equal(self.optimizer.state_dict(), original_optimizer)

    def test_missing_master_rejected_before_writes(self):
        weights = {"matrix": self.weights["matrix"]}
        with self.assertRaises(ValueError):
            restore_optimizer_state(self.model, self.optimizer, weights, self.moments, 200)
        self.assertEqual(len(self.optimizer.state), 0)
        self.assertEqual(torch.count_nonzero(self.model.matrix).item(), 0)

    def test_unsupported_optimizer_and_amsgrad(self):
        for optimizer in (torch.optim.SGD(self.model.parameters(), lr=.01),
                          torch.optim.AdamW(self.model.parameters(), amsgrad=True)):
            with self.subTest(optimizer=type(optimizer).__name__), self.assertRaises(ValueError):
                restore_optimizer_state(self.model, optimizer, self.weights, self.moments, 200)
            self.assertEqual(len(optimizer.state), 0)

    def test_hyper_adamw_uses_group_step(self):
        leaf = HyperAdamW(self.model.parameters(), lr=.003, betas=(.8, .95), eps=1e-8, weight_decay=.01)
        chained = ChainedOptimizer(self.model, {"adamw": leaf})
        self.assertEqual(restore_optimizer_state(self.model, chained, self.weights, self.moments, 200), 4)
        self.assertEqual(leaf.param_groups[0]["step"], 200)
        self.assertNotIn("step", leaf.state[self.model.matrix])
        for parameter in self.model.parameters():
            parameter.grad = torch.full_like(parameter, .125)
        chained.step()
        self.assertEqual(leaf.param_groups[0]["step"], 201)
        for name, parameter in self.model.named_parameters():
            first = self.moments[f"adam_m.{name}"].astype(np.float64) * .8 + .125 * .2
            second = self.moments[f"adam_v.{name}"].astype(np.float64) * .95 + .125**2 * .05
            expected = self.weights[name].astype(np.float64) * (1 - .003 * .01)
            expected -= .003 * first / (1 - .8**201) / (np.sqrt(second / (1 - .95**201)) + 1e-8)
            np.testing.assert_allclose(parameter.detach().numpy(), expected, rtol=3e-7, atol=3e-7)

    def test_fp32_master_and_next_update_follow_independent_adamw(self):
        model = _Model(torch.bfloat16)
        leaf = torch.optim.AdamW(model.parameters(), lr=.003, betas=(.8, .95), eps=1e-8, weight_decay=.01)
        wrapper = Float16OptimizerWithFloat16Params(ChainedOptimizer(model, {"adamw": leaf}), model)
        self.assertEqual(restore_optimizer_state(model, wrapper, self.weights, self.moments, 200), 4)
        main = wrapper.optimizer_param_by_model_param[model.matrix]
        torch.testing.assert_close(main, torch.from_numpy(self.weights["matrix"]), rtol=0, atol=0)
        self.assertFalse(torch.equal(main, model.matrix.float()))
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, .125)
        wrapper.step()
        for name, parameter in model.named_parameters():
            target = wrapper.optimizer_param_by_model_param.get(parameter, parameter)
            first = self.moments[f"adam_m.{name}"].astype(np.float64) * .8 + .125 * .2
            second = self.moments[f"adam_v.{name}"].astype(np.float64) * .95 + .125**2 * .05
            expected = self.weights[name].astype(np.float64) * (1 - .003 * .01)
            expected -= .003 * (first / (1 - .8**201)) / (np.sqrt(second / (1 - .95**201)) + 1e-8)
            np.testing.assert_allclose(target.detach().numpy(), expected, rtol=3e-7, atol=3e-7)
            np.testing.assert_allclose(leaf.state[target]["exp_avg"].numpy(), first, rtol=2e-7, atol=2e-7)
            np.testing.assert_allclose(leaf.state[target]["exp_avg_sq"].numpy(), second, rtol=2e-7, atol=2e-7)
            self.assertEqual(leaf.state[target]["step"].item(), 201)
            torch.testing.assert_close(parameter, target.to(parameter.dtype), rtol=0, atol=0)

    def test_source_that_overflows_model_dtype_is_rejected_before_writes(self):
        model = _Model(torch.float16)
        leaf = torch.optim.AdamW(model.parameters())
        wrapper = Float16OptimizerWithFloat16Params(ChainedOptimizer(model, {"adamw": leaf}), model)
        before = {parameter: parameter.detach().clone() for parameter in leaf.param_groups[0]["params"]}
        weights = {**self.weights, "matrix": np.full((2, 3), 70000, dtype=np.float32)}
        with self.assertRaises(ValueError):
            restore_optimizer_state(model, wrapper, weights, self.moments, 200)
        self.assertEqual(len(leaf.state), 0)
        for parameter, expected in before.items():
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)

    def test_single_device_dtensor_main_and_moment_layout(self):
        with patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=0):
            mesh = DeviceMesh("cpu", [0], mesh_dim_names=("dp",), _init_backend=False)
            model = torch.nn.Module()
            model.weight = torch.nn.Parameter(
                DTensor.from_local(torch.zeros((2, 3), dtype=torch.bfloat16), mesh, (Replicate(),))
            )
            leaf = HyperAdamW(model.parameters())
            wrapper = Float16OptimizerWithFloat16Params(ChainedOptimizer(model, {"adamw": leaf}), model)
            weights = {"weight": self.weights["matrix"]}
            moments = {"adam_m.weight": self.moments["adam_m.matrix"],
                       "adam_v.weight": self.moments["adam_v.matrix"]}
            self.assertEqual(restore_optimizer_state(model, wrapper, weights, moments, 200), 2)
            main = wrapper.optimizer_param_by_model_param[model.weight]
            self.assertIsInstance(main, DTensor)
            np.testing.assert_array_equal(main.to_local().detach().numpy(), weights["weight"])
            for prefix, field in (("adam_m", "exp_avg"), ("adam_v", "exp_avg_sq")):
                moment = leaf.state[main][field]
                self.assertIsInstance(moment, DTensor)
                self.assertEqual(tuple(moment.placements), (Replicate(),))
                np.testing.assert_array_equal(moment.to_local().numpy(), moments[f"{prefix}.weight"])

    def test_multiple_device_mesh_rejected_before_writes(self):
        with patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=0):
            mesh = DeviceMesh("cpu", [0, 1], mesh_dim_names=("dp",), _init_backend=False)
            model = torch.nn.Module()
            model.weight = torch.nn.Parameter(DTensor.from_local(torch.zeros(2), mesh, (Replicate(),)))
            leaf = HyperAdamW(model.parameters())
            with self.assertRaises(ValueError):
                restore_optimizer_state(model, leaf, {"weight": np.ones(2, np.float32)}, {}, 200)
            self.assertEqual(len(leaf.state), 0)
            self.assertEqual(torch.count_nonzero(model.weight.to_local()).item(), 0)

    def test_muon_momentum_and_inactive_adam_placeholders(self):
        model = torch.nn.Linear(3, 2, bias=False)
        leaf = Muon(model.parameters())
        optimizer = ChainedOptimizer(model, {"muon": leaf})
        weights = {"weight": self.weights["matrix"]}
        moments = {"muon_m.weight": np.full((2, 3), .2, np.float32),
                   "adam_m.weight": np.array([0], np.float32), "adam_v.weight": np.array([0], np.float32)}
        self.assertEqual(restore_optimizer_state(model, optimizer, weights, moments, 200), 1)
        self.assertEqual(leaf.param_groups[0]["step"], 200)
        self.assertEqual(set(leaf.state[model.weight]), {"momentum_buffer"})
        np.testing.assert_array_equal(leaf.state[model.weight]["momentum_buffer"].numpy(), moments["muon_m.weight"])
