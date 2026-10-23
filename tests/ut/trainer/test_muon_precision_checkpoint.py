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
"""Exercise Muon precision through real FQN conversion, DCP and trainer restore."""
# pylint: disable=protected-access

import copy
import tempfile
from types import SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.components.checkpoint.dcp_checkpointer import (
    DistributedCheckpointer,
    initialize_optimizer_state,
)
from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.core.distributed_checkpoint import save as dcp_save
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.trainer.callbacks.checkpoint_callback import CheckpointerCallback
from hyper_parallel.trainer.state import TrainerState


def _build_optimizer(flatten: bool, wrapped: bool, dtype: str) -> tuple:
    """Build both checkpoint formats, including real mixed-precision main weights."""
    model = torch.nn.Linear(3, 2, bias=flatten)
    if wrapped:
        model.bfloat16()
    leaves = {"muon": Muon([model.weight], ns_dtype=dtype, ns_variant="legacy")}
    if flatten:
        leaves["adamw"] = AdamW([model.bias])
    optimizer = ChainedOptimizer(model, leaves, flatten=flatten)
    if wrapped:
        optimizer = Float16OptimizerWithFloat16Params(optimizer, model)
    initialize_optimizer_state(optimizer)
    return model, optimizer


def _set_checkpoint_precision(state: dict, flatten: bool, value: Optional[str]) -> None:
    """Remove the new field to represent a legacy checkpoint, or replace it."""
    if flatten:
        if value is None:
            state.pop("param_groups.weight.ns_dtype")
        else:
            state["param_groups.weight.ns_dtype"] = value
    elif value is None:
        state["param_groups"][0].pop("ns_dtype")
    else:
        state["param_groups"][0]["ns_dtype"] = value


def _take_step(optimizer: object, iteration: int) -> None:
    """Apply deterministic CPU gradients directly to the optimizer-owned parameters."""
    chain = optimizer.optimizer if isinstance(optimizer, Float16OptimizerWithFloat16Params) else optimizer
    for group in chain.param_groups:
        for parameter in group["params"]:
            parameter.grad = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape_as(parameter)
            parameter.grad.add_(iteration + 1).mul_(0.1234567)
    if isinstance(optimizer, Float16OptimizerWithFloat16Params):
        optimizer.step_with_ready_grads()
    else:
        optimizer.step()


class TestMuonPrecisionCheckpoint(unittest.TestCase):
    """A checkpoint's saved recipe wins over constructor defaults in every entry point."""

    def test_direct_fqn_load_and_skeleton_do_not_change_saved_or_live_groups(self) -> None:
        """Old flattened state is normalized before Torch unflattens; saved FP32 is preserved."""
        for flatten in (False, True):
            for wrapped in (False, True):
                with self.subTest(flatten=flatten, wrapped=wrapped):
                    _, source = _build_optimizer(flatten, wrapped, "float32")
                    saved = copy.deepcopy(source.state_dict())
                    skeleton = source.state_dict_for_load()
                    _, target = _build_optimizer(flatten, wrapped, "bfloat16")
                    target.load_state_dict(saved)
                    self.assertEqual(target.param_groups[0]["ns_dtype"], "float32")
                    self.assertEqual(source.param_groups[0]["ns_dtype"], "float32")
                    field = "param_groups.weight.ns_dtype"
                    self.assertEqual(skeleton[field] if flatten else skeleton["param_groups"][0]["ns_dtype"],
                                     "bfloat16")
                    self.assertEqual(saved[field] if flatten else saved["param_groups"][0]["ns_dtype"], "float32")
                    _set_checkpoint_precision(saved, flatten, None)
                    target.load_state_dict(saved)
                    self.assertEqual(target.param_groups[0]["ns_dtype"], "bfloat16")
                    if flatten:
                        self.assertNotIn(field, saved)
                        self.assertNotIn("ns_dtype", target.param_groups[1])
                    else:
                        self.assertNotIn("ns_dtype", saved["param_groups"][0])

    @patch("hyper_parallel.core.distributed_checkpoint.api.dist.barrier")
    @patch("hyper_parallel.core.distributed_checkpoint.api.dist.get_world_size", return_value=1)
    @patch("hyper_parallel.core.distributed_checkpoint.api.dist.get_rank", return_value=0)
    def test_trainer_dcp_resume_uses_persisted_precision(self, *_mocks: object) -> None:
        """Both DCP formats restore legacy BF16 and saved FP32, then match a real next step."""
        for flatten in (False, True):
            for wrapped in (False, True):
                for saved_dtype in (None, "bfloat16", "float32"):
                    with self.subTest(flatten=flatten, wrapped=wrapped, saved_dtype=saved_dtype):
                        self._check_resume(flatten, wrapped, saved_dtype)

    def _check_resume(self, flatten: bool, wrapped: bool, saved_dtype: Optional[str]) -> None:
        """Write a small real checkpoint and restore it through the production callback."""
        expected_dtype = saved_dtype or "bfloat16"
        source_model, source = _build_optimizer(flatten, wrapped, expected_dtype)
        _take_step(source, 0)
        saved = copy.deepcopy(source.state_dict())
        _set_checkpoint_precision(saved, flatten, saved_dtype)
        target_model, target = _build_optimizer(flatten, wrapped,
                                                "bfloat16" if expected_dtype == "float32" else "float32")
        callback = object.__new__(CheckpointerCallback)
        callback._restore_optimizer = True
        callback._restore_train_state = False
        callback._is_peft = False
        callback.checkpointer = DistributedCheckpointer()
        callback.trainer = SimpleNamespace(
            model=target_model, optimizer=target, config=SimpleNamespace(
                model_init_dtype="bfloat16" if wrapped else "float32",
            ), state=TrainerState(), train_dataloader=[None] * 4, train_steps=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            dcp_save({"model": source_model.state_dict(), "optimizer": saved}, checkpoint_id=directory)
            with patch.object(callback, "_resolve_restore_path", return_value=directory), \
                    patch("hyper_parallel.trainer.callbacks.checkpoint_callback.empty_cache"):
                callback._load_checkpoint()
        self.assertEqual(target.param_groups[0]["ns_dtype"], expected_dtype)
        _take_step(source, 1)
        _take_step(target, 1)
        for actual, expected in zip(target_model.parameters(), source_model.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual_group, expected_group in zip(target.param_groups, source.param_groups):
            for actual, expected in zip(actual_group["params"], expected_group["params"]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_saved_precision_fails(self) -> None:
        """Malformed fields fail at the real FQN loader without replacing live groups."""
        for flatten in (False, True):
            with self.subTest(flatten=flatten):
                _, optimizer = _build_optimizer(flatten, False, "float32")
                invalid = copy.deepcopy(optimizer.state_dict())
                _set_checkpoint_precision(invalid, flatten, "invalid")
                with self.assertRaisesRegex(ValueError, "ns_dtype"):
                    optimizer.load_state_dict(invalid)
                self.assertEqual(optimizer.param_groups[0]["ns_dtype"], "float32")

    def test_invalid_precision_precedes_all_leaf_restores(self) -> None:
        """Production AdamW-before-Muon ordering must not partially load on invalid precision."""
        model = torch.nn.Linear(3, 2)
        adam = AdamW([model.bias], lr=0.01)
        muon = Muon([model.weight], ns_dtype="float32")
        optimizer = ChainedOptimizer(model, {"adamw": adam, "muon": muon}, flatten=True)
        initialize_optimizer_state(optimizer)
        _take_step(optimizer, 0)
        before = copy.deepcopy(adam.state_dict())
        invalid = copy.deepcopy(optimizer.state_dict())
        invalid["param_groups.bias.lr"] = 0.5
        invalid["state.bias.exp_avg"].fill_(12.5)
        invalid["param_groups.weight.ns_dtype"] = "invalid"
        with self.assertRaisesRegex(ValueError, "ns_dtype"):
            optimizer.load_state_dict(invalid)
        self.assertEqual(adam.param_groups[0]["lr"], before["param_groups"][0]["lr"])
        for key, value in before["state"][0].items():
            torch.testing.assert_close(adam.state[model.bias][key], value, rtol=0, atol=0)
        self.assertEqual(muon.param_groups[0]["ns_dtype"], "float32")

    def test_empty_chain_keeps_noop_load_contract(self) -> None:
        """An empty optimizer family has no metadata to normalize."""
        for flatten in (False, True):
            with self.subTest(flatten=flatten):
                optimizer = ChainedOptimizer(torch.nn.Linear(2, 2), {}, flatten=flatten)
                self.assertEqual(optimizer.state_dict_for_load(), {})
                optimizer.load_state_dict({})
                self.assertEqual(optimizer.state_dict(), {})

    @patch("hyper_parallel.core.distributed_checkpoint.api.dist.barrier")
    @patch("hyper_parallel.core.distributed_checkpoint.api.dist.get_world_size", return_value=1)
    @patch("hyper_parallel.core.distributed_checkpoint.api.dist.get_rank", return_value=0)
    def test_weights_only_restore_keeps_constructor_precision(self, *_mocks: object) -> None:
        """A model-only restore skips load skeleton normalization and keeps the fresh recipe."""
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                model, optimizer = _build_optimizer(True, wrapped, "float32")
                callback = object.__new__(CheckpointerCallback)
                callback._restore_optimizer = False
                callback._restore_train_state = False
                callback._is_peft = False
                callback.checkpointer = DistributedCheckpointer()
                callback.trainer = SimpleNamespace(
                    model=model, optimizer=optimizer, config=SimpleNamespace(
                        model_init_dtype="bfloat16" if wrapped else "float32",
                    ), state=TrainerState(), train_dataloader=[None] * 4, train_steps=4,
                )
                with tempfile.TemporaryDirectory() as directory:
                    dcp_save({"model": model.state_dict()}, checkpoint_id=directory)
                    with patch.object(callback, "_resolve_restore_path", return_value=directory), \
                            patch("hyper_parallel.trainer.callbacks.checkpoint_callback.empty_cache"), \
                            patch.object(optimizer, "state_dict_for_load") as build_skeleton:
                        callback._load_checkpoint()
                        build_skeleton.assert_not_called()
                self.assertEqual(optimizer.param_groups[0]["ns_dtype"], "float32")


if __name__ == "__main__":
    unittest.main()
