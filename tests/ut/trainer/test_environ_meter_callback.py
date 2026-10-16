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
"""Unit tests for Trainer environment and throughput metrics."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import hyper_parallel.trainer.callbacks.environ_meter_callback as environ_meter_module
from hyper_parallel.trainer.callbacks.environ_meter_callback import EnvironMeterCallback
from hyper_parallel.trainer.state import TrainerState
from tests.common.mark_utils import arg_mark


class _NoItemScalar:
    """Represent a device scalar whose host conversion is forbidden."""

    def detach(self) -> "_NoItemScalar":
        """Return a detached scalar view."""
        return self

    def clone(self) -> "_NoItemScalar":
        """Return an independently owned scalar."""
        return _NoItemScalar()

    def item(self) -> float:
        """Fail if metric collection synchronizes the scalar to the host."""
        raise AssertionError("micro-step metric collection must not call item()")


class TestEnvironMeterCallback(unittest.TestCase):
    """Verify token throughput uses streamed, prepared text batches."""

    @staticmethod
    def _build_trainer(mesh: object = None) -> SimpleNamespace:
        """Build the callback's minimal Trainer dependency surface."""
        return SimpleNamespace(
            mesh=mesh,
            lr_scheduler=None,
            optimizer=SimpleNamespace(param_groups=[{"lr": 0.001}]),
        )

    @patch.object(environ_meter_module, "get_device_type", return_value="cpu")
    @patch.object(environ_meter_module, "get_world_size_safe", return_value=1)
    @patch.object(environ_meter_module.time, "perf_counter", side_effect=(10.0, 12.0))
    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_micro_step_publishes_nonzero_token_throughput(
            self,
            mock_time: Mock,
            mock_world_size: Mock,
            mock_device_type: Mock,
    ) -> None:
        """Publish the accumulated micro-step token count as throughput.

        Feature: Trainer throughput metrics.
        Description: Stream a prepared text micro-batch into the environment meter.
        Expectation: Step tokens and tokens per second are positive and scalar state is released.
        """
        del mock_time, mock_world_size, mock_device_type
        trainer = self._build_trainer()
        callback = EnvironMeterCallback(trainer)
        state = TrainerState(global_step=1, epoch=0)

        callback.on_step_begin(state)
        callback.on_micro_step_begin(
            state,
            {
                "input_ids": torch.ones(2, 4),
                "token_count": torch.tensor(6),
            },
        )
        callback.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.5)

        metrics = trainer.step_env_metrics
        self.assertEqual(
            metrics["data/step_tokens"],
            6.0,
            f"expected six accumulated tokens, got={metrics['data/step_tokens']}",
        )
        self.assertEqual(
            metrics["performance/tokens_per_second"],
            3.0,
            f"expected three tokens per second, got={metrics['performance/tokens_per_second']}",
        )
        step_tokens = vars(callback)["_local_step_tokens"]
        self.assertIsNone(
            step_tokens,
            f"step token scalar must be released after publishing, got={step_tokens!r}",
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_micro_step_avoids_host_conversion_and_step_begin_resets(self) -> None:
        """Keep device token scalars asynchronous and clear them at the next step.

        Feature: Environment meter memory lifecycle.
        Description: Accumulate a detached scalar that rejects host conversion and begin a new step.
        Expectation: Accumulation avoids item calls and the next step clears prior scalar state.
        """
        callback = EnvironMeterCallback(self._build_trainer())
        state = TrainerState(global_step=0, epoch=0)

        callback.on_step_begin(state)
        callback.on_micro_step_begin(
            state,
            {"token_count": _NoItemScalar()},
        )

        step_tokens = vars(callback)["_local_step_tokens"]
        self.assertIsInstance(
            step_tokens,
            _NoItemScalar,
            f"expected detached scalar state, got={step_tokens!r}",
        )
        callback.on_step_begin(state)
        step_tokens = vars(callback)["_local_step_tokens"]
        self.assertIsNone(
            step_tokens,
            f"new step must release prior scalar state, got={step_tokens!r}",
        )

    @patch.object(environ_meter_module, "get_device_type", return_value="cpu")
    @patch.object(environ_meter_module, "get_world_size_safe", return_value=4)
    @patch.object(environ_meter_module, "all_reduce")
    @patch.object(environ_meter_module.time, "perf_counter", side_effect=(10.0, 12.0))
    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_context_parallel_reduction_does_not_duplicate_samples(
            self,
            mock_time: Mock,
            mock_all_reduce: Mock,
            mock_world_size: Mock,
            mock_device_type: Mock,
    ) -> None:
        """Sum CP-sharded tokens while counting each logical sample once.

        Feature: Context-parallel throughput accounting.
        Description: Reduce local token and sample counts over a mocked DP+CP group.
        Expectation: Tokens sum across ranks while CP-replicated samples are counted once.
        """
        del mock_time, mock_world_size, mock_device_type
        mock_all_reduce.side_effect = lambda value, op, group: float(value) * 4 if op == "sum" else float(value)
        mesh = SimpleNamespace(dp_cp_mesh=None, cp_size=2)
        trainer = self._build_trainer(mesh=mesh)
        callback = EnvironMeterCallback(trainer)
        state = TrainerState(global_step=1, epoch=0)

        callback.on_step_begin(state)
        callback.on_micro_step_begin(
            state,
            {
                "input_ids": torch.ones(2, 4),
                "token_count": torch.tensor(6),
            },
        )
        callback.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.5)

        metrics = trainer.step_env_metrics
        self.assertEqual(
            metrics["data/step_tokens"],
            24.0,
            f"expected tokens summed across DP+CP, got={metrics['data/step_tokens']}",
        )
        self.assertEqual(
            metrics["data/step_samples"],
            4.0,
            f"expected CP replicas removed from samples, got={metrics['data/step_samples']}",
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_tensor_parallel_resource_extrema_cover_all_ranks(self) -> None:
        """Include the other TP lane's resources without duplicating its data.

        Feature: Global resource metrics with tensor and context parallelism.
        Description: Simulate DP2 x CP2 x TP2 with the slowest and largest rank outside rank0's DP+CP group.
        Expectation: Resource maxima cover eight ranks; data counts and losses still use DP+CP.
        """
        gibibyte = 1024 ** 3
        dp_cp_group = object()
        dp_cp_ranks = (0, 2, 4, 6)
        resource_values = {
            2.0: (2.0, 8.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0),
            gibibyte: tuple(value * gibibyte for value in (1, 13, 2, 3, 4, 5, 6, 7)),
            2 * gibibyte: tuple(value * gibibyte for value in (2, 17, 3, 4, 5, 6, 7, 8)),
        }

        def simulate_collective(value: object, op: str, group: object) -> float:
            """Return the selected domain's resource maximum or data reduction."""
            ranks = range(8) if group is None else dp_cp_ranks
            self.assertIn(group, (None, dp_cp_group))
            if op == "max":
                return max(resource_values[float(value)][rank] for rank in ranks)
            self.assertIs(group, dp_cp_group,
                          f"data/loss reduction must exclude TP replicas: got={group!r}, expected={dp_cp_group!r}")
            return float(value) * len(ranks) if op == "sum" else float(value)

        mesh = SimpleNamespace(dp_cp_mesh=SimpleNamespace(get_group=lambda: dp_cp_group), cp_size=2)
        trainer = self._build_trainer(mesh)
        callback = EnvironMeterCallback(trainer)
        state = TrainerState(global_step=1, epoch=0)
        device = SimpleNamespace(max_memory_allocated=lambda: gibibyte, max_memory_reserved=lambda: 2 * gibibyte)
        with (
            patch.object(environ_meter_module, "get_world_size_safe", return_value=8),
            patch.object(environ_meter_module, "get_device_type", return_value="npu"),
            patch.object(environ_meter_module, "get_torch_device", return_value=device),
            patch.object(environ_meter_module, "all_reduce", side_effect=simulate_collective),
            patch.object(environ_meter_module.time, "perf_counter", side_effect=(10.0, 12.0)),
        ):
            callback.on_step_begin(state)
            callback.on_micro_step_begin(state, {"input_ids": torch.ones(2, 4), "token_count": torch.tensor(6)})
            callback.on_step_end(state, loss=1.0, loss_dict={"foundation_loss/lm": 1.0}, grad_norm=0.5)

        expected = {
            "performance/step_time": 9.0,
            "performance/tokens_per_second": 24.0 / 9.0,
            "memory/device_max_allocated_gb": 13.0,
            "memory/device_max_reserved_gb": 17.0,
            "data/step_tokens": 24.0,
            "data/step_samples": 4.0,
            "training/foundation_loss/lm": 1.0,
        }
        for name, value in expected.items():
            with self.subTest(metric=name):
                self.assertEqual(trainer.step_env_metrics[name], value, f"wrong reduction domain for {name}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_single_process_resource_metrics_need_no_collectives(self) -> None:
        """Keep local resource reporting usable without a distributed process group.

        Feature: Single-process environment metrics.
        Description: Publish simulated accelerator peaks and timing with world size one.
        Expectation: Local values are preserved and no collective is called.
        """
        trainer = self._build_trainer()
        callback = EnvironMeterCallback(trainer)
        state = TrainerState(global_step=1, epoch=0)
        device = SimpleNamespace(max_memory_allocated=lambda: 3 * 1024 ** 3,
                                 max_memory_reserved=lambda: 4 * 1024 ** 3)
        with (
            patch.object(environ_meter_module, "get_world_size_safe", return_value=1),
            patch.object(environ_meter_module, "get_device_type", return_value="npu"),
            patch.object(environ_meter_module, "get_torch_device", return_value=device),
            patch.object(environ_meter_module, "all_reduce") as collective,
            patch.object(environ_meter_module.time, "perf_counter", side_effect=(10.0, 12.0)),
        ):
            callback.on_step_begin(state)
            callback.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.5)
        collective.assert_not_called()
        self.assertEqual(trainer.step_env_metrics["performance/step_time"], 2.0)
        self.assertEqual(trainer.step_env_metrics["memory/device_max_allocated_gb"], 3.0)
        self.assertEqual(trainer.step_env_metrics["memory/device_max_reserved_gb"], 4.0)


if __name__ == "__main__":
    unittest.main()
