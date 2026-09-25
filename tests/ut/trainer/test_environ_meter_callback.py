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
"""Validate that the metrics producer collects declared step observations."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import hyper_parallel.trainer.callbacks.environ_meter_callback as environ_meter_module
from hyper_parallel.trainer.callbacks.environ_meter_callback import EnvironMeterCallback
from hyper_parallel.trainer.state import TrainerState
from tests.common.mark_utils import arg_mark


class TestEnvironMeterMetrics(unittest.TestCase):
    """The declared source is consumed once per step, on every rank."""

    @staticmethod
    def _trainer() -> SimpleNamespace:
        """Build the callback's minimal Trainer dependency surface."""
        return SimpleNamespace(
            mesh=None,
            lr_scheduler=None,
            optimizer=SimpleNamespace(param_groups=[{"lr": 0.001}]),
        )

    @staticmethod
    def _publish(trainer: SimpleNamespace) -> None:
        """Run one completed step through the callback."""
        meter = EnvironMeterCallback(trainer)
        state = TrainerState(global_step=1)
        meter.on_step_begin(state)
        meter.on_step_end(state, loss=3.0, loss_dict={"foundation_loss": 3.0}, grad_norm=0.5)

    @patch.object(environ_meter_module, "get_step_metrics_provider")
    @patch.object(environ_meter_module, "get_device_type", return_value="cpu")
    @patch.object(environ_meter_module, "get_world_size_safe", return_value=1)
    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_declared_metrics_reach_both_dictionaries(self, mock_world_size, mock_device_type, mock_resolve):
        """Feature: Declared step metrics.

        Description: Publish one step with a family-declared metrics provider.
        Expectation: The provider is consumed once and both dictionaries carry the observations.
        """
        del mock_world_size, mock_device_type
        provider = Mock(return_value={"training/mtp_loss": 0.25, "optimizer/qkclip_maxlogits": 1.5})
        mock_resolve.return_value = provider
        trainer = self._trainer()

        self._publish(trainer)

        provider.assert_called_once()
        for metrics in (trainer.step_train_metrics, trainer.step_env_metrics):
            self.assertEqual(metrics["training/total_loss"], 3.0)
            self.assertEqual(metrics["training/mtp_loss"], 0.25)
            self.assertEqual(metrics["optimizer/qkclip_maxlogits"], 1.5)

    @patch.object(environ_meter_module, "get_step_metrics_provider", return_value=None)
    @patch.object(environ_meter_module, "get_device_type", return_value="cpu")
    @patch.object(environ_meter_module, "get_world_size_safe", return_value=1)
    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_families_without_a_source_publish_shared_metrics_only(
            self, mock_world_size, mock_device_type, mock_resolve,
    ):
        """Feature: Optional source.

        Description: Publish one step for a family that declares no metrics source.
        Expectation: The shared training metrics are published unchanged.
        """
        del mock_world_size, mock_device_type, mock_resolve
        trainer = self._trainer()

        self._publish(trainer)

        self.assertEqual(
            set(trainer.step_train_metrics),
            {"training/total_loss", "training/grad_norm", "training/lr"},
        )


if __name__ == "__main__":
    unittest.main()
