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
"""Validate observational metrics independently of backward loss and log cadence."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from hyper_parallel.trainer.callbacks.logging_callback import LoggingCallback
from hyper_parallel.trainer.state import TrainerState
from tests.common.mark_utils import arg_mark


class TestLoggingMetrics(unittest.TestCase):
    """Presentation respects rank, cadence, and repeated dispatch."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_prints_only_on_eligible_rank_and_cadence(self):
        """Feature: Scalar logging.

        Description: Present one step's published metrics under different ranks and cadences.
        Expectation: Only rank zero on a due step prints, and never twice for one step.
        """
        for rank, cadence in [(0, 2), (1, 1), (0, 0)]:
            trainer = SimpleNamespace(mesh=None, global_rank=rank,
                                      config=SimpleNamespace(training=SimpleNamespace(logging_steps=cadence)),
                                      step_env_metrics={"training/loss": 3.0})
            callback = LoggingCallback(trainer)
            callback._write = Mock()
            callback.on_step_end(TrainerState(global_step=1))
            callback.on_step_end(TrainerState(global_step=1))
            callback._write.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_rank_zero_prints_published_metrics(self):
        """Feature: Terminal metrics.

        Description: Present the published metrics on one eligible step.
        Expectation: The output includes every published field.
        """
        trainer = SimpleNamespace(mesh=None, global_rank=0,
                                  config=SimpleNamespace(training=SimpleNamespace(logging_steps=1)),
                                  step_env_metrics={"training/loss": 2.0, "training/mtp_loss": 0.125})
        callback = LoggingCallback(trainer)
        callback._write = Mock()
        callback.on_step_end(TrainerState(global_step=1))
        message = callback._write.call_args.args[0]
        self.assertIn("training/loss=2", message)
        self.assertIn("training/mtp_loss=0.125", message)
