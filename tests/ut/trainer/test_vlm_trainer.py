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
"""VLM micro-batch streaming and incomplete-step cleanup."""

import importlib
import sys
import unittest
import weakref
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch


def _load_vlm_trainer() -> type:
    """Load the Trainer without the optional stateful DataLoader package."""
    module = ModuleType("torchdata.stateful_dataloader")
    module.StatefulDataLoader = torch.utils.data.DataLoader
    with patch.dict(sys.modules, {"torchdata": ModuleType("torchdata"), "torchdata.stateful_dataloader": module}):
        return importlib.import_module("hyper_parallel.trainer.vlm_trainer").VLMTrainer


VLMTrainer = _load_vlm_trainer()


class TestVLMTrainer(unittest.TestCase):
    """Verify inputs are released before fetching another accumulation batch."""

    def _trainer(self, events: list) -> object:
        references = []

        def get_batch(iterator: object) -> tuple[dict, dict]:
            """Check previous input release before materializing the next batch."""
            index = next(iterator)
            self.assertTrue(all(reference() is None for reference in references))
            tensor = torch.tensor([[index + 1, index + 2]])
            references.append(weakref.ref(tensor))
            events.append(("read", index))
            return {"input_ids": tensor}, {"labels": torch.ones_like(tensor), "shift_labels": torch.ones_like(tensor)}

        def forward(model_inputs: dict, loss_inputs: dict) -> tuple[torch.Tensor, dict]:
            """Record a completed micro-step without retaining input references."""
            del loss_inputs
            index = int(model_inputs["input_ids"][0, 0]) - 1
            events.append(("backward", index))
            return torch.tensor(1.0), {"lm": torch.tensor(1.0)}

        def metric(batch: dict) -> None:
            """Validate streaming token accounting before backward."""
            events.append(("metric", int(batch["input_ids"][0, 0]) - 1))
            self.assertEqual(int(batch["token_count"]), 2)

        trainer = VLMTrainer.__new__(VLMTrainer)
        trainer.base = SimpleNamespace(
            get_batch=get_batch, forward_backward_step=forward, on_micro_step_begin=metric,
            num_micro_batches=3, on_step_begin=Mock(), on_step_end=Mock(),
            model_integration=SimpleNamespace(begin_step=Mock()),
            model_reshard=Mock(), configure_fsdp_gradient_sync=Mock(), begin_fsdp_runtime_diagnostics=Mock(),
            prepare_optimizer_step=Mock(return_value=1.0), step_optimizers_and_schedulers=Mock(),
            state=SimpleNamespace(global_step=0), _end_model_integration_step=Mock(), optimizer=Mock(),
            config=SimpleNamespace(dataloader=SimpleNamespace(drop_last=True)), train_steps=1, train_iters=2,
        )
        return trainer

    def test_training_releases_inputs_between_micro_batches(self) -> None:
        """Interleave reads, metric callbacks and backward without retaining the full accumulation step."""
        events = []
        trainer = self._trainer(events)
        with patch("hyper_parallel.trainer.vlm_trainer.synchronize"):
            result = trainer.train_step(iter(range(3)))
        self.assertEqual(events, [(stage, index) for index in range(3) for stage in ("read", "metric", "backward")])
        self.assertEqual(result, {"loss": 3.0, "grad_norm": 1.0})
        self.assertEqual(trainer.base.state.global_step, 1)
        trainer.base.on_step_begin.assert_called_once_with(micro_batches=None)
        trainer.base.step_optimizers_and_schedulers.assert_called_once()

    def test_incomplete_epoch_discards_partial_gradients(self) -> None:
        """An exhausted accumulation step must clear gradients and never update parameters."""
        events = []
        trainer = self._trainer(events)
        trainer.base.train_dataloader = iter(range(2))
        with patch("hyper_parallel.trainer.vlm_trainer.synchronize"):
            self.assertTrue(trainer._train_epoch(epoch=0, collective_source=True))
        self.assertEqual([event for event in events if event[0] == "backward"], [("backward", 0), ("backward", 1)])
        trainer.base.optimizer.zero_grad.assert_called_once()
        trainer.base.step_optimizers_and_schedulers.assert_not_called()
        self.assertEqual(trainer.base.state.global_step, 0)
