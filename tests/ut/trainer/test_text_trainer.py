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
"""Unit tests for the text Trainer step lifecycle."""

import importlib
import sys
import unittest
from types import ModuleType, SimpleNamespace
from typing import Optional
from unittest.mock import Mock, patch

import torch

from hyper_parallel.trainer.runtime.metrics import mean_global_loss
from tests.common.mark_utils import arg_mark


def _load_text_trainer() -> type:
    """Load TextTrainer without requiring the unrelated stateful dataloader dependency."""
    torchdata_module = ModuleType("torchdata")
    stateful_dataloader_module = ModuleType("torchdata.stateful_dataloader")
    stateful_dataloader_module.StatefulDataLoader = torch.utils.data.DataLoader
    with patch.dict(
            sys.modules,
            {
                "torchdata": torchdata_module,
                "torchdata.stateful_dataloader": stateful_dataloader_module,
            },
    ):
        trainer_module = importlib.import_module("hyper_parallel.trainer.text_trainer")
    return trainer_module.TextTrainer


TextTrainer = _load_text_trainer()


class TestTextTrainer(unittest.TestCase):
    """Verify text training streams prepared batches into callbacks."""

    @staticmethod
    def _run_optimizer_step(
            batches: list[tuple[dict, dict]],
            num_micro_batches: Optional[int] = None,
    ) -> tuple[dict, torch.Tensor]:
        """Exercise real text-step aggregation while isolating communication and devices."""
        weight = torch.tensor(1.0, requires_grad=True)
        trainer = TextTrainer.__new__(TextTrainer)

        def prepare_optimizer_step() -> torch.Tensor:
            """Read the accumulated gradient only after backward has run."""
            return weight.grad.abs()

        base = SimpleNamespace(
            num_micro_batches=len(batches) if num_micro_batches is None else num_micro_batches,
            state=SimpleNamespace(global_step=0),
            get_batch=next,
            on_step_begin=Mock(), on_micro_step_begin=Mock(), on_step_end=Mock(),
            model_integration=Mock(), model_reshard=Mock(), configure_fsdp_gradient_sync=Mock(),
            begin_fsdp_runtime_diagnostics=Mock(), step_optimizers_and_schedulers=Mock(),
            _end_model_integration_step=Mock(), prepare_optimizer_step=prepare_optimizer_step,
        )
        trainer.base = base
        mesh = SimpleNamespace(dp_cp_mesh=None, sequence_parallel=False, dp_size=1, cp_size=1)

        def forward_backward(model_inputs: dict, loss_inputs: dict) -> tuple[torch.Tensor, dict]:
            """Differentiate a token mean through the production global-loss reducer."""
            valid = loss_inputs["shift_labels"].ge(0)
            valid = valid & loss_inputs.get("loss_mask", torch.ones_like(valid)).bool()
            squared = (weight * model_inputs["input_ids"]).square()
            value = (squared * valid).sum() / valid.sum().clamp_min(1)
            losses = mean_global_loss(value, base.current_token_counts, base.step_token_counts, mesh)
            loss = sum(losses.values())
            loss.backward()
            return loss, losses

        base.forward_backward_step = forward_backward
        with patch("hyper_parallel.trainer.runtime.metrics.all_reduce", side_effect=lambda value, **_: value), \
             patch("hyper_parallel.trainer.text_trainer.synchronize"), \
             patch("hyper_parallel.trainer.text_trainer.memory_profiler"):
            result = trainer.train_step(iter(batches))
        return result, weight.grad

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_accumulation_matches_whole_step_token_mean(self) -> None:
        """Compare uneven, empty and masked micro-batches with a single whole batch.

        Feature: Token-weighted gradient accumulation.
        Description: Execute the real training step and reducer on differentiable token losses.
        Expectation: Changing the micro-batch split preserves the loss and parameter gradient.
        """
        values = torch.tensor([[1., 11., 13.], [3., 3., 3.], [5., 7., 9.]])
        for labels, mask in (
                ([[1, -100, -100], [1, 1, 1], [1, -100, -100]], None),
                ([[-100, -100, -100], [1, 1, 1], [1, -100, -100]], None),
                ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 0, 0], [1, 1, 1], [1, 0, 0]]),
        ):
            with self.subTest(labels=labels, mask=mask):
                supervision = {"shift_labels": torch.tensor(labels)}
                if mask is not None:
                    supervision["loss_mask"] = torch.tensor(mask)
                expected, expected_grad = self._run_optimizer_step([({"input_ids": values}, supervision)])
                batches = [
                    ({"input_ids": values[index:index + 1]},
                     {name: tensor[index:index + 1] for name, tensor in supervision.items()})
                    for index in range(len(values))
                ]
                actual, actual_grad = self._run_optimizer_step(batches)
                self.assertAlmostEqual(actual["loss"], expected["loss"], places=6,
                                       msg=f"accumulated={actual['loss']}, whole_step={expected['loss']}")
                torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-6,
                                           msg=f"accumulated={actual_grad}, whole_step={expected_grad}")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_exhausted_step_preserves_stop_iteration(self) -> None:
        """Keep the epoch loop's exhaustion signal for an incomplete accumulation step.

        Feature: Text training iterator exhaustion.
        Description: Exhaust a custom batch callable before the first or second micro-batch.
        Expectation: StopIteration reaches the epoch loop before any forward or optimizer update.
        """
        batch = ({"input_ids": torch.ones(1, 2)}, {"shift_labels": torch.ones(1, 2)})
        for batches in ([], [batch]):
            with self.subTest(available=len(batches)), self.assertRaises(StopIteration):
                self._run_optimizer_step(batches, num_micro_batches=2)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_forward_backward_dispatches_metric_hook_before_forward(self) -> None:
        """Send detached metric inputs before executing the model forward.

        Feature: Text Trainer micro-step callbacks.
        Description: Fetch one prepared batch and execute one forward-backward step.
        Expectation: Metric callback receives token metadata immediately before model execution.
        """
        events = []
        labels = torch.tensor([[1, 2, 3, 4], [5, 6, 7, -100]])
        model_inputs = {"input_ids": torch.ones(2, 4)}
        loss_inputs = {"labels": labels, "shift_labels": labels}

        def on_micro_step_begin(metric_inputs: dict[str, object]) -> None:
            """Record and validate the metric callback payload."""
            events.append("hook")
            token_count = int(metric_inputs["token_count"])
            self.assertEqual(token_count, 7, f"expected seven valid tokens, got={token_count}")

        def forward_backward_step(
                actual_model_inputs: dict[str, torch.Tensor],
                actual_loss_inputs: dict[str, torch.Tensor],
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            """Record model execution and validate forwarded batch identity."""
            events.append("forward")
            self.assertIs(
                actual_model_inputs,
                model_inputs,
                f"model input identity changed, got={actual_model_inputs!r}",
            )
            self.assertIs(
                actual_loss_inputs,
                loss_inputs,
                f"loss input identity changed, got={actual_loss_inputs!r}",
            )
            return torch.tensor(1.0), {"foundation_loss": torch.tensor(1.0)}

        base = SimpleNamespace(
            get_batch=Mock(return_value=(model_inputs, loss_inputs)),
            on_micro_step_begin=Mock(side_effect=on_micro_step_begin),
            forward_backward_step=Mock(side_effect=forward_backward_step),
        )
        trainer = TextTrainer.__new__(TextTrainer)
        trainer.base = base

        loss, loss_dict = trainer.forward_backward_step(model_inputs, loss_inputs)

        self.assertEqual(events, ["hook", "forward"], f"expected streaming hook order, got={events}")
        self.assertEqual(float(loss), 1.0, f"expected loss 1.0, got={float(loss)}")
        self.assertEqual(
            float(loss_dict["foundation_loss"]),
            1.0,
            f"expected foundation loss 1.0, got={float(loss_dict['foundation_loss'])}",
        )


if __name__ == "__main__":
    unittest.main()
