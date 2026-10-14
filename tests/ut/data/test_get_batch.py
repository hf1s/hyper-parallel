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
"""Tests for bounded preparation of text gradient-accumulation steps."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.data.batching.get_batch import TextParallelBatch
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token
from tests.common.mark_utils import arg_mark


class TestTextStepPreparation(unittest.TestCase):
    """Step-wide token counts must not retain all dense attention masks."""

    @staticmethod
    def _runtime(labels_are_shifted: bool = True, eod_mask_loss: bool = False) -> TextParallelBatch:
        return TextParallelBatch(
            SimpleNamespace(device_mesh=None), torch.device("cpu"), SimpleNamespace(eod=0),
            {"labels_are_shifted": labels_are_shifted}, False, source_type="indexed",
            attention_mode="dense", reset_position_ids=True, eod_mask_loss=eod_mask_loss,
        )

    @staticmethod
    def _batches() -> list[dict[str, torch.Tensor]]:
        return [
            {"tokens": torch.tensor([[1, 2, 0, 4]]), "labels": torch.tensor([[8, -100, 6, 5]])},
            {"tokens": torch.tensor([[5, 0, 7, 8]]), "labels": torch.tensor([[-100, 3, 2, 4]])},
        ]

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_prepare_step_preserves_supervision_and_defers_attention(self) -> None:
        """Prepare all token counts without constructing all forward-only masks.

        Feature: Bounded optimizer-step batch preparation.
        Description: Compare prepared batches with ordinary calls for shifted labels and EOD masks.
        Expectation: Inputs and counts agree, with only one runtime-input build per consumed micro-step.
        """
        for shifted, eod_mask, expected_count in ((True, False, 6), (True, True, 4),
                                                   (False, False, 5), (False, True, 3)):
            with self.subTest(shifted=shifted, eod_mask=eod_mask):
                runtime = self._runtime(shifted, eod_mask)
                source = iter(self._batches())
                expected = [runtime(source), runtime(source)]
                with patch.object(runtime, "_build_runtime_inputs", wraps=runtime._build_runtime_inputs) as build:
                    supervision, batches = runtime.prepare_step(iter(self._batches()), 2)
                    self.assertEqual(build.call_count, 0, f"expected no runtime builds, got={build.call_count}")
                    count = int(count_loss_token(supervision)["foundation_tokens"])
                    self.assertEqual(count, expected_count, f"expected={expected_count}, got={count}")
                    for index, (expected_model, expected_loss) in enumerate(expected):
                        actual_model, actual_loss = next(batches)
                        torch.testing.assert_close(actual_model, expected_model, rtol=0, atol=0)
                        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
                        torch.testing.assert_close(supervision[index], expected_loss, rtol=0, atol=0)
                        self.assertEqual(build.call_count, index + 1,
                                         f"expected={index + 1} runtime builds, got={build.call_count}")
                    with self.assertRaises(StopIteration):
                        next(batches)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_incomplete_step_preserves_stop_iteration(self) -> None:
        """Preserve the iterator protocol before constructing a partial step's masks.

        Feature: Incomplete optimizer-step handling.
        Description: Provide fewer source batches than requested by gradient accumulation.
        Expectation: Preparation raises StopIteration without creating forward-only inputs.
        """
        for available in (0, 1):
            with self.subTest(available=available):
                runtime = self._runtime()
                with patch.object(runtime, "_build_runtime_inputs") as build:
                    with self.assertRaises(StopIteration):
                        runtime.prepare_step(iter(self._batches()[:available]), 2)
                    self.assertEqual(build.call_count, 0, f"expected no runtime builds, got={build.call_count}")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_prepare_step_rejects_invalid_micro_steps(self) -> None:
        """Reject invalid accumulation lengths at the public preparation boundary.

        Feature: Text step preparation validation.
        Description: Supply non-positive, boolean and fractional micro-step counts.
        Expectation: Preparation raises a descriptive ValueError before consuming source data.
        """
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "positive integer"):
                self._runtime().prepare_step(iter(()), count)


if __name__ == "__main__":
    unittest.main()
