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
"""Unit tests for whole-step token accounting in gradient accumulation."""
import unittest

import torch

from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token


def _micro_batch(tokens: int) -> dict:
    """Build one micro-batch whose supervision has exactly ``tokens`` targets."""
    labels = torch.full((1, tokens + 1), 5, dtype=torch.long)
    labels[0, 0] = -100  # the causal shift drops the first position
    return {"shift_labels": labels}


class TestWholeStepTokenCount(unittest.TestCase):
    """The optimizer step must divide by its own total, not an extrapolation."""

    def test_whole_step_count_equals_the_sum_over_micro_batches(self):
        """Counting the prepared step must equal the sum of its parts."""
        counts = [812, 640, 733, 401, 398, 512, 690, 486]
        micro_batches = [_micro_batch(n) for n in counts]

        counted = count_loss_token([batch for batch in micro_batches])

        self.assertEqual(counted["foundation_tokens"], sum(counts))

    def test_first_micro_batch_extrapolation_is_wrong_when_lengths_differ(self):
        """A per-micro-batch count times the micro-batch count is not the total.

        This is the defect the patch removes: with variable-length sequences the
        extrapolation over-counts (or under-counts) the optimizer-step total, and
        that total is the denominator of the backward loss scale.
        """
        counts = [812, 640, 733, 401, 398, 512, 690, 486]
        micro_batches = [_micro_batch(n) for n in counts]

        first_only = count_loss_token(micro_batches[0])["foundation_tokens"]
        extrapolated = first_only * len(micro_batches)
        truth = sum(counts)

        self.assertNotEqual(extrapolated, truth)
        self.assertGreater(abs(extrapolated - truth), 0.2 * truth)

    def test_uniform_micro_batches_make_the_extrapolation_exact(self):
        """With equal lengths the old estimate happens to be right, which hid the bug."""
        counts = [500] * 8
        micro_batches = [_micro_batch(n) for n in counts]

        first_only = count_loss_token(micro_batches[0])["foundation_tokens"]
        self.assertEqual(first_only * len(micro_batches), sum(counts))

    def test_single_micro_batch_is_unaffected(self):
        """The common DP-only case has one micro-batch, so nothing changes for it."""
        micro_batches = [_micro_batch(6304)]
        first_only = count_loss_token(micro_batches[0])["foundation_tokens"]
        self.assertEqual(first_only * 1, count_loss_token(micro_batches)["foundation_tokens"])


if __name__ == "__main__":
    unittest.main()
