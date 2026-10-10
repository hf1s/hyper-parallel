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
"""Multi-token-prediction objective: per-depth targets and depth weighting."""

import unittest

import torch
import torch.nn.functional as F
from transformers.loss.loss_utils import ForCausalLMLoss

from hyper_parallel.components.losses import calculate_mtp_loss
from tests.common.mark_utils import arg_mark


class TestMultiTokenPredictionLoss(unittest.TestCase):
    """Depth ``k`` scores the token ``k`` positions after the main target."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_depth_targets_shift_without_wrapping(self):
        """Feature: MTP loss.

        Description: Score two depths of logits against main targets containing ignored positions.
        Expectation: Depth k uses the main targets shifted left by k and padded with the ignore
            index, not wrapped; the depths share ``loss_factor`` equally.
        """
        generator = torch.Generator().manual_seed(0)
        shift_labels = torch.tensor([[3, 1, -100, 4, 6], [2, 5, 0, -100, 1]])
        logits = [torch.randn(2, 5, 7, generator=generator) for _ in range(2)]

        loss = calculate_mtp_loss(logits, shift_labels, ForCausalLMLoss, vocab_size=7, loss_factor=0.3)

        targets = (
            torch.tensor([[1, -100, 4, 6, -100], [5, 0, -100, 1, -100]]),
            torch.tensor([[-100, 4, 6, -100, -100], [0, -100, 1, -100, -100]]),
        )
        expected = sum(0.15 * F.cross_entropy(depth_logits.reshape(-1, 7), depth_targets.reshape(-1))
                       for depth_logits, depth_targets in zip(logits, targets))
        torch.testing.assert_close(loss, expected)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_misaligned_logits_are_rejected(self):
        """Feature: MTP loss.

        Description: Pass depth logits one position shorter than the main targets.
        Expectation: ValueError, instead of scoring positions against the wrong tokens.
        """
        with self.assertRaisesRegex(ValueError, "align"):
            calculate_mtp_loss([torch.zeros(1, 4, 7)], torch.zeros(1, 5, dtype=torch.long),
                               ForCausalLMLoss, vocab_size=7)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_depth_without_targets_contributes_zero(self):
        """Feature: MTP loss.

        Description: Score two depths of a sequence where only depth 1 has a valid target.
        Expectation: Depth 2 adds zero and receives zero gradient instead of turning the loss into NaN.
        """
        generator = torch.Generator().manual_seed(1)
        logits = [torch.randn(1, 4, 7, generator=generator).requires_grad_() for _ in range(2)]
        shift_labels = torch.tensor([[-100, 3, -100, -100]])

        loss = calculate_mtp_loss(logits, shift_labels, ForCausalLMLoss, vocab_size=7, loss_factor=0.3)
        loss.backward()

        torch.testing.assert_close(loss, 0.15 * F.cross_entropy(logits[0][0, :1].detach(), torch.tensor([3])))
        torch.testing.assert_close(logits[1].grad, torch.zeros_like(logits[1]))


if __name__ == "__main__":
    unittest.main()
