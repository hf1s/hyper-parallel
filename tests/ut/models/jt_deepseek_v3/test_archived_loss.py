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
"""Check the archived JT cross-entropy against native CE and FP64 references."""

import unittest

import torch
from torch.nn import functional as F

from hyper_parallel.models.jt_deepseek_v3.archived_loss import TreeBackwardCE, archived_cross_entropy, tiled_sum
from tests.common.mark_utils import arg_mark

VOCAB = 152064
TILE = 108 * 64
TINY = 2.0 ** -24  # Half an FP32 ulp of 1.0: 1 + TINY rounds back to 1 (ties to even).


def _rows() -> tuple[torch.Tensor, torch.Tensor]:
    """Return three full-vocabulary FP32 logit rows and targets with one ignored row."""
    logits = torch.randn(3, VOCAB, generator=torch.Generator().manual_seed(2026)) * 2
    logits[0] += 100  # exp would overflow FP32 without the max subtraction
    return logits, torch.tensor([7, -100, VOCAB - 1])


class TestArchivedCrossEntropy(unittest.TestCase):
    """Native forward value, FP64-accurate derivative and the fixed reduction schedule."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_forward_is_native_sum_cross_entropy(self) -> None:
        """The loss value is exactly native sum-reduced CE, including the ignored target."""
        logits, targets = _rows()
        expected = F.cross_entropy(logits, targets, reduction='sum', ignore_index=-100)
        torch.testing.assert_close(TreeBackwardCE.apply(logits, targets, -100), expected, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_backward_matches_double_precision_gradient(self) -> None:
        """The recomputed softmax derivative matches FP64 autograd and is zero on ignored rows."""
        logits, targets = _rows()
        count = (targets != -100).sum()
        inputs = logits.clone().requires_grad_()
        (TreeBackwardCE.apply(inputs, targets, -100) / count).backward()
        reference = logits.double().requires_grad_()
        (F.cross_entropy(reference, targets, reduction='sum', ignore_index=-100) / count).backward()
        self.assertEqual(inputs.grad.dtype, torch.float32)
        torch.testing.assert_close(inputs.grad[1], torch.zeros(VOCAB), rtol=0, atol=0)
        torch.testing.assert_close(inputs.grad.double(), reference.grad, rtol=1e-5, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_tiled_sum_matches_double_precision_row_sums(self) -> None:
        """Rows inside and after the complete 40-row blocks keep FP32 summation accuracy."""
        generator = torch.Generator().manual_seed(11)
        for rows in (3, 41):
            scale = torch.arange(1, rows + 1, dtype=torch.float32)[:, None]
            values = torch.rand(rows, VOCAB, generator=generator) * scale
            with self.subTest(rows=rows):
                sums = tiled_sum(values)
                self.assertEqual((sums.shape, sums.dtype), ((rows, 1), torch.float32))
                # Each value meets at most 36 FP32 adds (9 in the 108-tree, 6 over lanes,
                # 21 over tiles); with nonnegative values the relative error is below 36 * 2**-24.
                torch.testing.assert_close(sums.double(), values.double().sum(-1, keepdim=True), rtol=2.2e-6, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_tiled_sum_follows_archived_pairing_and_40_row_tail(self) -> None:
        """Hand-rounded FP32 sums pin the tile order before and after the last complete 40-row block."""
        # Tile sums 1, TINY, ..., TINY. Left-to-right tile adds round every TINY away; the tail
        # adds the 11 odd tiles first, and 1 + 11 * TINY rounds to 1 + 12 * TINY.
        tiles = torch.zeros(VOCAB)
        tiles[0] = 1.0
        tiles[TILE::TILE] = TINY
        expected = torch.tensor([1.0] * 40 + [1.0 + 12 * TINY])[:, None]
        torch.testing.assert_close(tiled_sum(tiles.expand(41, VOCAB).contiguous()), expected, rtol=0, atol=0)
        torch.testing.assert_close(tiled_sum(tiles[None]), expected[-1:], rtol=0, atol=0)
        # Within a tile, both TINY values survive only if they meet each other before meeting 1:
        # entries 1 and 55 of lane 0 pair in the first halving level; lanes 2 and 3 pair first.
        values = torch.zeros(2, VOCAB)
        values[0, [0, 64, 55 * 64]] = torch.tensor([1.0, TINY, TINY])
        values[1, [0, 2, 3]] = torch.tensor([1.0, TINY, TINY])
        torch.testing.assert_close(tiled_sum(values), torch.full((2, 1), 1.0 + 2 * TINY), rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_wrapper_uses_caller_denominator_and_ignores_labels(self) -> None:
        """The caller's valid-token count divides the native sum; ``labels`` is accepted and unused."""
        logits, targets = _rows()
        count = torch.tensor(2)
        expected = F.cross_entropy(logits, targets, reduction='sum', ignore_index=-100) / count
        for labels in (None, torch.zeros(3, dtype=torch.long)):
            with self.subTest(labels=labels is not None):
                actual = archived_cross_entropy(logits=logits, labels=labels, vocab_size=VOCAB,
                                                shift_labels=targets, num_items_in_batch=count)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_ignore_index_outside_the_vocabulary_is_masked_in_the_derivative(self) -> None:
        """An ignore_index beyond the vocabulary behaves like -100: native forward, zero gradient on its rows."""
        logits, targets = _rows()
        targets[1] = VOCAB
        inputs = logits.clone().requires_grad_()
        loss = TreeBackwardCE.apply(inputs, targets, VOCAB)
        torch.testing.assert_close(loss, F.cross_entropy(logits, targets, reduction='sum', ignore_index=VOCAB),
                                   rtol=0, atol=0)
        loss.backward()
        torch.testing.assert_close(inputs.grad[1], torch.zeros(VOCAB), rtol=0, atol=0)
        self.assertTrue(bool((inputs.grad[[0, 2]] != 0).any()))


if __name__ == "__main__":
    unittest.main()
