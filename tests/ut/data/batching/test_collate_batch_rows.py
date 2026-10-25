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
"""Batch-row packing keeps one document boundary list over physical tokens."""

import unittest

import torch

from hyper_parallel.data.batching.build_collate_fn import (
    TextPackingCollator,
    build_online_text_collate_fn,
)
from tests.common.mark_utils import arg_mark


def _sample(tokens: list[int], boundaries: list[int] | None = None) -> dict[str, torch.Tensor]:
    """Build one pre-tokenized sample with optional packed document ends."""
    sample = {
        "input_ids": torch.tensor(tokens, dtype=torch.long),
        "labels": torch.tensor([token + 1 for token in tokens], dtype=torch.long),
    }
    if boundaries is not None:
        sample["cu_seq_lens"] = torch.tensor(boundaries, dtype=torch.int32)
    return sample


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
class TestTextPackingCollator(unittest.TestCase):
    """Both layouts end at the physical token count and keep inner documents."""

    def test_single_sequence_preserves_internal_documents(self) -> None:
        """The historical layout still concatenates samples into one sequence."""
        batch = TextPackingCollator()([_sample([1, 2, 3, 4], [0, 2, 4]), _sample([5, 6, 7])])

        self.assertEqual(tuple(batch["input_ids"].shape), (1, 7))
        torch.testing.assert_close(batch["cu_seq_lens"], torch.tensor([0, 2, 4, 7], dtype=torch.int32))

    def test_batch_rows_stack_equal_length_samples(self) -> None:
        """Each sample becomes one row and keeps its own document ends."""
        batch = TextPackingCollator(pack_across_samples=False)(
            [_sample([1, 2, 3, 4], [0, 2, 4]), _sample([5, 6, 7, 8])])

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 4))
        self.assertEqual(batch["cu_seq_lens"].dtype, torch.int32)
        torch.testing.assert_close(batch["cu_seq_lens"], torch.tensor([0, 2, 4, 8], dtype=torch.int32))

    def test_batch_rows_pad_every_row_before_its_next_boundary(self) -> None:
        """Sequence-parallel padding is its own document in each row."""
        batch = TextPackingCollator(sequence_parallel_size=4, pack_across_samples=False)(
            [_sample([1, 2, 3, 4, 5], [0, 2, 5]), _sample([6, 7, 8, 9, 10])])

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 8))
        torch.testing.assert_close(
            batch["cu_seq_lens"], torch.tensor([0, 2, 5, 8, 13, 16], dtype=torch.int32))
        self.assertEqual(int(batch["labels"][0, 5]), -100)
        self.assertEqual(int(batch["input_ids"][1, 5]), 0)

    def test_batch_rows_reject_unequal_lengths(self) -> None:
        """Rows cannot be stacked without a common token count."""
        with self.assertRaisesRegex(ValueError, "equal sample lengths"):
            TextPackingCollator(pack_across_samples=False)([_sample([1, 2, 3, 4]), _sample([5, 6, 7])])

    def test_malformed_boundaries_are_rejected(self) -> None:
        """A non-increasing boundary would silently merge two documents."""
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            TextPackingCollator()([_sample([1, 2, 3, 4], [0, 3, 3])])

    def test_factory_forwards_the_row_layout(self) -> None:
        """The shared builder can request independent rows without new plumbing."""
        collate_fn = build_online_text_collate_fn(pack_across_samples=False)
        batch = collate_fn([_sample([1, 2, 3, 4], [0, 2, 4]), _sample([5, 6, 7, 8])])

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 4))
        torch.testing.assert_close(batch["cu_seq_lens"], torch.tensor([0, 2, 4, 8], dtype=torch.int32))


if __name__ == "__main__":
    unittest.main()
