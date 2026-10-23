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
"""Pre-tokenized SFT uses the same supervision and boundaries through direct Arrow loading."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from hyper_parallel.data.batching.build_collate_fn import TextPackingCollator
from hyper_parallel.data.text.text_transform import PreTokenizedSFTTransform
from tests.common.mark_utils import arg_mark


class TestPreTokenizedSFTTransform(unittest.TestCase):
    """Check prepared record contracts, ownership and nested packing."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_complete_record_preserves_targets_and_boundaries(self) -> None:
        """A nonstandard length remains one record without splitting or shifting labels."""
        record = {"input_ids": [151000, 2, 3, 4, 5, 6, 7],
                  "labels": [2, -100, 4, 5, 6, 7, -100], "cu_seqlens": [0, 3, 7]}
        samples = PreTokenizedSFTTransform()(record)
        self.assertEqual(len(samples), 1)
        torch.testing.assert_close(samples[0]["input_ids"], torch.tensor(record["input_ids"]))
        torch.testing.assert_close(samples[0]["labels"], torch.tensor(record["labels"]))
        self.assertEqual(samples[0]["cu_seq_lens"].tolist(), [0, 3, 7])
        self.assertEqual(record["labels"], [2, -100, 4, 5, 6, 7, -100])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_optional_mask_filter_and_field_mapping(self) -> None:
        """Prepared sources may rename columns and omit single-document boundaries."""
        transform = PreTokenizedSFTTransform(input_ids_key="tokens", labels_key="targets")
        record = {"tokens": [1, 2, 3, 4], "targets": [2, 3, 4, -100], "loss_mask": [0, 0, 1, 0]}
        samples = transform(record)
        self.assertEqual(len(samples), 1)
        torch.testing.assert_close(samples[0]["input_ids"], torch.tensor([1, 2, 3, 4]))
        torch.testing.assert_close(samples[0]["labels"], torch.tensor([-100, -100, 4, -100]))
        self.assertFalse(transform.is_valid_sample(dict(record, loss_mask=[0, 0, 0, 0])))
        self.assertEqual(transform(dict(record, loss_mask=[0, 0, 0, 0])), [])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_invalid_contracts_fail(self) -> None:
        """Invalid supervision is reported instead of silently coercing token IDs."""
        valid = {"input_ids": [1, 2], "labels": [2, -100], "cu_seqlens": [0, 2]}
        changes = [{"input_ids": [1.5, 2]}, {"labels": [2]}, {"labels": [2, -1]},
                   {"cu_seqlens": [0, 2, 2]}, {"cu_seqlens": [0, 1]},
                   {"loss_mask": [1, 0.5]}, {"input_ids": [1, 2**32]}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                PreTokenizedSFTTransform()(dict(valid, **change))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_readonly_arrays_keep_source_ownership(self) -> None:
        """Mask folding and downstream mutation must not write into source buffers."""
        for dtype in (np.int32, np.int64, np.uint32, np.uint64):
            with self.subTest(dtype=dtype):
                record = {"input_ids": np.asarray([1, 2, 3, 4], dtype=dtype),
                          "labels": np.asarray([2, 3, 4, 5], dtype=dtype),
                          "cu_seqlens": np.asarray([0, 2, 4], dtype=dtype),
                          "loss_mask": np.asarray([1, 0, 1, 0])}
                original = {key: value.copy() for key, value in record.items()}
                for value in record.values():
                    value.flags.writeable = False
                sample = PreTokenizedSFTTransform()(record)[0]
                self.assertEqual(sample["labels"].tolist(), [2, -100, 4, -100])
                self.assertEqual(sample["input_ids"].dtype, torch.long)
                self.assertEqual(sample["labels"].dtype, torch.long)
                self.assertEqual(sample["cu_seq_lens"].dtype, torch.int32)
                for value in sample.values():
                    value.zero_()
                for key, value in record.items():
                    np.testing.assert_array_equal(value, original[key])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_collator_preserves_internal_boundaries_and_padding(self) -> None:
        """Repacking a packed record must not merge its internal documents."""
        samples = PreTokenizedSFTTransform()(
            {"input_ids": [1, 2, 3], "labels": [2, -100, 4], "cu_seqlens": [0, 2, 3]})
        samples.append({"input_ids": torch.tensor([4, 5]), "labels": torch.tensor([5, -100])})
        batch = TextPackingCollator(sequence_parallel_size=4)(samples)
        self.assertEqual(batch["cu_seq_lens"].tolist(), [0, 2, 3, 5, 8])
        torch.testing.assert_close(batch["labels"], torch.tensor([[2, -100, 4, 5, -100, -100, -100, -100]]))
        self.assertEqual(samples[0]["cu_seq_lens"].tolist(), [0, 2, 3])
        for ends in ([0, 3, 2], [0, 2], [0., 3.]):
            with self.subTest(ends=ends), self.assertRaises(ValueError):
                TextPackingCollator()([dict(samples[0], cu_seq_lens=ends)])
