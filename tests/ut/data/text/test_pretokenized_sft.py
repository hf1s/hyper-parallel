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
"""Pre-tokenized SFT uses the same supervision and boundaries online and offline."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch

from hyper_parallel.data.batching import TokenBatchLoader
from hyper_parallel.data.batching.build_collate_fn import TextPackingCollator
from hyper_parallel.data.indexed.indexed_supervised_dataset import IndexedSupervisedDataset
from hyper_parallel.data.online.mapping import MappingTransformDataset
from hyper_parallel.data.parallel.batch_sampler import build_dataset_batch_sampler
from hyper_parallel.data.text.build_dataset import build_online_text_mapping_dataset
from hyper_parallel.data.text.pretokenized_sft import PreTokenizedSFTTransform
from hyper_parallel.data.tools.prepare_packed_sft import convert_sft
from tests.common.mark_utils import arg_mark


def _write_arrow(path: Path, records: list[dict]) -> None:
    table = pa.Table.from_pylist(records)
    with pa.OSFile(str(path), "wb") as stream, ipc.new_stream(stream, table.schema) as writer:
        writer.write_table(table)


def _loader(dataset, length: int, dp_rank: int | None = None) -> TokenBatchLoader:
    sampler = None
    if dp_rank is not None:
        sampler = build_dataset_batch_sampler(total_samples=len(dataset), micro_batch_size=1,
                                              global_batch_size=2, dp_rank=dp_rank, dp_world_size=2)
    return TokenBatchLoader(dataset, collate_fn=TextPackingCollator(sequence_parallel_size=2),
                            batch_sampler=sampler, batch_size=1, dp_world_size=1 if dp_rank is None else 2,
                            max_seq_len=length, min_buffered_samples=1)


class TestPreTokenizedSFTTransform(unittest.TestCase):
    """Exercise real Arrow IO, window boundaries, nested packing and loader resume."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_windows_preserve_targets_and_tail(self) -> None:
        """A cut across a document retains its next-token target without shifting twice."""
        record = {"input_ids": [151000, 2, 3, 4, 5, 6, 7],
                  "labels": [2, -100, 4, 5, 6, 7, -100], "cu_seqlens": [0, 3, 7]}
        samples = PreTokenizedSFTTransform(max_seq_len=4)(record)
        self.assertEqual(len(samples), 2)
        torch.testing.assert_close(samples[0]["labels"], torch.tensor([2, -100, 4, 5]))
        torch.testing.assert_close(samples[1]["labels"], torch.tensor([6, 7, -100]))
        self.assertEqual(samples[0]["cu_seq_lens"].tolist(), [0, 3, 4])
        self.assertEqual(samples[1]["cu_seq_lens"].tolist(), [0, 3])
        self.assertEqual(record["labels"], [2, -100, 4, 5, 6, 7, -100])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_optional_mask_filter_and_field_mapping(self) -> None:
        """Prepared sources may rename columns and omit single-document boundaries."""
        transform = PreTokenizedSFTTransform(max_seq_len=2, input_ids_key="tokens", labels_key="targets")
        record = {"tokens": [1, 2, 3, 4], "targets": [2, 3, 4, -100], "loss_mask": [0, 0, 1, 0]}
        samples = transform(record)
        self.assertEqual(len(samples), 1)
        torch.testing.assert_close(samples[0]["input_ids"], torch.tensor([3, 4]))
        torch.testing.assert_close(samples[0]["labels"], torch.tensor([4, -100]))
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
        for width in (0, -1, True, 2.5):
            with self.subTest(width=width), self.assertRaises(ValueError):
                PreTokenizedSFTTransform(max_seq_len=width)

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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_arrow_online_indexed_equivalence_and_resume(self) -> None:
        """Real source loading, lazy multi-window expansion and saved buffer replay agree."""
        records = [{"input_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                    "labels": [2, 3, -100, 5, 6, 7, 8, -100], "cu_seqlens": [0, 3, 8]}]
        with TemporaryDirectory() as directory:
            source = Path(directory) / "train.arrow"
            _write_arrow(source, records)
            prefix = Path(directory) / "indexed" / "train"
            self.assertEqual(convert_sft(source, prefix, 4), 2)
            indexed = IndexedSupervisedDataset(prefix, sequence_length=4, packed=True)
            online = build_online_text_mapping_dataset(
                data_path=str(source), data_config={"cache_dir": str(Path(directory) / "cache")},
                transform=PreTokenizedSFTTransform(max_seq_len=4))
            direct = online.get_item(0)
            for sample, saved in zip(direct, indexed):
                np.testing.assert_array_equal(sample["input_ids"].numpy(), saved["tokens"])
                np.testing.assert_array_equal(sample["labels"].numpy(), saved["labels"])
                np.testing.assert_array_equal(sample["cu_seq_lens"].numpy(), saved["cu_seq_lens"])
            loader = _loader(online, 4)
            iterator = iter(loader)
            first = next(iterator)
            state = loader.state_dict()
            remaining = list(iterator)
            restored = _loader(online, 4)
            restored.load_state_dict(state)
            replayed = list(restored)
            self.assertEqual(len(remaining), 1)
            self.assertEqual(len(replayed), 1)
            self.assertEqual(first["cu_seq_lens"].tolist(), [0, 3, 4])
            for key in remaining[0]:
                torch.testing.assert_close(replayed[0][key], remaining[0][key])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_dp_shards_restore_different_window_counts(self) -> None:
        """Each DP lane resumes its own unconsumed windows without overlap or loss."""
        records = []
        for index, length in enumerate((8, 12, 4, 8)):
            tokens = list(range(index * 20 + 1, index * 20 + 1 + length))
            records.append({"input_ids": tokens, "labels": tokens[1:] + [-100]})
        dataset = MappingTransformDataset(records, PreTokenizedSFTTransform(max_seq_len=4))
        seen_tokens = []
        counts = []
        for rank in range(2):
            loader = _loader(dataset, 4, dp_rank=rank)
            iterator = iter(loader)
            first = next(iterator)
            state = loader.state_dict()
            remaining = list(iterator)
            restored = _loader(dataset, 4, dp_rank=rank)
            restored.load_state_dict(state)
            replayed = list(restored)
            self.assertEqual(len(replayed), len(remaining))
            for actual, expected in zip(replayed, remaining):
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key])
            with self.assertRaisesRegex(ValueError, "DP world-size changes"):
                _loader(dataset, 4).load_state_dict(state)
            seen_tokens.append({value for batch in [first, *remaining] for value in batch["input_ids"].flatten().tolist()})
            counts.append(1 + len(remaining))
        self.assertEqual(counts, [3, 5])
        self.assertFalse(seen_tokens[0] & seen_tokens[1])
        self.assertEqual(seen_tokens[0] | seen_tokens[1], {value for row in records for value in row["input_ids"]})
