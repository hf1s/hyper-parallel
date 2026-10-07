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
"""Supervised indexed storage through the shared text batch."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from hyper_parallel.data.tools.io import IndexedDatasetBuilder
from hyper_parallel.models.jt_deepseek_v3.adapter.runtime.jt_data import IndexedSupervisedDataset
from hyper_parallel.data.batching import FixedBatchDataLoader, TextParallelBatch
from tests.common.mark_utils import arg_mark


def _write(prefix, name, records, dtype):
    writer = IndexedDatasetBuilder(str(prefix) + "." + name + ".bin", dtype=dtype)
    for row in records:
        writer.add_item(torch.tensor(row))
        writer.end_document()
    writer.finalize(str(prefix) + "." + name + ".idx")


class TestIndexedSupervisedDataset(unittest.TestCase):
    """Fold the 0/1 supervision mask into labels and leave sampling to the DataLoader."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_public_batch_derives_mask_from_folded_labels(self):
        """Feature: Public supervised data.

        Description: Load paired indexed fields through the public loader and the default text batch.
        Expectation: Masked valid targets become IGNORE_INDEX; the batch mask equals the stored 0/1 mask.
        """
        with TemporaryDirectory() as directory:
            prefix = Path(directory) / "data"
            tokens = [[3, 4, 5, 6], [6, 5, 4, 3]]
            labels = [[4, -100, 6, -100], [5, 4, 3, -100]]
            masks = [[1., 0., 1., 0.], [0., 1., 1., 0.]]
            folded = [[4, -100, 6, -100], [-100, 4, 3, -100]]
            for name, records, dtype in [("tokens", tokens, np.int32), ("labels", labels, np.int32),
                                          ("loss_mask", masks, np.float32)]:
                _write(prefix, name, records, dtype)
            dataset = IndexedSupervisedDataset(prefix, sequence_length=4)
            iterator = iter(FixedBatchDataLoader(dataset, batch_size=1, drop_last=True, num_workers=0))
            mesh = SimpleNamespace(cp_size=1, pp_size=1, tp_size=1, dp_size=1, dp_rank=0, device_mesh=None)
            batch = TextParallelBatch(mesh, torch.device("cpu"), None, {}, False,
                                      source_type="indexed", attention_mode="compressed")
            for index in range(2):
                model, loss = batch(iterator)
                torch.testing.assert_close(model["input_ids"], torch.tensor([tokens[index]]), rtol=0, atol=0)
                torch.testing.assert_close(loss["shift_labels"], torch.tensor([folded[index]]), rtol=0, atol=0)
                torch.testing.assert_close(loss["loss_mask"], torch.tensor([masks[index]], dtype=torch.int64),
                                           rtol=0, atol=0)
            # Returned arrays must not modify the read-only indexed payload.
            sample = dataset[0]
            sample["tokens"][0] = 99
            self.assertEqual(dataset[0]["tokens"][0], 3)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_record_mismatch_and_invalid_supervision(self):
        """Feature: Dataset validation.

        Description: Supply mismatched stream lengths, a masked-label conflict, then a fractional mask.
        Expectation: Invalid supervision fails before model execution.
        """
        with TemporaryDirectory() as directory:
            prefix = Path(directory) / "data"
            _write(prefix, "tokens", [[1, 2]], np.int32)
            _write(prefix, "labels", [[2]], np.int32)
            _write(prefix, "loss_mask", [[1., 0.]], np.float32)
            with self.assertRaisesRegex(ValueError, "lengths"):
                IndexedSupervisedDataset(prefix)
            _write(prefix, "labels", [[-100, 3]], np.int32)
            with self.assertRaisesRegex(ValueError, "Ignored labels"):
                _ = IndexedSupervisedDataset(prefix)[0]
            _write(prefix, "labels", [[2, 3]], np.int32)
            _write(prefix, "loss_mask", [[0.5, 1.]], np.float32)
            with self.assertRaisesRegex(ValueError, "0 or 1"):
                _ = IndexedSupervisedDataset(prefix)[0]
            with self.assertRaisesRegex(ValueError, "sequence_length"):
                IndexedSupervisedDataset(prefix, sequence_length=3)
