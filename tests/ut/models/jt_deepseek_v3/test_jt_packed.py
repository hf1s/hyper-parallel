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
"""Packed SFT boundaries through native JT loading, attention and MTP."""

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
import torch.nn.functional as F

from examples.training_demo.jt_deepseek_v3.prepare_packed_sft import convert_sft
from hyper_parallel.components.losses.mtp import calculate_mtp_loss
from hyper_parallel.components.modules.mtp import shift_mtp_sequence
from hyper_parallel.data.batching import FixedBatchDataLoader, TextParallelBatch
from hyper_parallel.data.batching.build_collate_fn import TextPackingCollator
from hyper_parallel.data.batching.runtime_input import IndexedBoundaryResolver, OnlineBoundaryResolver
from hyper_parallel.data.indexed.indexed_supervised_dataset import IndexedSupervisedDataset
from hyper_parallel.models.jt_deepseek_v3.adapter.data.runtime import JTPackedRuntime
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3ForCausalLM
from tests.common.mark_utils import arg_mark
from tests.ut.models.jt_deepseek_v3.test_jt_data import _write
from tests.ut.models.jt_deepseek_v3.test_modeling_jt_deepseek_v3 import small_config


class TestJTPacked(unittest.TestCase):
    """Use real indexed files and CPU forward/backward for the optional packed path."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_dataset_to_model_inputs(self):
        """Stored boundaries override EOD inference and reset positions at each document."""
        with TemporaryDirectory() as directory:
            prefix = Path(directory) / "train"
            for name, records in (("tokens", [[3, 4, 5, 6, 7, 8]]),
                                  ("labels", [[4, 5, -100, 7, 8, -100]]),
                                  ("loss_mask", [[1, 1, 0, 1, 1, 0]]),
                                  ("cu_seqlens", [[0, 3, 6]])):
                _write(prefix, name, records, np.int32)
            dataset = IndexedSupervisedDataset(prefix, sequence_length=6, packed=True)
            iterator = iter(FixedBatchDataLoader(dataset, batch_size=1, drop_last=True, num_workers=0))
            mesh = SimpleNamespace(cp_size=1, pp_size=1, tp_size=1, dp_size=1, dp_rank=0, device_mesh=None)
            batch = TextParallelBatch(mesh, torch.device("cpu"), SimpleNamespace(eod=4), {}, False,
                                      source_type="indexed", attention_mode="compressed", reset_position_ids=True,
                                      runtime_input_adapter=JTPackedRuntime())
            model, loss = batch(iterator)
            self.assertEqual(model["cu_seq_lens"], (0, 3, 6))
            torch.testing.assert_close(model["position_ids"], torch.tensor([[0, 1, 2, 0, 1, 2]]))
            torch.testing.assert_close(loss["shift_labels"], torch.tensor([[4, 5, -100, 7, 8, -100]]))
            sample = dataset[0]
            sample["cu_seq_lens"][1] = 99
            self.assertEqual(dataset[0]["cu_seq_lens"].tolist(), [0, 3, 6])
            _write(prefix, "cu_seqlens", [[0, 4, 3, 6]], np.int32)
            with self.assertRaisesRegex(ValueError, "strictly increase"):
                _ = IndexedSupervisedDataset(prefix, packed=True)[0]

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_explicit_boundary_validation_and_legacy_fallback(self):
        """Malformed boundaries fail, while ordinary indexed records still infer EOD."""
        resolver = IndexedBoundaryResolver(9)
        batch = {"input_ids": torch.tensor([[1, 9, 2, 3]])}
        self.assertEqual(resolver.resolve(batch).tolist(), [0, 2, 4])
        for value in ([0, 4, 4], [1, 4], [0, 3], [0., 4.]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolver.resolve(dict(batch, cu_seq_lens=torch.tensor(value)))
        two_rows = {"input_ids": torch.ones(2, 4, dtype=torch.long),
                    "cu_seq_lens": torch.tensor([[0, 1, 4], [0, 2, 4]])}
        self.assertEqual(resolver.resolve(two_rows).tolist(), [0, 1, 4, 6, 8])
        two_rows["cu_seq_lens"] = torch.tensor([0, 2, 4, 5, 8])
        self.assertEqual(resolver.resolve(two_rows).tolist(), [0, 2, 4, 5, 8])
        two_rows["cu_seq_lens"] = torch.tensor([0, 3, 8])
        with self.assertRaisesRegex(ValueError, "row end"):
            resolver.resolve(two_rows)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_online_and_indexed_boundaries_share_runtime(self):
        """In-batch packing and indexed records use the same storage-independent metadata."""
        online = TextPackingCollator()([
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([2, 3, -100])},
            {"input_ids": torch.tensor([4, 5]), "labels": torch.tensor([5, -100])},
        ])
        indexed = dict(online, cu_seq_lens=online["cu_seq_lens"].unsqueeze(0))
        online["cu_seq_lens"] = OnlineBoundaryResolver.resolve(online)
        indexed["cu_seq_lens"] = IndexedBoundaryResolver(None).resolve(indexed)
        mesh = SimpleNamespace(cp_rank=0, tp_rank=0, cp_world_size=1, tp_world_size=1)
        runtime = JTPackedRuntime()
        self.assertEqual(runtime.build(batch=online, parallel_context=mesh),
                         runtime.build(batch=indexed, parallel_context=mesh))
        self.assertEqual(runtime.build(batch=online, parallel_context=mesh)["cu_seq_lens"], (0, 3, 5))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_single_document_keeps_native_losses_and_gradients(self):
        """Enabling packed metadata for one document preserves the native path."""
        torch.manual_seed(5)
        native = JTDeepseekV3ForCausalLM(small_config())
        packed = copy.deepcopy(native)
        tokens = torch.arange(8).unsqueeze(0)
        labels = (tokens + 1) % 32
        left = native(tokens, labels).loss
        right = packed(tokens, labels, cu_seq_lens=(0, 8)).loss
        for key in left:
            torch.testing.assert_close(left[key], right[key])
        sum(left.values()).backward()
        sum(right.values()).backward()
        for (name, param), (_, reference) in zip(packed.named_parameters(), native.named_parameters()):
            with self.subTest(parameter=name):
                torch.testing.assert_close(param.grad, reference.grad)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_packed_lm_mtp_equal_independent_documents(self):
        """Packed losses and gradients equal token-weighted independent document runs."""
        torch.manual_seed(8)
        config = small_config()
        config.moe_aux_loss_coeff = 0.0
        config.num_nextn_predict_layers = 2
        packed = JTDeepseekV3ForCausalLM(config)
        separate = copy.deepcopy(packed)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        labels = torch.tensor([[2, 3, 4, -100, 6, 7, 8, 9, -100]])
        output = packed(tokens, labels, cu_seq_lens=(0, 4, 9)).loss
        independent = [separate(tokens[:, start:end], labels[:, start:end]).loss
                       for start, end in ((0, 4), (4, 9))]
        # LM has 3/4 valid targets, while MTP depths have 2/3 then 1/2.
        expected_lm = (independent[0]["foundation_loss/lm"] * 3 +
                       independent[1]["foundation_loss/lm"] * 4) / 7
        torch.testing.assert_close(output["foundation_loss/lm"], expected_lm, rtol=2e-5, atol=2e-6)
        # Equal document lengths make every depth use the same weighting for gradient comparison.
        tokens = tokens[:, :8]
        labels = torch.tensor([[2, 3, 4, -100, 6, 7, 8, -100]])
        output = packed(tokens, labels, cu_seq_lens=(0, 4, 8)).loss
        left = separate(tokens[:, :4], labels[:, :4]).loss
        right = separate(tokens[:, 4:], labels[:, 4:]).loss
        keys = ("foundation_loss/lm", "foundation_loss/mtp")
        expected = {key: (left[key] + right[key]) / 2 for key in keys}
        for key in keys:
            torch.testing.assert_close(output[key], expected[key], rtol=2e-5, atol=2e-6)
        sum(output[key] for key in keys).backward()
        sum(expected.values()).backward()
        for (name, param), (_, reference) in zip(packed.named_parameters(), separate.named_parameters()):
            with self.subTest(parameter=name):
                torch.testing.assert_close(param.grad, reference.grad, rtol=3e-4, atol=2e-6)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_mtp_short_documents_never_read_neighbors(self):
        """Repeated shifts mask short documents, including all-ignored MTP targets."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        shifted = shift_mtp_sequence(tokens, sequence_ends=(1, 3, 4))
        torch.testing.assert_close(shifted, torch.tensor([[0, 3, 0, 0]]))
        shifted = shift_mtp_sequence(shifted, sequence_ends=(1, 3, 4))
        self.assertEqual(shifted.count_nonzero().item(), 0)
        logits = [torch.randn(1, 4, 8, requires_grad=True) for _ in range(2)]

        def loss_fn(*, logits, shift_labels, num_items_in_batch, **_kwargs):
            return F.cross_entropy(logits.flatten(0, 1), shift_labels.flatten(),
                                   reduction="sum", ignore_index=-100) / num_items_in_batch

        loss = calculate_mtp_loss(logits, tokens, loss_fn, vocab_size=8, sequence_ends=(1, 2, 3, 4))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        for value in logits:
            torch.testing.assert_close(value.grad, torch.zeros_like(value))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_arrow_conversion_preserves_labels_and_clips_windows(self):
        """Convert real Arrow buffers without shifting labels or dropping internal boundaries."""
        tokens = [151000, 3, 4, 5, 6, 7, 8, 9]
        labels = [3, -100, 5, 6, 7, -100, 9, -100]
        with TemporaryDirectory() as directory:
            source = Path(directory) / "demo.arrow"
            table = pa.table({"input_ids": [tokens], "labels": [labels], "cu_seqlens": [[0, 3, 8]]})
            with pa.OSFile(str(source), "wb") as stream, ipc.new_stream(stream, table.schema) as writer:
                writer.write_table(table)
            full = Path(directory) / "full"
            windows = Path(directory) / "windows"
            self.assertEqual(convert_sft(source, full), 1)
            sample = IndexedSupervisedDataset(full, packed=True)[0]
            np.testing.assert_array_equal(sample["tokens"], tokens)
            np.testing.assert_array_equal(sample["labels"], labels)
            np.testing.assert_array_equal(sample["cu_seq_lens"], [0, 3, 8])
            self.assertEqual(convert_sft(source, windows, 4), 2)
            dataset = IndexedSupervisedDataset(windows, sequence_length=4, packed=True)
            for index, ends in enumerate(([0, 3, 4], [0, 4])):
                np.testing.assert_array_equal(dataset[index]["tokens"], tokens[index * 4:(index + 1) * 4])
                np.testing.assert_array_equal(dataset[index]["labels"], labels[index * 4:(index + 1) * 4])
                np.testing.assert_array_equal(dataset[index]["cu_seq_lens"], ends)
            with self.assertRaises(FileExistsError):
                convert_sft(source, full)
            with self.assertRaisesRegex(ValueError, "divisible"):
                convert_sft(source, Path(directory) / "invalid", 3)
