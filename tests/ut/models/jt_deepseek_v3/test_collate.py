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
"""Standalone JT padding metadata through the shared text data path."""

from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.data.batching import TextParallelBatch
from hyper_parallel.data.parallel.batch_parallel import CPBatchSharder, TPBatchBroadcaster
from hyper_parallel.data.text.text_transform import PreTokenizedSFTTransform
from hyper_parallel.models.jt_deepseek_v3.adapter.data.collate import build_attention_padding_collate_fn
from hyper_parallel.models.jt_deepseek_v3.adapter.data.runtime import JTSequenceRuntime


class TestJTAttentionPaddingCollator(unittest.TestCase):
    """Verify tokens, supervision and global lengths independently of balancing."""

    def test_prepared_record_to_model_inputs(self) -> None:
        """Ignored prompt and interior pad IDs survive; only the trailing run is excluded."""
        record = {"input_ids": [9, 2, 3, 4, 9, 9], "labels": [-100, -100, 4, 5, -100, -100],
                  "cu_seqlens": [0, 2, 6]}
        samples = PreTokenizedSFTTransform()(record)
        mesh = SimpleNamespace(cp_size=1, pp_size=1, tp_size=1, dp_size=1, dp_rank=0, device_mesh=None)
        collator = build_attention_padding_collate_fn(mesh, padding_token_id=9)
        batch = collator(samples)
        runtime = TextParallelBatch(mesh, torch.device("cpu"), None, {}, False, source_type="online",
                                    attention_mode="compressed", reset_position_ids=True,
                                    runtime_input_adapter=JTSequenceRuntime())
        inputs, loss = runtime(iter([batch]))
        self.assertEqual(inputs["actual_seq_len"], (2, 4, 6))
        self.assertEqual(inputs["attention_seq_len"], (2, 4))
        torch.testing.assert_close(inputs["input_ids"], torch.tensor([record["input_ids"]]))
        torch.testing.assert_close(loss["shift_labels"], torch.tensor([record["labels"]]))
        torch.testing.assert_close(inputs["position_ids"], torch.tensor([[0, 1, 0, 1, 0, 1]]))
        self.assertEqual(samples[0]["cu_seq_lens"].tolist(), [0, 2, 6])

    def test_alignment_padding_and_full_record(self) -> None:
        """An omitted pad ID excludes only new alignment padding, preserving all input tokens."""
        samples = PreTokenizedSFTTransform()({"input_ids": [1, 2, 9, 9, 9],
                                            "labels": [2, -100, -100, -100, -100]})
        for alignment, expected in ((1, [0, 5]), (4, [0, 5, 8])):
            with self.subTest(alignment=alignment):
                mesh = SimpleNamespace(cp_size=alignment, tp_size=1, sequence_parallel=False)
                batch = build_attention_padding_collate_fn(mesh)(samples)
                self.assertEqual(batch["cu_seq_lens"].tolist(), expected)
                self.assertEqual(batch["attention_token_count"].tolist(), [5])
                torch.testing.assert_close(batch["input_ids"][0, :5], samples[0]["input_ids"])
                self.assertTrue(batch["labels"][0, 5:].eq(-100).all())

    def test_explicit_padding_validation(self) -> None:
        """Invalid contracts fail before model execution instead of silently removing targets."""
        for value in (-1, True, 1.5):
            with self.subTest(pad_id=value), self.assertRaisesRegex(ValueError, "non-negative integer"):
                build_attention_padding_collate_fn(padding_token_id=value)
        collator = build_attention_padding_collate_fn(padding_token_id=9)
        for tokens, labels, message in (([1, 9], [2, 3], "ignored labels"),
                                        ([9, 9], [-100, -100], "non-padding token")):
            with self.subTest(tokens=tokens), self.assertRaisesRegex(ValueError, message):
                collator([{"input_ids": torch.tensor(tokens), "labels": torch.tensor(labels)}])
        for samples in ([], [{}, {}]):
            with self.assertRaisesRegex(ValueError, "one prepacked record"):
                collator(samples)

    def test_global_metadata_survives_cp_and_tp(self) -> None:
        """An entirely padded CP shard still receives the same global Attention domain on both TP ranks."""
        samples = PreTokenizedSFTTransform()({"input_ids": [1, 2, 9, 9, 9, 9, 9, 9],
                                            "labels": [2, 3, -100, -100, -100, -100, -100, -100]})
        batch = build_attention_padding_collate_fn(padding_token_id=9)(samples)
        for cp_rank in (0, 1):
            with self.subTest(cp_rank=cp_rank):
                topology = SimpleNamespace(cp_world_size=2, cp_rank=cp_rank, tp_rank=0,
                                           tp_world_size=2, tp_group=object())
                local = CPBatchSharder(topology).shard(batch)
                self.assertEqual(local["input_ids"].shape, (1, 4))
                self.assertEqual(local["attention_token_count"].tolist(), [2])
                wire = []

                def broadcast(tensor: torch.Tensor, **_kwargs: Any) -> None:
                    """Replay source messages in order without a real process group."""
                    if topology.tp_rank == 0:
                        wire.append(tensor.clone())
                    else:
                        tensor.copy_(wire.pop(0))

                with patch("hyper_parallel.data.parallel.batch_parallel.dist.broadcast", side_effect=broadcast):
                    transport = TPBatchBroadcaster(topology, torch.device("cpu"))
                    source = transport.broadcast(local, batch["cu_seq_lens"])
                    topology.tp_rank = 1
                    receiver = transport.broadcast(None)
                self.assertFalse(wire)
                torch.testing.assert_close(source, receiver)
                for parallel_batch in (source, receiver):
                    runtime = JTSequenceRuntime().build(batch=parallel_batch, parallel_context=topology)
                    self.assertEqual(runtime, {"actual_seq_len": (2, 8), "attention_seq_len": (2,)})
