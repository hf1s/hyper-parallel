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
"""Build final batch collators for Indexed and Online sources."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import default_collate

from hyper_parallel.data.constants import IGNORE_INDEX, OMNI_CONCAT_FIELDS


def _get_sequence_parallel_size(mesh_context: Any | None) -> int:
    """Return the unified sequence-sharding degree used for batch alignment."""
    if mesh_context is None:
        return 1

    cp_size = int(getattr(mesh_context, "cp_size", 1))
    tp_size = int(getattr(mesh_context, "tp_size", 1))
    sequence_parallel = bool(getattr(mesh_context, "sequence_parallel", False))
    if cp_size <= 0 or tp_size <= 0:
        raise ValueError("mesh_context tp_size and cp_size must be positive")

    return cp_size * (tp_size if sequence_parallel else 1)


@dataclass
class TextPackingCollator:
    """Pack unpadded Online text samples and emit ``cu_seq_lens``.

    Existing per-sample ``cu_seq_lens`` are preserved when packing again.
    ``input_ids`` and pre-shifted ``labels`` are assembled here. Unified
    ``get_batch`` constructs loss, position, mask, and CP runtime fields.

    Args:
        sequence_parallel_size: Positive alignment width for each output row.
        pack_across_samples: Concatenate samples into one row by default;
            false stacks equal-length samples into independent rows.
    """

    sequence_parallel_size: int = 1
    pack_across_samples: bool = True

    def __call__(self, model_samples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Pack samples into one packed sequence or one packed batch of rows.

        ``pack_across_samples`` keeps the historical ``[1, packed_length]``
        layout, where every sample becomes one or more packed documents.
        Disabling it emits ``[batch, padded_length]`` with one independent
        sample per row, which is what ``micro_batch_size > 1`` means for a
        model that requires a fixed row width.

        Args:
            model_samples: Online samples selected for one forward-backward step.

        Returns:
            Packed ``input_ids`` and ``labels`` plus int32 ``cu_seq_lens`` whose
            entries are cumulative ends of the flattened row stream.

        Raises:
            ValueError: If no samples are provided, boundaries are malformed,
                or batch rows do not share one token count.
        """
        if not model_samples:
            raise ValueError("model_samples must contain at least one Online sample")
        if self.pack_across_samples:
            return self._pack_one_sequence(model_samples)
        return self._pack_batch_rows(model_samples)

    @staticmethod
    def _sample_ends(sample: Mapping[str, Any], offset: int) -> torch.Tensor:
        """Validate one sample's document ends and move them to packed coordinates."""
        tokens = sample["input_ids"]
        boundaries = torch.as_tensor(sample.get("cu_seq_lens", [0, tokens.numel()]), device=tokens.device)
        if (boundaries.dtype not in (torch.int32, torch.int64) or boundaries.ndim != 1
                or boundaries.numel() < 2 or boundaries[0] != 0 or boundaries[-1] != tokens.numel()
                or torch.any(boundaries[1:] <= boundaries[:-1])):
            raise ValueError("Sample boundaries must strictly increase from zero to its token count")
        return boundaries[1:].to(torch.int64) + offset

    def _pack_one_sequence(self, model_samples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Concatenate every sample into the historical single packed sequence."""
        packed_batch = {}
        for field in ("input_ids", "labels"):
            packed_batch[field] = torch.cat([sample[field] for sample in model_samples], dim=-1).unsqueeze(0)

        pad_len = (-packed_batch["input_ids"].shape[-1]) % self.sequence_parallel_size
        if pad_len:
            packed_batch["input_ids"] = torch.cat(
                (packed_batch["input_ids"], packed_batch["input_ids"].new_zeros((1, pad_len))), dim=-1)
            packed_batch["labels"] = torch.cat(
                (packed_batch["labels"], packed_batch["labels"].new_full((1, pad_len), IGNORE_INDEX)), dim=-1)

        sample_ends = []
        offset = 0
        for sample in model_samples:
            sample_ends.append(self._sample_ends(sample, offset))
            offset += sample["input_ids"].numel()
        seq_ends = torch.cat(sample_ends)
        if pad_len:
            # Represent the alignment tail as one synthetic packed sequence so
            # attention metadata covers every physical Q/KV token. Its labels
            # remain IGNORE_INDEX and do not contribute to supervised token losses.
            seq_ends = torch.cat((seq_ends, seq_ends[-1:] + pad_len))
        packed_batch["cu_seq_lens"] = torch.cat((seq_ends.new_zeros(1), seq_ends)).to(torch.int32)
        lengths = [sample["input_ids"].numel() for sample in model_samples]
        ends = torch.tensor(lengths, dtype=torch.int32, device=seq_ends.device).cumsum(0)
        packed_batch["sample_ranges"] = torch.stack((ends - ends.new_tensor(lengths), ends), dim=-1)
        return packed_batch

    def _pack_batch_rows(self, model_samples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Stack equal-length samples into independent rows of one batch."""
        for sample in model_samples:
            if (sample["input_ids"].ndim != 1 or sample["input_ids"].numel() == 0
                    or sample["labels"].shape != sample["input_ids"].shape):
                raise ValueError("Batch rows require nonempty one-dimensional input_ids and matching labels")
        lengths = {int(sample["input_ids"].numel()) for sample in model_samples}
        if len(lengths) != 1:
            raise ValueError("Batch rows require equal sample lengths; pack the samples instead")
        length = lengths.pop()
        row_stride = length + (-length) % self.sequence_parallel_size
        pad_len = row_stride - length

        input_ids = torch.stack([sample["input_ids"] for sample in model_samples], dim=0)
        labels = torch.stack([sample["labels"] for sample in model_samples], dim=0)
        if pad_len:
            input_ids = torch.cat((input_ids, input_ids.new_zeros((input_ids.shape[0], pad_len))), dim=-1)
            labels = torch.cat((labels, labels.new_full((labels.shape[0], pad_len), IGNORE_INDEX)), dim=-1)

        sample_ends = []
        for row, sample in enumerate(model_samples):
            offset = row * row_stride
            sample_ends.append(self._sample_ends(sample, offset))
            if pad_len:
                # Every row owns its alignment tail as a separate document, so the
                # flattened varlen metadata still covers each physical token.
                sample_ends.append(input_ids.new_tensor([offset + row_stride]))
        seq_ends = torch.cat(sample_ends)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "cu_seq_lens": torch.cat((seq_ends.new_zeros(1), seq_ends)).to(torch.int32),
            "sample_ranges": input_ids.new_tensor([
                [row * row_stride, row * row_stride + length] for row in range(len(model_samples))]),
        }


class OmniCollator:
    """Batch already prepared Omni samples without performing sequence packing."""

    def __call__(self, model_samples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Stack token fields and concatenate variable-length modality fields."""
        if not model_samples:
            raise ValueError("model_samples must contain at least one Omni sample")

        sample_fields = set()
        for model_sample in model_samples:
            sample_fields.update(model_sample)

        batch = {}
        for field in sample_fields:
            values = []
            for model_sample in model_samples:
                if field in model_sample:
                    values.append(model_sample[field])

            if field in OMNI_CONCAT_FIELDS:
                batch[field] = torch.cat(values, dim=0)
            else:
                batch[field] = default_collate(values)

        return batch


def build_default_collate_fn() -> Callable[[list[Any]], Any]:
    """Build ordinary PyTorch collation for already prepared samples.

    Returns:
        PyTorch default collation.
    """
    collate_fn = default_collate

    return collate_fn


# Indexed samples are already prepared and use the same ordinary collation.
build_indexed_collate_fn = build_default_collate_fn


def build_omni_collate_fn() -> OmniCollator:
    """Build ordinary collation for encoded or finally packed Omni samples."""
    collate_fn = OmniCollator()
    return collate_fn


def build_online_text_collate_fn(mesh_context: Any | None = None, *,
                                 pack_across_samples: bool = True) -> TextPackingCollator:
    """Build Online text collation shared by fixed N and dynamic K batching.

    Packing concatenates ``input_ids`` and ``labels`` and emits ``cu_seq_lens``.
    Tail padding is derived from the runtime topology rather than user config.
    The unified alignment size is ``cp_size * tp_size`` when TP sequence
    parallelism is enabled, otherwise it is ``cp_size``. Real sample boundaries
    remain unchanged and one synthetic final boundary covers the physical
    padding segment. With ``pack_across_samples=False``, equal-length samples
    remain independent rows and each row has its own alignment segment.

    Args:
        mesh_context: Runtime TP/CP topology injected by the Trainer.
        pack_across_samples: Whether to concatenate samples into one row.

    Returns:
        A collator producing one forward-backward batch.
    """
    sequence_parallel_size = _get_sequence_parallel_size(mesh_context)
    collate_fn = TextPackingCollator(sequence_parallel_size=sequence_parallel_size,
                                    pack_across_samples=pack_across_samples)

    return collate_fn
