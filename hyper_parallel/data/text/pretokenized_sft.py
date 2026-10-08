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
"""Preserve pre-tokenized SFT supervision and document boundaries across data sources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from hyper_parallel.data.constants import IGNORE_INDEX


@dataclass
class PreTokenizedSFTTransform:
    """Convert already aligned SFT records into one or more model samples.

    Labels must already identify the next-token target at each input position;
    this transform never tokenizes or shifts labels. Optional cumulative document
    boundaries include zero and the full record length. Missing boundaries mean
    one document. Only windows without any supervised targets are omitted.

    Args:
        max_seq_len: Optional maximum window length. A shorter final window is
            retained; the collator owns parallel alignment padding.
        input_ids_key: Source field containing token IDs.
        labels_key: Source field containing pre-shifted labels.
        boundaries_key: Optional source field containing cumulative boundaries.
        loss_mask_key: Optional source field containing a binary supervision mask.
    """

    max_seq_len: int | None = None
    input_ids_key: str = "input_ids"
    labels_key: str = "labels"
    boundaries_key: str = "cu_seqlens"
    loss_mask_key: str = "loss_mask"

    def __post_init__(self) -> None:
        """Validate window configuration before reading any records."""
        if self.max_seq_len is not None and (
                isinstance(self.max_seq_len, bool) or not isinstance(self.max_seq_len, int) or self.max_seq_len <= 0):
            raise ValueError("max_seq_len must be a positive integer or None")

    def _read_record(self, sample: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate source fields without changing their token or target semantics."""
        try:
            tokens = np.asarray(sample[self.input_ids_key])
            labels = np.asarray(sample[self.labels_key])
        except KeyError as error:
            raise ValueError("SFT records require input IDs and pre-shifted labels") from error
        boundaries = np.asarray(sample.get(self.boundaries_key, [0, tokens.size]))
        if any(value.ndim != 1 or value.dtype.kind not in "iu" for value in (tokens, labels, boundaries)):
            raise ValueError("SFT fields must be one-dimensional integer arrays")
        if not len(tokens) or tokens.shape != labels.shape:
            raise ValueError("Tokens and pre-shifted labels must have matching nonempty shapes")
        if (np.any(tokens < 0) or np.any(tokens > np.iinfo(np.int32).max)
                or np.any((labels < 0) & (labels != IGNORE_INDEX))
                or np.any(labels > np.iinfo(np.int32).max)):
            raise ValueError("Token IDs must fit int32; ignored labels must equal -100")
        if (len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != len(tokens)
                or np.any(boundaries[1:] <= boundaries[:-1])):
            raise ValueError("Packed boundaries must strictly increase from zero to token count")
        labels = labels.astype(np.int64, copy=True)
        if self.loss_mask_key in sample:
            mask = np.asarray(sample[self.loss_mask_key])
            if mask.shape != labels.shape or not np.isin(mask, (0, 1)).all():
                raise ValueError("SFT loss_mask must be binary and match labels")
            labels[mask == 0] = IGNORE_INDEX
        return tokens, labels, boundaries

    def is_valid_sample(self, sample: Mapping[str, Any]) -> bool:
        """Filter wholly unsupervised records before source sampling."""
        _, labels, _ = self._read_record(sample)
        return bool(np.any(labels != IGNORE_INDEX))

    def __call__(self, sample: Mapping[str, Any]) -> list[dict[str, torch.Tensor]]:
        """Preserve labels and clip document boundaries into nonoverlapping windows."""
        tokens, labels, boundaries = self._read_record(sample)
        width = len(tokens) if self.max_seq_len is None else self.max_seq_len
        windows = []
        for start in range(0, len(tokens), width):
            end = min(start + width, len(tokens))
            targets = labels[start:end]
            if not np.any(targets != IGNORE_INDEX):
                continue
            internal_ends = boundaries[(boundaries > start) & (boundaries < end)] - start
            local_boundaries = np.concatenate(([0], internal_ends, [end - start]))
            windows.append({
                "input_ids": torch.tensor(tokens[start:end], dtype=torch.long),
                "labels": torch.tensor(targets, dtype=torch.long),
                "cu_seq_lens": torch.tensor(local_boundaries, dtype=torch.int32),
            })
        return windows
