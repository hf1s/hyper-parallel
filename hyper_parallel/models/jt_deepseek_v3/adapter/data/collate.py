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
"""Mark trailing padding for JT Attention without changing physical batch shapes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from hyper_parallel.data.batching.build_collate_fn import TextPackingCollator, build_online_text_collate_fn
from hyper_parallel.data.constants import IGNORE_INDEX


class JTAttentionPaddingCollator:
    """Wrap shared collation with a real-token count for one prepacked record."""

    def __init__(self, collator: TextPackingCollator, padding_token_id: int | None = None) -> None:
        """Bind shared collation and the explicit dataset padding contract.

        Args:
            collator: Shared text collator owning sequence-parallel alignment.
            padding_token_id: Dataset-specific trailing pad ID. If absent, only
                padding added by the shared collator is excluded from Attention.
        """
        if padding_token_id is not None and (
                isinstance(padding_token_id, bool) or not isinstance(padding_token_id, int) or padding_token_id < 0):
            raise ValueError("padding_token_id must be a non-negative integer")
        self.collator = collator
        self.padding_token_id = padding_token_id

    def __call__(self, model_samples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Preserve tokens and labels while separating their trailing padding boundary.

        Args:
            model_samples: One prepacked record with explicit document boundaries.

        Returns:
            Aligned batch with physical boundaries and the global real-token count.
        """
        if len(model_samples) != 1:
            raise ValueError("JT Attention padding requires one prepacked record per micro-batch")
        batch = self.collator(model_samples)
        tokens = model_samples[0]["input_ids"]
        count = tokens.numel()
        if self.padding_token_id is not None:
            nonpadding = torch.nonzero(tokens != self.padding_token_id, as_tuple=False).flatten()
            count = int(nonpadding[-1]) + 1 if nonpadding.numel() else 0
        if not count:
            raise ValueError("JT Attention requires at least one non-padding token")
        if torch.any(batch["labels"][:, count:] != IGNORE_INDEX):
            raise ValueError("Trailing padding tokens must have ignored labels")
        boundaries = batch["cu_seq_lens"]
        physical_length = batch["input_ids"].shape[1]
        ends = [count, physical_length] if count < physical_length else [count]
        batch["cu_seq_lens"] = torch.cat((boundaries[boundaries < count], boundaries.new_tensor(ends)))
        batch["attention_token_count"] = boundaries.new_tensor([count], dtype=torch.int64)
        return batch


def build_attention_padding_collate_fn(
        mesh_context: Any = None, padding_token_id: int | None = None,
) -> JTAttentionPaddingCollator:
    """Build opt-in JT collation independently of data balancing.

    Args:
        mesh_context: Trainer topology used by the shared text collator.
        padding_token_id: Explicit trailing pad ID; ignored labels alone are not padding.

    Returns:
        Collator preserving physical tokens while marking the Attention domain.
    """
    return JTAttentionPaddingCollator(build_online_text_collate_fn(mesh_context), padding_token_id)
