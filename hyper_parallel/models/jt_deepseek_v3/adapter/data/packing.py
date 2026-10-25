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
"""JT packed document positions and original-sample routing groups."""

from __future__ import annotations

from numbers import Integral

import torch


class JTPackedSequence:
    """Materialize linear-size metadata for one local interval of a packed batch.

    ``sample_ranges`` contains real token intervals, excluding alignment tails.
    Documents control attention and future targets; samples control sequence aux.
    Neither prompt masking nor token IDs define padding.
    """

    def __init__(self, input_ids: torch.Tensor, ends: tuple[int, ...], *,
                 sample_ranges: tuple[tuple[int, int], ...] | None = None,
                 sequence_start: int = 0) -> None:
        """Validate global metadata and build local positions, tails and sample IDs."""
        batch, length = input_ids.shape
        if (not ends or any(isinstance(end, bool) or not isinstance(end, Integral) for end in ends)
                or ends[0] <= 0 or any(left >= right for left, right in zip(ends, ends[1:]))
                or sequence_start < 0 or sequence_start + length > ends[-1]):
            raise ValueError("JT cumulative document ends must increase and cover the local token interval")
        if batch > 1:
            if sequence_start != 0 or ends[-1] != input_ids.numel():
                raise ValueError("JT packed rows require cp_size == 1 and flattened batch boundaries")
            if any(row * length not in ends for row in range(1, batch)):
                raise ValueError("JT cumulative document ends must include every row end to isolate batch rows")
        if sample_ranges is None:
            sample_ranges = tuple((row * length, (row + 1) * length) for row in range(batch))
            if batch == 1:
                sample_ranges = ((0, ends[-1]),)
        if (not sample_ranges
                or any(len(interval) != 2 or any(isinstance(value, bool) or not isinstance(value, Integral)
                                                 for value in interval) for interval in sample_ranges)
                or any(start < 0 or start >= end or end > ends[-1] for start, end in sample_ranges)
                or any(left[1] > right[0] for left, right in zip(sample_ranges, sample_ranges[1:]))):
            raise ValueError("JT sample_ranges must be ordered, nonempty, non-overlapping real token intervals")
        boundaries = {0, *ends}
        if any(start not in boundaries or end not in boundaries for start, end in sample_ranges):
            raise ValueError("JT sample_ranges must align with document boundaries")
        self.sample_ranges = sample_ranges
        self.num_samples = len(sample_ranges)
        indices = torch.arange(input_ids.numel(), device=input_ids.device).reshape_as(input_ids) + sequence_start
        end_tensor = input_ids.new_tensor(ends)
        documents = torch.bucketize(indices, end_tensor, right=True)
        starts = input_ids.new_tensor((0, *ends[:-1]))
        self.position_ids = indices - starts[documents]
        self.sequence_end_mask = indices == end_tensor[documents] - 1
        ranges = input_ids.new_tensor(sample_ranges)
        sample_ids = torch.bucketize(indices, ranges[:, 1].contiguous(), right=True)
        safe_ids = sample_ids.clamp_max(self.num_samples - 1)
        real = (sample_ids < self.num_samples) & (indices >= ranges[:, 0][safe_ids])
        self.sample_ids = safe_ids.masked_fill(~real, -1)
        self.token_mask = real


__all__ = ["JTPackedSequence"]
