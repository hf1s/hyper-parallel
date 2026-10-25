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
"""Translate shared packed metadata to the JT forward contract."""

from collections.abc import Mapping
from typing import Any

import torch

from hyper_parallel.data.batching.runtime_input import RuntimeInputAdapter, RuntimeInputContext
from hyper_parallel.models.jt_deepseek_v3.adapter.data.packing import JTPackedSequence


class JTSequenceRuntime(RuntimeInputAdapter):
    """Pass the public batch's document boundaries to JT without constructing a dense mask."""

    def runtime_input_fields(self) -> tuple[str, ...]:
        """Declare the model-owned cumulative-length input."""
        return ("actual_seq_len", "sample_ranges")

    def build_runtime_inputs(self, *, batch: Mapping[str, Any],
                             context: RuntimeInputContext) -> Mapping[str, Any]:
        """Convert global leading-zero boundaries once at the data/model boundary.

        Args:
            batch: Local tokens and global cumulative document boundaries of
                the flattened rows.
            context: Local row shape and CP degree from the public data path.
        """
        batch_size, local_length = context.local_input_shape
        cp_size = context.parallel_sizes["cp"]
        if batch_size < 1 or local_length < 1 or cp_size < 1:
            raise ValueError("JT requires positive batch, sequence and CP sizes")
        if batch_size > 1 and cp_size > 1:
            # CP slices one global stream; rows would interleave non-contiguous
            # intervals, so the varlen metadata could no longer describe them.
            raise ValueError("JT packed rows require cp_size == 1")
        boundaries = batch.get("cu_seq_lens")
        if boundaries is None:
            if batch_size > 1:
                raise ValueError("JT packed rows require explicit cumulative document ends")
            return {}
        if boundaries.ndim != 1:
            raise ValueError("JT packed boundaries must be one-dimensional integers")
        values = tuple(boundaries.tolist())
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("JT packed boundaries must be one-dimensional integers")
        global_length = batch_size * local_length * cp_size
        if (len(values) < 2 or values[0] != 0
                or values[-1] != global_length
                or any(left >= right for left, right in zip(values, values[1:]))):
            raise ValueError(
                "JT packed boundaries must cover the flattened global batch: "
                f"expected end {global_length}, got {values[-1] if values else None}")
        row_ends = set(values)
        if batch_size > 1 and any(row * local_length not in row_ends for row in range(1, batch_size)):
            raise ValueError("JT packed boundaries must include every row end to isolate batch rows")
        runtime_inputs = {"actual_seq_len": values[1:]}
        sample_ranges = batch.get("sample_ranges")
        if sample_ranges is not None:
            sample_ranges = tuple(tuple(interval) for interval in sample_ranges.tolist())
            runtime_inputs["sample_ranges"] = sample_ranges
        config = context.options.get("model_config")
        if config is None and "labels" in batch:
            raise ValueError("JTSequenceRuntime requires model_config for per-depth objective counts")
        if config is not None:
            if "labels" not in batch:
                raise ValueError("JT objective counts require pre-shifted labels")
            sequence = JTPackedSequence(batch["input_ids"], values[1:], sample_ranges=sample_ranges,
                                        sequence_start=context.parallel_ranks["cp"] * local_length)
            valid = batch["labels"].ge(0) & batch["loss_mask"].bool() & sequence.token_mask
            counts = {"foundation_loss/lm": valid.sum(),
                      "foundation_loss/aux": torch.tensor(sequence.num_samples / cp_size,
                                                           device=batch["input_ids"].device)}
            # Count targets on their owning rank, rather than shifted hidden positions.
            # Summing CP shards gives each depth's denominator without exchanging labels.
            for depth in range(1, config.num_nextn_predict_layers + 1):
                counts[f"foundation_loss/mtp_{depth}"] = (valid & sequence.position_ids.ge(depth)).sum()
            runtime_inputs["loss_token_counts"] = counts
        return runtime_inputs
