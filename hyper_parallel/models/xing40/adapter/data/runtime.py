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
"""Carry the packed document boundaries into the Xing4.0 forward.

``TextParallelBatch`` resolves the global cumulative document ends of one micro-batch and
keeps them as ``cu_seq_lens``, but its model inputs are only ``input_ids``, ``labels``,
``position_ids`` and ``shift_labels`` plus whatever a runtime adapter contributes
(``data/batching/get_batch.py``, ``_split_model_and_loss_inputs``). This adapter is that
contribution: it publishes the boundaries as ``actual_seq_len``, the name the fused TND
attention of ``components/functional/npu_fusion_attention.py`` and
``modules.MLAAttention.forward`` already read, so that attention is block-diagonal causal
over the documents of the pack instead of causal over the whole flat sequence.

The MindFormers reference reaches the same kernel arguments by a different route: its
dataset emits one fixed-length ``actual_seq_len`` column per sample, tail-padded with the
sequence length to ``compressed_eod_mask_length``
(``mindformers/dataset/blended_datasets/gpt_dataset.py``, ``_get_eod_attention_mask``), and
``asl_batch_wrapper`` then offsets each row by the previous row's end so the column is
cumulative over the flattened micro-batch
(``mindformers/dataset/causal_language_model_dataset.py``). The tail padding is only there
because a MindSpore dataset column has a fixed shape; a repeated end is a zero-length
segment the kernel ignores. Nothing here needs a fixed shape, and
``npu_fusion_attention._length_list`` rejects a non-increasing list outright, so the
boundaries are published exactly, without padding.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hyper_parallel.data.batching.runtime_input import RuntimeInputAdapter, RuntimeInputContext

__all__ = ["Xing40PackedRuntime"]


class Xing40PackedRuntime(RuntimeInputAdapter):
    """Publish the pack's document ends as ``actual_seq_len``.

    ``cu_seq_lens`` is the global, leading-zero cumulative end offset of every document in
    the micro-batch, counted over the flattened ``[batch, sequence]`` tokens: the layout the
    fused attention sees after it reshapes to TND. Position IDs, shifted labels and loss
    masks stay owned by ``TextParallelBatch``.
    """

    def runtime_input_fields(self) -> tuple[str, ...]:
        """Declare the one forward field this adapter owns."""
        return ("actual_seq_len", "loss_mask")

    def build_runtime_inputs(
            self,
            *,
            batch: Mapping[str, Any],
            context: RuntimeInputContext,
    ) -> Mapping[str, Any]:
        """Convert the batch's global boundaries into host-side cumulative ends.

        Args:
            batch: Parallel-local batch; ``cu_seq_lens`` stays global across CP ranks.
            context: Local geometry and parallel coordinates of this rank.

        Returns:
            ``{"actual_seq_len": (end, ...), "loss_mask": tensor}`` -- the cumulative document
            ends without the leading zero, as a tuple of Python integers, and the batch's loss
            mask. ``_split_model_and_loss_inputs`` routes ``loss_mask`` to the loss inputs only,
            so this is the one channel that carries it into the forward; the model scores the
            positions the Trainer weights by rather than every non-negative label.

        Raises:
            ValueError: If the batch carries no boundaries, or if they are not strictly
                increasing ends of the flattened micro-batch.
            NotImplementedError: If context parallelism is enabled, which would require
                re-segmenting the global boundaries onto this rank's sequence slice.
        """
        local_shape = tuple(context.local_input_shape)
        if len(local_shape) != 2:
            raise ValueError(
                f"Xing4.0 attention requires [batch, sequence] input_ids, got shape {local_shape}"
            )
        if context.parallel_sizes.get("cp", 1) != 1:
            raise NotImplementedError(
                "Xing4.0 document-boundary attention is not implemented for context parallelism: "
                "the global document ends would have to be re-segmented onto each CP sequence "
                "slice before the fused attention sees them"
            )

        cu_seq_lens = batch.get("cu_seq_lens")
        if cu_seq_lens is None:
            raise ValueError(
                "Xing4.0 document-boundary attention requires cu_seq_lens: the batch carries no "
                "document ends, so attention would cross the boundaries of the packed documents"
            )

        # One host transfer per micro-batch. The attention interface would otherwise convert the
        # device tensor once per layer, and this list is read by every one of them.
        boundaries = [int(boundary) for boundary in cu_seq_lens.reshape(-1).tolist()]
        token_count = local_shape[0] * local_shape[1]
        if len(boundaries) < 2 or boundaries[0] != 0:
            raise ValueError("cu_seq_lens must contain a leading zero and at least one document")
        if any(end <= start for start, end in zip(boundaries[:-1], boundaries[1:])):
            raise ValueError(f"cu_seq_lens must be strictly increasing, got {boundaries}")
        if boundaries[-1] != token_count:
            raise ValueError(
                "cu_seq_lens must end at the flattened micro-batch token count "
                f"({token_count}), but got {boundaries[-1]}"
            )

        published: dict[str, Any] = {"actual_seq_len": tuple(boundaries[1:])}
        loss_mask = batch.get("loss_mask")
        if loss_mask is not None:
            published["loss_mask"] = loss_mask
        return published
