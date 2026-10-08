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
"""Checkpoint output projection and causal CE together, including TP boundaries."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.tensor_parallel.loss_parallel import loss_parallel


def projected_cross_entropy(
    hidden_states: torch.Tensor,
    targets: torch.Tensor,
    *,
    head: Callable[[torch.Tensor], torch.Tensor],
    loss_fn: Callable[..., torch.Tensor],
    vocab_size: int,
    chunk_size: int,
    sequence_parallel_size: int = 1,
    tp_mesh: Any = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return token-mean CE without materializing or retaining full-sequence logits.

    The ordinary head and loss keep ownership of vocabulary sharding and gradient
    communication. With sequence parallelism, the head gathers rank-local chunks
    in rank order; targets are rearranged into exactly that same order. One global
    valid-target denominator is used for every chunk, including empty chunks.

    Args:
        hidden_states: Local hidden activations in [batch, local_sequence, hidden].
        targets: Pre-shifted labels in [batch, global_sequence].
        head: Position-wise output projection, including its existing parallel boundary wrappers.
        loss_fn: Transformers-compatible CE accepting num_items_in_batch.
        vocab_size: Global vocabulary size.
        chunk_size: Maximum global token positions per row projected at once;
            must be divisible by sequence_parallel_size.
        sequence_parallel_size: Number of contiguous equal sequence shards.
        tp_mesh: Loss-parallel mesh, re-entered during checkpoint recomputation.
        ignore_index: Label value excluded from both loss and token count.

    Returns:
        Scalar FP32 token mean, or differentiable zero when all labels are ignored.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if (isinstance(sequence_parallel_size, bool) or not isinstance(sequence_parallel_size, int)
            or sequence_parallel_size <= 0 or chunk_size % sequence_parallel_size):
        raise ValueError("chunk_size must be divisible by a positive sequence_parallel_size")
    if (hidden_states.ndim != 3 or targets.ndim != 2 or hidden_states.shape[0] != targets.shape[0]
            or hidden_states.shape[1] == 0
            or hidden_states.shape[1] * sequence_parallel_size != targets.shape[1]):
        raise ValueError("Hidden sequence shards must evenly partition aligned global targets")
    denominator = (targets != ignore_index).sum().clamp_min(1)
    local_length = hidden_states.shape[1]
    local_chunk_size = chunk_size // sequence_parallel_size

    def project_and_score(states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        context = nullcontext() if tp_mesh is None else loss_parallel(mesh=tp_mesh)
        with context:
            logits = head(states)
            if logits.shape[:-1] != labels.shape:
                raise ValueError("Output head must gather sequence shards in rank order")
            return loss_fn(logits=logits, labels=None, vocab_size=vocab_size, shift_labels=labels,
                           num_items_in_batch=denominator, ignore_index=ignore_index).reshape(())

    total = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
    for start in range(0, local_length, local_chunk_size):
        stop = min(start + local_chunk_size, local_length)
        states = hidden_states[:, start:stop]
        labels = torch.cat([targets[:, rank * local_length + start:rank * local_length + stop]
                            for rank in range(sequence_parallel_size)], dim=1)
        total = total + checkpoint(project_and_score, states, labels, use_reentrant=False)
    return total
