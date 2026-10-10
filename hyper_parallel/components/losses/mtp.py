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
"""Multi-Token-Prediction auxiliary loss objective."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

# AutoModels loss components implement the Transformers/PyTorch Trainer API.
# pylint: disable-next=forbidden-backend-import
import torch
# pylint: disable-next=forbidden-backend-import
import torch.nn.functional as F

from hyper_parallel.data.constants import IGNORE_INDEX


def iter_mtp_targets(
    shift_labels: torch.Tensor,
    depths: int,
    *,
    ignore_index: int = IGNORE_INDEX,
    shift_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    sequence_end_mask: torch.Tensor | None = None,
) -> Iterator[torch.Tensor]:
    """Yield future targets using the same document and partition boundaries.

    Args:
        shift_labels: Pre-shifted main LM labels.
        depths: Number of future prediction depths.
        ignore_index: Padding label excluded from the loss.
        shift_fn: One-token shift including partition halos, if needed.
        sequence_end_mask: Local document tails where future targets are ignored.
    """
    if sequence_end_mask is not None and sequence_end_mask.shape != shift_labels.shape:
        raise ValueError("MTP document-tail mask must match shift_labels")
    targets = shift_labels
    for _ in range(depths):
        targets = (F.pad(targets[..., 1:], (0, 1), value=ignore_index)
                   if shift_fn is None else shift_fn(targets))
        if sequence_end_mask is not None:
            targets = targets.masked_fill(sequence_end_mask, ignore_index)
        yield targets


def calculate_mtp_loss(
    mtp_per_depth_logits: Sequence[torch.Tensor],
    shift_labels: torch.Tensor,
    loss_fn: Callable[..., torch.Tensor],
    *,
    vocab_size: int,
    loss_factor: float = 1.0,
    ignore_index: int = IGNORE_INDEX,
    shift_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    sequence_end_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """DeepSeek-V3 Multi-Token-Prediction loss ``loss_factor / D * sum_k L_k``.

    Depth ``k`` (1-based) predicts the token ``k`` positions after the main
    next-token target, so its targets are ``shift_labels`` shifted left by
    ``k`` and padded with ``ignore_index`` without wrapping.

    Args:
        mtp_per_depth_logits: Logits of depths ``1..D``, each aligned position by
            position with ``shift_labels``. Vocabulary-sharded logits are accepted
            when ``loss_fn`` supports them.
        shift_labels: Main LM targets already shifted by one token, as produced
            by the shared text batch; ignored targets hold ``ignore_index``.
        loss_fn: Causal-LM loss with the Transformers ``loss_function``
            signature that accepts ``num_items_in_batch``, such as a model's
            ``loss_function`` (``ForCausalLMLoss``, or ``causal_lm_loss_parallel``
            under loss parallelism).
        vocab_size: Global vocabulary size.
        loss_factor: Total MTP weight, divided equally across depths.
        ignore_index: Target value excluded from every depth.
        shift_fn: One-token left shift, padding with ignore_index; may supply partition halos.
        sequence_end_mask: Document tails where future targets must be ignored.

    Returns:
        The weighted 0-d MTP loss; zero when no depth is given. A depth whose
        targets are all ``ignore_index`` contributes zero.

    Raises:
        ValueError: If a depth's logits are not aligned with ``shift_labels``.
    """
    total = torch.zeros((), device=shift_labels.device, dtype=torch.float32)
    depths = len(mtp_per_depth_logits)
    targets_by_depth = iter_mtp_targets(shift_labels, depths, ignore_index=ignore_index,
                                        shift_fn=shift_fn, sequence_end_mask=sequence_end_mask)
    for logits, targets in zip(mtp_per_depth_logits, targets_by_depth):
        if logits.shape[:-1] != shift_labels.shape:
            raise ValueError("MTP logits must align with shift_labels position by position")
        # Each depth averages over its own valid targets; one without any adds zero instead of 0/0.
        depth_loss = loss_fn(logits=logits, labels=None, vocab_size=vocab_size, shift_labels=targets,
                             num_items_in_batch=(targets != ignore_index).sum().clamp_min(1),
                             ignore_index=ignore_index)
        total = total + depth_loss.reshape(()) * (loss_factor / depths)
    return total


__all__ = ["calculate_mtp_loss"]
