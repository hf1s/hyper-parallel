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

from collections.abc import Callable, Sequence

# AutoModels loss components implement the Transformers/PyTorch Trainer API.
# pylint: disable-next=forbidden-backend-import
import torch
# pylint: disable-next=forbidden-backend-import
import torch.nn.functional as F

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.components.modules.mtp import shift_mtp_sequence


def calculate_mtp_loss(
    mtp_per_depth_logits: Sequence[torch.Tensor],
    shift_labels: torch.Tensor,
    loss_fn: Callable[..., torch.Tensor],
    *,
    vocab_size: int,
    loss_factor: float = 1.0,
    ignore_index: int = IGNORE_INDEX,
    sequence_ends: tuple[int, ...] | None = None,
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
            signature, such as a model's ``loss_function`` (``ForCausalLMLoss``,
            or ``causal_lm_loss_parallel`` under loss parallelism).
        vocab_size: Global vocabulary size.
        loss_factor: Total MTP weight, divided equally across depths.
        ignore_index: Target value excluded from every depth.
        sequence_ends: Optional exclusive document ends, validated by the model.

    Returns:
        The weighted 0-d MTP loss; zero when no depth is given.

    Raises:
        ValueError: If a depth's logits are not aligned with ``shift_labels``.
    """
    total = torch.zeros((), device=shift_labels.device, dtype=torch.float32)
    depths = len(mtp_per_depth_logits)
    packed_targets = shift_labels
    for depth, logits in enumerate(mtp_per_depth_logits, start=1):
        if logits.shape[:-1] != shift_labels.shape:
            raise ValueError("MTP logits must align with shift_labels position by position")
        if sequence_ends is None:
            targets = F.pad(shift_labels[..., depth:], (0, depth), value=ignore_index)
        else:
            packed_targets = shift_mtp_sequence(packed_targets, sequence_ends=sequence_ends, pad_value=ignore_index)
            targets = packed_targets
        loss_kwargs = {}
        if sequence_ends is not None:
            # Short documents may have no valid target at deeper MTP depths.
            loss_kwargs["num_items_in_batch"] = (targets != ignore_index).sum().clamp_min(1)
        depth_loss = loss_fn(logits=logits, labels=None, vocab_size=vocab_size,
                             shift_labels=targets, **loss_kwargs)
        total = total + depth_loss.reshape(()) * (loss_factor / depths)
    return total


__all__ = ["calculate_mtp_loss"]
