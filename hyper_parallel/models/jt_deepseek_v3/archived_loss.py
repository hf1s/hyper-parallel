# Copyright 2026 Huawei Technologies Co., Ltd.
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

"""Archived MF cross-entropy for JT's fixed 4096 x 152064 loss chunks.

``archived_cross_entropy`` accepts the keywords with which
``projected_cross_entropy`` calls its ``loss_fn``. An explicitly selected
archive model passes it there instead of the framework ``loss_function``;
the default JT model never selects it.

The loss value is native sum-reduced ``F.cross_entropy`` divided by the
caller's valid-token count, as in the default path. Only the derivative
differs: softmax is recomputed from the saved FP32 logits with the archived
compiled graph's FP32 accumulation order over the vocabulary. That order is a
reproducibility rule, not a more accurate reduction, and it is defined only
for vocabulary 152064 and 4096 rows.
"""

# This explicitly selected recipe is Torch/NPU-only.
# pylint: disable=forbidden-backend-import

from typing import Any

import torch
from torch.nn import functional as F

from hyper_parallel.models.jt_deepseek_v3.archived_reductions import half_tree


def tiled_sum(exponent: torch.Tensor) -> torch.Tensor:
    """Sum FP32 rows of width 152064 in the archived order; every add stays FP32.

    Each row is split into 22 tiles of 108 x 64 values. A tile is reduced by a
    halving tree over its 108 entries (entry ``j`` meets ``j + count // 2``; an
    odd remainder is added to the first partial sum), then by a native FP32 sum
    over its 64 lanes (issue #252 arm D2 found the archived adjacent-pair lane
    tree numerically inert here).
    Rows in complete 40-row blocks add the 22 tile sums left to right; the
    remaining rows sum even and odd tiles separately and then add the two
    partial sums.

    Args:
        exponent: FP32 values of shape ``[rows, 152064]``.

    Returns:
        FP32 row sums of shape ``[rows, 1]``.
    """
    tiles = exponent.reshape(exponent.shape[0], 22, 108, 64).transpose(-1, -2)
    values = half_tree(tiles).sum(dim=-1).unsqueeze(-1)
    total = values[:, 0, :]
    for tile in range(1, 22):
        total = total + values[:, tile, :]
    regular = values.shape[0] // 40 * 40
    tail = values[regular:]
    even, odd = tail[:, 0, :], tail[:, 1, :]
    for tile in range(2, 22, 2):
        even = even + tail[:, tile, :]
        odd = odd + tail[:, tile + 1, :]
    return torch.cat((total[:regular], even + odd), dim=0)


def log_probabilities(inputs: torch.Tensor) -> torch.Tensor:
    """Return log-softmax rows in max, sub, exp, ``tiled_sum``, log, sub order."""
    shifted = inputs - inputs.max(dim=-1, keepdim=True).values
    total = tiled_sum(shifted.exp())
    return shifted - total.log()


class TreeBackwardCE(torch.autograd.Function):
    """Native sum-reduced CE forward; softmax derivative recomputed in the archived order."""

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, targets: torch.Tensor, ignore_index: int) -> torch.Tensor:
        """Return native sum-reduced CE while saving inputs for the explicit derivative."""
        ctx.save_for_backward(inputs, targets)
        ctx.ignore_index = ignore_index
        return F.cross_entropy(inputs, targets, reduction='sum', ignore_index=ignore_index)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Recompute normalized probabilities from live logits and masked targets."""
        inputs, targets = ctx.saved_tensors
        result = log_probabilities(inputs).exp()
        valid = targets != ctx.ignore_index
        hot = torch.zeros_like(result).scatter_(1, torch.where(valid, targets, 0)[:, None], 1)
        result = (result - hot) * (valid[:, None] * gradient)
        return result, None, None


def archived_cross_entropy(*, logits: torch.Tensor, labels: Any, vocab_size: int,
                           shift_labels: torch.Tensor, num_items_in_batch: torch.Tensor,
                           ignore_index: int = -100) -> torch.Tensor:
    """Average the sum-reduced archived CE over the caller's valid-token denominator.

    Args:
        logits: Full-vocabulary local logits.
        labels: Unused; targets arrive pre-shifted in ``shift_labels``. Accepted so the signature matches
            the ``projected_cross_entropy`` ``loss_fn`` contract.
        vocab_size: Global vocabulary size of the local logits.
        shift_labels: Pre-shifted targets aligned with ``logits``.
        num_items_in_batch: Valid-target denominator chosen by the caller; ``projected_cross_entropy``
            passes the same whole-sequence value to every chunk.
        ignore_index: Target value excluded from the loss and its derivative.

    Returns:
        FP32 CE sum divided by ``num_items_in_batch``.
    """
    inputs = logits.float().reshape(-1, vocab_size)
    targets = shift_labels.reshape(-1).to(inputs.device)
    return TreeBackwardCE.apply(inputs, targets, ignore_index) / num_items_in_batch
