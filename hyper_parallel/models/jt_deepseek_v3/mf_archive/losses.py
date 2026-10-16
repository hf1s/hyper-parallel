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

"""Archived CE backward arithmetic with the current Trainer loss normalization."""

# This explicitly selected recipe is Torch/NPU-only.
# pylint: disable=forbidden-backend-import

from typing import Any

import torch
from torch.nn import functional as F


def tiled_sum(exponent: torch.Tensor) -> torch.Tensor:
    """Use the audited 22x6912, 108x64 tree; every add remains FP32 on the same device."""
    if exponent.ndim != 2 or exponent.shape[-1] != 152064 or exponent.dtype != torch.float32:
        raise ValueError('Measured DVM compatibility tree requires FP32 vocabulary152064')
    values = exponent.reshape(exponent.shape[0], 22, 108, 64)
    while values.shape[-2] > 1:
        count = values.shape[-2]
        half = count // 2
        result = values[..., :half, :] + values[..., half:half * 2, :]
        if count % 2:
            first = result[..., :1, :] + values[..., -1:, :]
            result = torch.cat((first, result[..., 1:, :]), dim=-2) if half > 1 else first
        values = result
    values = values.squeeze(-2)
    while values.shape[-1] > 1:
        values = values[..., 0::2] + values[..., 1::2]
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
    """Retain max/sub/exp/sum/log/sub order, with no reference-stage values."""
    shifted = inputs - inputs.max(dim=-1, keepdim=True).values
    total = tiled_sum(shifted.exp())
    return shifted - total.log()


class TreeBackwardCE(torch.autograd.Function):
    """Keep original forward loss and compute the derivative using independent graph arithmetic."""

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
        hot = torch.zeros_like(result).scatter_(1, targets.clamp_min(0)[:, None], 1)
        result = (result - hot) * (valid[:, None] * gradient)
        return result, None, None

def archived_cross_entropy(*, logits: torch.Tensor, labels: Any, vocab_size: int,
                           shift_labels: torch.Tensor, num_items_in_batch: torch.Tensor,
                           ignore_index: int = -100) -> torch.Tensor:
    """Average the sum-reduced archived CE over this chunk's valid-token denominator."""
    if labels is not None or logits.shape[-1] != vocab_size or vocab_size != 152064:
        raise ValueError("The archived CE expects pre-shifted targets and vocabulary 152064")
    inputs = logits.float().reshape(-1, vocab_size)
    if inputs.shape[0] != 4096:
        raise ValueError("The archived CE dispatch schedule is defined only for 4096 rows")
    targets = shift_labels.reshape(-1).to(inputs.device)
    return TreeBackwardCE.apply(inputs, targets, ignore_index) / num_items_in_batch
