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

"""Explicit CANN 8.5 FP32 reduction schedules for the archived JT recipe.

The archived MoE routing-weight derivative, cross-entropy derivative, Muon NS
norms and gradient-clipping norm all reuse these schedules. They are
reproducibility operations, not more accurate replacements for native
reductions. Only small partial sums cross the device/host boundary. All inputs
are live tensors; there are no recorded norms, coordinates or training steps.
"""

# This explicitly selected recipe is Torch/NPU-only.
# pylint: disable=forbidden-backend-import

import numpy as np
import torch


def half_tree(values: torch.Tensor) -> torch.Tensor:
    """Reduce the last axis by halves, folding an odd tail into the first lane."""
    work = values
    while work.shape[-1] > 1:
        size = work.shape[-1]
        half = size // 2
        reduced = work[..., :half] + work[..., half:2 * half]
        if size % 2:
            reduced[..., 0] = reduced[..., 0] + work[..., -1]
        work = reduced
    return work[..., 0]


def adjacent_tree(values: torch.Tensor) -> torch.Tensor:
    """Reduce adjacent pairs, padding the final axis to a power of two."""
    width = values.shape[-1]
    padded_width = 1 << (width - 1).bit_length()
    if padded_width != width:
        padded = values.new_zeros((*values.shape[:-1], padded_width))
        padded[..., :width] = values
        values = padded
    while values.shape[-1] > 1:
        values = values[..., ::2] + values[..., 1::2]
    return values[..., 0]


def sequential_sum(values: np.ndarray) -> np.float32:
    """Preserve one FP32 rounding after each scalar accumulation."""
    total = np.float32(0)
    for value in values:
        total = np.float32(total + value)
    return total


def core_sum(partials: np.ndarray, cores: int = 40) -> np.float32:
    """Accumulate tiles on strided vector cores, then accumulate cores in order."""
    accumulators = np.zeros(cores, dtype=np.float32)
    for index, partial in enumerate(partials):
        core = index % cores
        accumulators[core] = np.float32(accumulators[core] + partial)
    return sequential_sum(accumulators)


def tiled_square_sum(value: torch.Tensor, tile: int, width: int) -> np.float32:
    """Square FP32 values and execute a bounded-memory row/lane/core schedule."""
    flat = value.reshape(-1)
    partials = []
    for start in range(0, flat.numel(), tile * 512):
        piece = flat[start:start + tile * 512]
        tile_count = (piece.numel() + tile - 1) // tile
        padded = piece.new_zeros(tile_count * tile)
        padded[:piece.numel()] = piece * piece
        lanes = half_tree(padded.reshape(tile_count, tile // width, width).transpose(-1, -2))
        partials.append(lanes.sum(dim=-1).cpu().numpy())
    parts = np.concatenate(partials)
    return parts[0] if len(parts) == 1 else core_sum(parts)


def gradient_square_sum(value: torch.Tensor) -> np.float32:
    """Select the archived SumX tiling from the current tensor's element count."""
    count = value.numel()
    tile, width = (count, 64) if count <= 5120 else ((1024, 64) if count == 40960 else (8144, 16))
    return tiled_square_sum(value, tile, width)


def weighted_sum(product: torch.Tensor) -> torch.Tensor:
    """Sum a BF16 weighted derivative: FP32 80-row half tree, then a native 64-lane sum.

    issue #252's NPU single-factor ablation (arm R10: 200 steps from a shared checkpoint,
    same recipe otherwise) found the archived 64-lane adjacent-pair tree bit-identical to a
    native FP32 sum over the same axis on that backend -- swapping just the lane reduction
    left the training trajectory unchanged (first_diff: none vs. the unmodified recipe).
    The 80-row half tree is not interchangeable the same way: swapping it alone (arm R9,
    same setup) left a late, weak divergence (first difference at step 479 of a 301-500
    window, 1 of 200 steps over the 5e-3 threshold), so it stays archived here.
    """
    rows = product.float().reshape(-1, 80, 64).transpose(-1, -2)
    return half_tree(rows).sum(dim=-1).reshape(*product.shape[:-1], 1)


__all__ = [
    "adjacent_tree", "core_sum", "gradient_square_sum", "half_tree", "sequential_sum", "tiled_square_sum",
    "weighted_sum",
]
