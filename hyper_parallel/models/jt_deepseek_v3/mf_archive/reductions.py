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

These are reproducibility operations, not more accurate replacements for native
reductions. Only small partial sums cross the device/host boundary. All inputs
are live tensors; there are no recorded norms, coordinates or training steps.
"""

# This explicitly selected recipe is Torch/NPU-only.
# pylint: disable=forbidden-backend-import

import numpy as np
import torch


def half_tree(values: torch.Tensor) -> torch.Tensor:
    """Reduce the last axis by halves, folding an odd tail into the first lane."""
    if values.ndim == 0 or values.shape[-1] == 0:
        raise ValueError("A reduction requires a nonempty final dimension")
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
    if width <= 0:
        raise ValueError("A reduction requires at least one lane")
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
    if cores <= 0:
        raise ValueError("The core count must be positive")
    accumulators = np.zeros(cores, dtype=np.float32)
    for index, partial in enumerate(partials):
        core = index % cores
        accumulators[core] = np.float32(accumulators[core] + partial)
    return sequential_sum(accumulators)


def tiled_square_sum(value: torch.Tensor, tile: int, width: int) -> np.float32:
    """Square FP32 values and execute a bounded-memory row/lane/core schedule."""
    if value.dtype != torch.float32 or tile <= 0 or width <= 0 or tile % width:
        raise ValueError("Expected FP32 values and a positive tile divisible by its SIMD width")
    flat = value.reshape(-1)
    if not flat.numel():
        raise ValueError("Cannot reduce an empty tensor")
    partials = []
    for start in range(0, flat.numel(), tile * 512):
        piece = flat[start:start + tile * 512]
        tile_count = (piece.numel() + tile - 1) // tile
        padded = piece.new_zeros(tile_count * tile)
        padded[:piece.numel()] = piece * piece
        lanes = half_tree(padded.reshape(tile_count, tile // width, width).transpose(-1, -2))
        partials.append(adjacent_tree(lanes).cpu().numpy())
    parts = np.concatenate(partials)
    return parts[0] if len(parts) == 1 else core_sum(parts)


def gradient_square_sum(value: torch.Tensor) -> np.float32:
    """Select the archived SumX tiling from the current tensor's element count."""
    count = value.numel()
    tile, width = (count, 64) if count <= 5120 else ((1024, 64) if count == 40960 else (8144, 16))
    return tiled_square_sum(value, tile, width)


def weighted_sum(product: torch.Tensor) -> torch.Tensor:
    """Sum a BF16 weighted derivative with FP32 80-row/64-lane arithmetic."""
    if product.shape[-1] != 5120 or product.dtype != torch.bfloat16:
        raise ValueError("The archived weighted reduction requires BF16 hidden width 5120")
    rows = product.float().reshape(-1, 80, 64).transpose(-1, -2)
    return adjacent_tree(half_tree(rows)).reshape(*product.shape[:-1], 1)


def normalize_ns(value: torch.Tensor) -> torch.Tensor:
    """Return the archived BF16 NS input, including its documented cast boundaries."""
    shape = tuple(value.shape)
    current = value.mT if value.shape[-2] > value.shape[-1] else value
    epsilon = 1.0011717677116394e-7
    if value.ndim == 3:
        if value.dtype != torch.bfloat16:
            raise ValueError("Archived expert NS requires BF16 input")
        if shape == (8, 5120, 1536):
            rows = current.float().square().reshape(8, 768, 160, 64).transpose(-1, -2)
            partials = adjacent_tree(half_tree(rows)).cpu().numpy()
            totals = np.asarray([core_sum(row, 5) for row in partials], dtype=np.float32)
            norm = torch.from_numpy(totals).to(value.device).sqrt().reshape(8, 1, 1)
        else:
            # The compiled archive adds epsilon before its one BF16 conversion;
            # rounding the norm earlier changes values near a BF16 midpoint.
            norm = current.float().norm(dim=(-2, -1), keepdim=True)
        return (current / (norm + 1e-7).bfloat16()).bfloat16()
    if value.ndim != 2 or value.dtype != torch.float32:
        raise ValueError("Archived matrix NS requires FP32 two-dimensional input")
    if shape in ((1536, 5120), (5120, 16384), (5120, 10240), (5120, 12288)):
        total = tiled_square_sum(current, 6984, 24)
        norm = torch.tensor(total, device=value.device).sqrt()
    elif shape in ((16384, 1536), (8192, 1536), (12288, 5120), (5120, 3072)):
        total = tiled_square_sum(current, 9776, 16)
        norm = torch.tensor(total, device=value.device).sqrt()
    elif shape == (8, 5120):
        norm = torch.tensor(gradient_square_sum(current), device=value.device).sqrt()
        epsilon = 1e-7
    else:
        norm = current.norm(dim=(-2, -1), keepdim=True)
    return (current / (norm + epsilon)).bfloat16()
