# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Opt-in archived objective arithmetic for JT model integration.

This module contains only the packed-MTP target and absolute-position policy,
plus the branch-gradient join used by the archived decoder.  The native,
document-aware objective does not import or call these helpers.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch


class ArchiveBranchGradient(torch.autograd.Function):
    """Expose one live input to two branches and join their gradients in FP32.

    No forward value is saved: backward consumes only gradients produced by the
    live branch outputs.  The single cast back at the join preserves the
    archived residual/branch boundary and the input dtype.
    """

    @staticmethod
    def forward(ctx: object, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.input_dtype = value.dtype  # type: ignore[attr-defined]
        return value.view_as(value), value.float()

    @staticmethod
    def backward(ctx: object, first: torch.Tensor | None,
                 second: torch.Tensor | None) -> tuple[torch.Tensor | None]:
        if first is None:
            if second is None:
                return (None,)
            joined = second.float()
        elif second is None:
            joined = first.float()
        else:
            joined = first.float() + second.float()
        return (joined.to(ctx.input_dtype),)  # type: ignore[attr-defined]


def split_archive_branches(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return branch views whose backward joins gradients at the live input."""
    return ArchiveBranchGradient.apply(value)


def shift_packed_mtp(value: torch.Tensor, *, pad_value: int = 0) -> torch.Tensor:
    """Shift a complete packed sequence left, crossing document boundaries.

    ``value`` is expected to have batch and sequence dimensions.  The final
    global position is padded; no per-document reset or tail masking is done.
    This is deliberately separate from the default document-aware shift.
    """
    if value.ndim < 2:
        raise ValueError("packed MTP values must have batch and sequence dimensions")
    return torch.cat((value[:, 1:], torch.full_like(value[:, :1], pad_value)), dim=1)


def iter_packed_mtp_targets(shift_labels: torch.Tensor, depths: int,
                            *, ignore_index: int = -100) -> Iterator[torch.Tensor]:
    """Yield archived future targets, preserving cross-document shifts."""
    if shift_labels.ndim < 2:
        raise ValueError("packed MTP labels must have batch and sequence dimensions")
    if depths < 0:
        raise ValueError("MTP depth must be non-negative")
    targets = shift_labels
    for _ in range(depths):
        targets = shift_packed_mtp(targets, pad_value=ignore_index)
        yield targets


def absolute_positions(sequence_length: int, *, sequence_start: int = 0,
                       device: torch.device | str | None = None) -> torch.Tensor:
    """Return archive RoPE positions without resetting at packed documents."""
    if sequence_length < 0 or sequence_start < 0:
        raise ValueError("sequence length and start must be non-negative")
    return torch.arange(sequence_start, sequence_start + sequence_length, device=device).unsqueeze(0)


def archive_rope_embeddings(position_ids: torch.Tensor, *, rope_theta: float,
                            rope_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build FP32 RoPE tables using the archived NumPy inverse-frequency order."""
    if position_ids.ndim != 2:
        raise ValueError("position_ids must be [batch, sequence]")
    if rope_dim <= 0 or rope_dim % 2:
        raise ValueError("rope_dim must be a positive even number")
    inverse = 1.0 / (rope_theta ** (np.arange(0, rope_dim, 2, dtype=np.float32) / rope_dim))
    inverse_tensor = torch.from_numpy(inverse.astype(np.float32)).to(position_ids.device)
    frequency = position_ids.to(dtype=torch.float32).unsqueeze(-1) * inverse_tensor
    frequency = torch.cat((frequency, frequency), dim=-1)
    return frequency.cos(), frequency.sin()
