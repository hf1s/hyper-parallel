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
"""SwiGLU with explicit activation precision and optional projection clamping."""

from __future__ import annotations

import math

import torch  # pylint: disable=forbidden-backend-import
import torch.nn.functional as F  # pylint: disable=forbidden-backend-import


def _validate_options(clamp_limit: float | None, compute_dtype: torch.dtype | None) -> None:
    """Reject invalid activation policies at the public boundary."""
    if clamp_limit is not None and (not math.isfinite(clamp_limit) or clamp_limit <= 0):
        raise ValueError("SwiGLU clamp_limit must be finite and positive, or None")
    if compute_dtype is not None and (
            not isinstance(compute_dtype, torch.dtype) or not compute_dtype.is_floating_point):
        raise ValueError("SwiGLU compute_dtype must be a floating-point torch.dtype, or None")


def _activate(gate: torch.Tensor, up: torch.Tensor, dim: int,
              clamp_limit: float | None, use_fused: bool) -> torch.Tensor:
    """Apply the asymmetric gate clamp and symmetric up clamp before SwiGLU."""
    if clamp_limit is not None:
        gate = gate.clamp(max=clamp_limit)
        up = up.clamp(min=-clamp_limit, max=clamp_limit)
    if use_fused and gate.device.type == "npu":
        # Optional backend: CPU-only model construction must not import its runtime.
        import torch_npu  # pylint: disable=import-outside-toplevel

        return torch_npu.npu_swiglu(torch.cat((gate, up), dim=dim), dim=dim)
    return F.silu(gate) * up


def swiglu_from_projections(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    clamp_limit: float | None = None,
    compute_dtype: torch.dtype | None = None,
    use_fused: bool = True,
) -> torch.Tensor:
    """Apply SwiGLU to separate projections and return their original dtype.

    Args:
        gate: Gate projection output.
        up: Up projection output with the same shape, device and dtype.
        clamp_limit: Upper bound for gate and absolute bound for up; None disables clamping.
        compute_dtype: Activation calculation dtype; None retains the projection dtype.
        use_fused: Whether an available NPU fused activation may be used.

    Returns:
        ``silu(min(gate, limit)) * clamp(up, -limit, limit)`` in the projection dtype,
        with each clamp omitted when no limit is selected.
    """
    _validate_options(clamp_limit, compute_dtype)
    if gate.shape != up.shape or gate.dtype != up.dtype or gate.device != up.device:
        raise ValueError("SwiGLU gate and up must have matching shape, dtype and device")
    dtype = gate.dtype
    if compute_dtype is not None:
        gate, up = gate.to(compute_dtype), up.to(compute_dtype)
    return _activate(gate, up, -1, clamp_limit, use_fused).to(dtype)


def swiglu(
    x: torch.Tensor,
    dim: int = -1,
    *,
    clamp_limit: float | None = None,
    compute_dtype: torch.dtype | None = None,
    use_fused: bool = True,
) -> torch.Tensor:
    """Apply SwiGLU to packed gate/up projections with an explicit activation policy.

    Args:
        x: Packed gate then up projections.
        dim: Dimension along which the projections are split.
        clamp_limit: Optional gate upper bound and symmetric up bound.
        compute_dtype: Optional calculation dtype, independent of the returned input dtype.
        use_fused: Whether to use the NPU fused activation; False preserves the native formula.

    Returns:
        SwiGLU output in the input dtype. The default NPU path uses the existing fused kernel.
    """
    _validate_options(clamp_limit, compute_dtype)
    if use_fused and clamp_limit is None and compute_dtype is None and x.device.type == "npu":
        # Keep the default fused path free of additional projection buffers.
        import torch_npu  # pylint: disable=import-outside-toplevel

        return torch_npu.npu_swiglu(x, dim=dim)
    dtype = x.dtype
    typed = x.to(compute_dtype) if compute_dtype is not None else x
    if typed.shape[dim] % 2:
        raise ValueError("SwiGLU packed projection dimension must be even")
    gate, up = typed.chunk(2, dim=dim)
    return _activate(gate, up, dim, clamp_limit, use_fused).to(dtype)
