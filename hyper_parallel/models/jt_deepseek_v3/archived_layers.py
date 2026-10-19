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

"""Opt-in archived norm, embedding, and output-head layers."""

# This explicitly selected recipe is Torch/NPU-only.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from typing import Any

import torch
import torch_npu
from torch import nn
from torch.nn import functional as F

from hyper_parallel.models.jt_deepseek_v3 import modeling_jt_deepseek_v3 as jt
from hyper_parallel.models.jt_deepseek_v3.legacy_gemm import head_weight_gradient

__all__ = ["ArchivedNorm", "ArchivedEmbedding", "ArchivedHead"]


class NormResidual(torch.autograd.Function):
    """Round once after joining residual and RMSNorm input gradients."""

    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, weight: torch.Tensor,
                epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized activations and the residual identity."""
        inputs = values.float()
        normalized, rstd = torch_npu.npu_rms_norm(inputs, weight.float(), epsilon=epsilon)
        ctx.save_for_backward(inputs, weight, rstd)
        ctx.input_dtype = values.dtype
        return normalized.to(values.dtype), values

    @staticmethod
    def backward(ctx: Any, norm_gradient: torch.Tensor,
                 residual_gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Join both input derivatives before rounding to the activation dtype."""
        inputs, weight, rstd = ctx.saved_tensors
        dx, dw = torch.ops.npu.npu_rms_norm_backward(
            norm_gradient.float(), inputs, weight.float(), rstd)
        return (dx + residual_gradient.float()).to(ctx.input_dtype), dw, None


class ArchivedNorm(jt.DeepseekV32RMSNorm):
    """FP32 affine storage and RMS arithmetic, with an explicit output boundary."""

    keep_compute_in_fp32 = True

    def __init__(self, hidden_size: int, eps: float, *, residual: bool = False,
                 output_dtype: torch.dtype | None = None) -> None:
        """Choose the residual gradient join and output cast boundary."""
        super().__init__(hidden_size, eps)
        self.residual = residual
        self.output_dtype = output_dtype

    def forward(self, hidden_states: torch.Tensor) -> Any:
        """Return the normalized tensor, or the normalization/residual pair."""
        if self.residual:
            return NormResidual.apply(hidden_states, self.weight, self.variance_epsilon)
        value = torch_npu.npu_rms_norm(
            hidden_states.float(), self.weight.float(), epsilon=self.variance_epsilon)[0]
        return value.to(self.output_dtype or hidden_states.dtype)


class ArchivedEmbedding(nn.Embedding):
    """Keep the shared main/MTP embedding parameter and gradient in FP32."""

    keep_compute_in_fp32 = True


class _HeadGradient(torch.autograd.Function):
    """Use the archived GEMM route for the output-head weight derivative."""

    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Project with the ordinary linear forward operation."""
        ctx.save_for_backward(values, weight)
        return F.linear(values, weight)  # pylint: disable=not-callable

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Use the archived NZ GEMM only for the weight derivative."""
        values, weight = ctx.saved_tensors
        dy = gradient.reshape(-1, gradient.shape[-1])
        inputs = values.reshape(-1, values.shape[-1])
        dx = torch.mm(dy, weight).reshape_as(values) if ctx.needs_input_grad[0] else None
        dw = head_weight_gradient(dy.contiguous(), inputs.contiguous()) if ctx.needs_input_grad[1] else None
        return dx, dw


class ArchivedHead(nn.Linear):
    """Cast each use separately, then join the shared parameter gradients in FP32."""

    keep_compute_in_fp32 = True

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Use the archived GEMM route only for this head's weight derivative."""
        return _HeadGradient.apply(values, self.weight.to(values.dtype))
