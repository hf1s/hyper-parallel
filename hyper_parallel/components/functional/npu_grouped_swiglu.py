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
"""Ascend grouped-GEMM implementation for packed SwiGLU experts."""
# pylint: disable=forbidden-backend-import

from typing import Any

import torch
from torch.distributed.tensor import DTensor as TorchDTensor

from hyper_parallel.core.dtensor.dtensor_base import DTensorBase


_NPU_GROUPED_MATMUL = None


def _get_npu_grouped_matmul() -> Any:
    """Build and cache the autograd wrapper after the Torch backend is selected."""
    global _NPU_GROUPED_MATMUL  # pylint: disable=global-statement
    if _NPU_GROUPED_MATMUL is not None:
        return _NPU_GROUPED_MATMUL

    # The NPU extension is optional for users of other backends.
    import torch_npu  # pylint: disable=C0415

    class _NPUGroupedMatmul(torch.autograd.Function):  # pylint: disable=abstract-method
        """Autograd wrapper around ``torch_npu.npu_grouped_matmul``."""

        @staticmethod
        def forward(ctx: Any, inputs: Any, weight: Any, group_list: Any, original_weight: Any) -> Any:
            """Run one grouped matrix multiplication and retain backward inputs."""
            if original_weight is not None:
                _validate_fused_weight(original_weight, inputs)
            ctx.original_weight = original_weight
            ctx.save_for_backward(inputs, weight, group_list)
            return torch_npu.npu_grouped_matmul(
                x=[inputs],
                weight=[weight],
                bias=[],
                group_list=group_list,
                split_item=3,
                group_type=0,
                group_list_type=0,
            )[0]

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, Any, None, None]:
            """Compute grouped input and weight gradients."""
            inputs, weight, group_list = ctx.saved_tensors
            grad_output = grad_output.contiguous()
            grad_inputs = torch_npu.npu_grouped_matmul(
                x=[grad_output],
                weight=[weight.transpose(-1, -2)],
                bias=[],
                group_list=group_list,
                split_item=3,
                group_type=0,
                group_list_type=0,
            )[0]
            if not ctx.needs_input_grad[1]:
                grad_weight = None
            elif ctx.original_weight is not None:
                _accumulate_fp32_weight_gradient(ctx.original_weight, inputs, grad_output, group_list)
                # Keep ordinary FSDP autograd scheduling, including mixed uses of
                # this parameter. The real contribution is in its FP32 main_grad.
                grad_weight = torch.zeros_like(weight)
            elif inputs.numel() == 0:
                grad_weight = torch.zeros_like(weight)
            else:
                grad_weight = torch_npu.npu_grouped_matmul(
                    x=[inputs.transpose(0, 1)],
                    weight=[grad_output],
                    bias=[],
                    group_list=group_list,
                    split_item=2,
                    group_type=2,
                    group_list_type=0,
                )[0]
            return grad_inputs, grad_weight, None, None

    _NPU_GROUPED_MATMUL = _NPUGroupedMatmul
    return _NPU_GROUPED_MATMUL


def _validate_fused_weight(weight: torch.Tensor, inputs: torch.Tensor) -> None:
    """Require an explicit main-gradient owner instead of guessing view ownership."""
    if isinstance(weight, (DTensorBase, TorchDTensor)) or isinstance(inputs, (DTensorBase, TorchDTensor)):
        raise ValueError("FP32 expert gradients require local tensors, not DTensor inputs or weights")
    if not hasattr(weight, "main_grad") or not hasattr(weight, "grad_added_to_main_grad"):
        raise ValueError("FP32 expert gradients require an unsharded main_grad owner and FP32 FSDP reduction")
    if inputs.dtype not in (torch.float16, torch.bfloat16) or weight.dtype != inputs.dtype:
        raise ValueError("FP32 expert gradients require matching FP16 or BF16 inputs and weights")
    if weight.ndim != 3 or not weight.is_contiguous():
        raise ValueError("FP32 expert gradients require contiguous three-dimensional local weights")
    gradient = weight.main_grad
    if gradient is not None and (
            not isinstance(gradient, torch.Tensor) or isinstance(gradient, (DTensorBase, TorchDTensor))
            or gradient.dtype != torch.float32 or gradient.shape != weight.shape
            or gradient.device != weight.device or not gradient.is_contiguous() or gradient.requires_grad):
        raise ValueError("The fused expert gradient buffer must match the weight shape and use contiguous FP32 storage")


def _accumulate_fp32_weight_gradient(
    weight: torch.Tensor, inputs: torch.Tensor, grad_output: torch.Tensor, group_list: torch.Tensor,
) -> None:
    """Accumulate directly in the parameter layout, avoiding a transposed FP32 copy."""
    if inputs.numel() == 0:
        return
    # The optional backend is needed only when this NPU kernel executes.
    import torch_npu  # pylint: disable=C0415

    if weight.main_grad is None:
        weight.main_grad = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
    torch_npu.npu_grouped_matmul_add_(
        weight.main_grad, grad_output, inputs, group_list,
        transpose_x=True, transpose_weight=False, group_type=2,
    )
    weight.grad_added_to_main_grad = True


def npu_grouped_swiglu(
    inputs: Any,
    gate_up_weight: Any,
    down_weight: Any,
    tokens_per_expert: Any,
    *,
    fp32_weight_gradients: bool = False,
) -> Any:
    """Run packed local SwiGLU experts with two grouped GEMMs.

    Args:
        inputs: Expert-major routed tokens with shape ``[T, H]``.
        gate_up_weight: Packed projection weights with shape ``[E, 2I, H]``.
        down_weight: Down projection weights with shape ``[E, H, I]``.
        tokens_per_expert: Token count for each local expert, shape ``[E]``.
        fp32_weight_gradients: Accumulate BF16/FP16 expert weight gradients directly
            into FP32 ``main_grad`` buffers. Requires FSDP's FP32 main-gradient
            policy and FP32 reduction; inputs and input gradients keep their dtype.

    Returns:
        Expert-major output with shape ``[T, H]``.

    Raises:
        ValueError: If called for a non-NPU tensor or incompatible shapes.
    """
    if inputs.device.type != "npu":
        raise ValueError("npu_grouped_swiglu requires inputs on an Ascend NPU")
    if gate_up_weight.ndim != 3 or down_weight.ndim != 3:
        raise ValueError("grouped SwiGLU weights must be three-dimensional")
    if gate_up_weight.shape[0] != tokens_per_expert.numel():
        raise ValueError("tokens_per_expert must contain one count per local expert")
    if not isinstance(fp32_weight_gradients, bool):
        raise TypeError("fp32_weight_gradients must be a bool")

    # The optional extension must not be imported by CPU-only callers.
    import torch_npu  # pylint: disable=C0415

    grouped_matmul = _get_npu_grouped_matmul()
    group_list = torch.cumsum(tokens_per_expert.to(torch.int64), dim=0)
    gate_up = grouped_matmul.apply(
        inputs,
        gate_up_weight.transpose(1, 2),
        group_list,
        gate_up_weight if fp32_weight_gradients else None,
    )
    intermediate = torch_npu.npu_swiglu(gate_up, dim=-1)
    return grouped_matmul.apply(
        intermediate,
        down_weight.transpose(1, 2),
        group_list,
        down_weight if fp32_weight_gradients else None,
    )
