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

"""Scoped MatMulV2 NZ head-gradient route for the explicit archived NPU recipe."""

# This explicitly selected recipe is Torch/NPU-only.
# pylint: disable=forbidden-backend-import

import ctypes as ct
from functools import lru_cache

import torch
import torch_npu


@lru_cache(maxsize=1)
def _api() -> dict:
    library = ct.CDLL("libacl_op_compiler.so")
    pointer = ct.c_void_p
    pointers = ct.POINTER(pointer)
    signatures = {
        "create_desc": ("aclCreateTensorDesc", [ct.c_int, ct.c_int, ct.POINTER(ct.c_int64), ct.c_int], pointer),
        "destroy_desc": ("aclDestroyTensorDesc", [pointer], None),
        "create_buffer": ("aclCreateDataBuffer", [pointer, ct.c_size_t], pointer),
        "destroy_buffer": ("aclDestroyDataBuffer", [pointer], ct.c_int32),
        "create_attr": ("aclopCreateAttr", [], pointer),
        "destroy_attr": ("aclopDestroyAttr", [pointer], None),
        "set_bool": ("aclopSetAttrBool", [pointer, ct.c_char_p, ct.c_uint8], ct.c_int32),
        "set_storage_shape": ("aclSetTensorStorageShape", [pointer, ct.c_int, ct.POINTER(ct.c_int64)], ct.c_int32),
        "set_storage_format": ("aclSetTensorStorageFormat", [pointer, ct.c_int], ct.c_int32),
        "execute": ("aclopCompileAndExecuteV2", [ct.c_char_p, ct.c_int, pointers, pointers, ct.c_int,
                                                pointers, pointers, pointer, ct.c_int, ct.c_int,
                                                ct.c_char_p, pointer], ct.c_int32),
    }
    result = {"library": library}
    for key, (name, arguments, return_type) in signatures.items():
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, return_type
        result[key] = function
    return result


def _check(status: int, operation: str) -> None:
    if status:
        raise RuntimeError(f"{operation}: CANN status {status}")


def _nz_matmul(gradient: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    api = _api()
    pointer = ct.c_void_p
    left, right = [torch_npu.npu_format_cast(value, 29) for value in (gradient, inputs)]
    output = torch_npu.empty_with_format((gradient.shape[1], inputs.shape[1]),
                                         dtype=torch.bfloat16, device=inputs.device, acl_format=29)
    descriptors, buffers = [], []
    attribute = api["create_attr"]()
    if not attribute:
        raise RuntimeError("aclopCreateAttr returned null")
    try:
        for value in (left, right, output):
            shape = list(value.shape)
            storage = [shape[1] // 16, shape[0] // 16, 16, 16]
            if torch_npu.get_npu_format(value) != 29 or torch_npu.get_storage_size(value) != value.numel():
                raise ValueError("Unexpected NZ storage format or physical element count")
            descriptor = api["create_desc"](27, 2, (ct.c_int64 * 2)(*shape), 2)
            if not descriptor:
                raise RuntimeError("aclCreateTensorDesc returned null")
            descriptors.append(descriptor)
            _check(api["set_storage_shape"](descriptor, 4, (ct.c_int64 * 4)(*storage)), "aclSetTensorStorageShape")
            _check(api["set_storage_format"](descriptor, 29), "aclSetTensorStorageFormat")
            buffer = api["create_buffer"](value.data_ptr(), value.untyped_storage().nbytes())
            if not buffer:
                raise RuntimeError("aclCreateDataBuffer returned null")
            buffers.append(buffer)
        _check(api["set_bool"](attribute, b"transpose_x1", 1), "transpose_x1")
        _check(api["set_bool"](attribute, b"transpose_x2", 0), "transpose_x2")
        torch.npu.synchronize()
        _check(api["execute"](b"MatMulV2", 2, (pointer * 2)(*descriptors[:2]), (pointer * 2)(*buffers[:2]),
                              1, (pointer * 1)(descriptors[2]), (pointer * 1)(buffers[2]), attribute,
                              0, 0, None, torch.npu.current_stream().npu_stream), "MatMulV2/NZ")
        torch.npu.synchronize()
        result = torch_npu.npu_format_cast(output, 2)
        torch.npu.synchronize()
        return result
    finally:
        for buffer in buffers:
            _check(api["destroy_buffer"](buffer), "aclDestroyDataBuffer")
        for descriptor in descriptors:
            api["destroy_desc"](descriptor)
        api["destroy_attr"](attribute)


def head_weight_gradient(gradient: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Compute ``gradient.T @ inputs`` with scoped compiler state and validated storage.

    This route reproduces the archived head's GEMM accumulation. It is restricted
    to the single-threaded trainer of this profile; compiler options are global
    runtime state and must not be changed concurrently by another Python thread.
    """
    if gradient.device != inputs.device or gradient.shape[0] != inputs.shape[0]:
        raise ValueError("Head gradient operands must have identical device and reduction size")
    for value in (gradient, inputs):
        if (value.ndim != 2 or value.dtype != torch.bfloat16 or not value.is_contiguous()
                or value.storage_offset() != 0 or any(size % 16 for size in value.shape)
                or torch_npu.get_npu_format(value) not in (0, 2)):
            raise ValueError("Expected zero-offset, 16-aligned BF16 ND matrices")
    was_eager = torch_npu.npu.is_jit_compile_false()
    # TorchNPU 2.12 exposes a setter-only config attribute. Read its backing
    # option so this scoped operation restores the caller's actual setting.
    internal_option = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")  # pylint: disable=protected-access
    if internal_option not in (b"enable", b"disable"):
        raise ValueError("Set allow_internal_format explicitly before using the archived GEMM route")
    internal_format = internal_option == b"enable"
    with torch.npu.device(inputs.device):
        try:
            torch_npu.npu.set_compile_mode(jit_compile=True)
            torch.npu.config.allow_internal_format = True
            return _nz_matmul(gradient, inputs)
        finally:
            torch_npu.npu.set_compile_mode(jit_compile=not was_eager)
            torch.npu.config.allow_internal_format = internal_format
