# Copyright 2025-2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trainer-side seeding and numerical-determinism controls.

Split out of the former ``auto_models/components/utils/helper.py`` in stage 7
(05 §10.4); function names and signatures are unchanged.
"""

import os
import random

import numpy as np
import torch
from transformers import set_seed as set_seed_func

from hyper_parallel.models.build_options import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE

if IS_NPU_AVAILABLE:
    import torch_npu  # noqa: F401


def enable_high_precision_for_bf16() -> None:
    """
    Set high accumulation dtype for matmul and reduction.
    """
    if IS_CUDA_AVAILABLE:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    if IS_NPU_AVAILABLE:
        torch.npu.matmul.allow_tf32 = False
        torch.npu.matmul.allow_bf16_reduced_precision_reduction = False


def enable_full_determinism(seed: int, *, close_matmul_k_shift: bool = True) -> None:
    """
    Helper function for reproducibility in distributed training.
    See https://pytorch.org/docs/stable/notes/randomness.html for details.

    Args:
        seed: Seed applied to Python, NumPy, and framework generators.
        close_matmul_k_shift: On NPU, disable CANN matmul K-shift when true.
            False explicitly permits K-shift while retaining the other deterministic
            settings. Select this policy before the first matmul in a fresh process.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["NCCL_DETERMINISTIC"] = "1"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    if IS_NPU_AVAILABLE:
        # Keep the historical default while allowing explicit reference-compatible matmul ordering.
        os.environ["NCCL_DETERMINISTIC"] = "true"
        os.environ["CLOSE_MATMUL_K_SHIFT"] = "1" if close_matmul_k_shift else "0"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    if IS_NPU_AVAILABLE:
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)


def set_seed(seed: int | None, full_determinism: bool = False, *, close_matmul_k_shift: bool = True) -> None:
    """
    Sets a manual seed on all devices.

    Args:
        seed: Seed to apply, or ``None`` to leave the process unchanged.
        full_determinism: Whether to configure deterministic algorithms as well as RNGs.
        close_matmul_k_shift: NPU matmul policy used only with full determinism.
    """
    if seed is None:
        return
    if full_determinism:
        enable_full_determinism(seed, close_matmul_k_shift=close_matmul_k_shift)
    else:
        set_seed_func(seed)


__all__ = [
    "enable_full_determinism",
    "enable_high_precision_for_bf16",
    "set_seed",
]
