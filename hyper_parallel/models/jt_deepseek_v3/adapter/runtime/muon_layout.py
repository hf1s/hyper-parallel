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

"""Preserve logical Muon matrices when JT fuses parameter storage."""

from collections.abc import Callable

import torch

from hyper_parallel.core.optimizer.muon import NSInputTransform


def build_fused_matrix_transform(
        q_rows: int | None, kv_rows: int) -> Callable[[str, torch.Tensor], NSInputTransform | None]:
    """Build a reversible transform for storage-only Q/KV and expert fusion.

    The latent-KV projection remains one logical matrix, including its rotary
    rows. Further splitting it or Q-up into rotary/nonrotary parts is a
    different optimization recipe, not a consequence of storage fusion.

    Args:
        q_rows: Rows belonging to the unfused Q down-projection, or None for
            models without a low-rank Q projection.
        kv_rows: Rows belonging to the unfused latent-KV down-projection.

    Returns:
        Model-named Muon transform that preserves all other matrix boundaries.
    """
    if (q_rows is not None and q_rows <= 0) or kv_rows <= 0:
        raise ValueError("Q and latent-KV row counts must be positive")

    def transform(name: str, value: torch.Tensor) -> NSInputTransform | None:
        if name.endswith(".self_attn.linear_qkv.weight"):
            if q_rows is None or value.ndim != 2 or value.shape[0] != q_rows + kv_rows:
                raise ValueError(f"Unexpected fused Q/KV matrix: {name}, shape={tuple(value.shape)}")
            tensors = list(value.split((q_rows, kv_rows), dim=0))
            dimension = 0
        elif name.endswith(".mlp.experts.gate_up_proj"):
            if value.ndim != 3 or value.shape[1] == 0 or value.shape[1] % 2:
                raise ValueError(f"Unexpected fused expert gate/up matrix: {name}, shape={tuple(value.shape)}")
            tensors = list(value.chunk(2, dim=1))
            dimension = 1
        else:
            return None

        def restore(updates: list[torch.Tensor], output: torch.Tensor) -> None:
            if len(updates) != len(tensors) or any(
                    update.shape != tensor.shape for update, tensor in zip(updates, tensors)):
                raise ValueError(f"Muon update pieces do not match the fused matrix: {name}")
            output.copy_(torch.cat(updates, dim=dimension))

        return NSInputTransform(tensors=tensors, restore=restore)

    return transform
