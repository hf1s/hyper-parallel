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
"""Muon logical matrix views matching MindFormers' DeepSeek MLA/MLP splits."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.optimizer.muon import NSInputTransform


@dataclass(frozen=True)
class JTMuonLayout:
    """Provide paired Muon callbacks for the JT model's native parameter layout.

    ``reshape_fn`` handles fused contiguous blocks as storage-sharing views.
    ``ns_transform_fn`` additionally gathers periodic attention rows into the
    two whole-head-group matrices used by MF, then restores their original rows.
    Both callbacks must be supplied for complete logical-matrix alignment.
    Each MF parameter's split matrices share its last matrix's RMS scale shape.
    The Q-down block inside HP fused QKV remains an independent MF parameter.
    NS coefficients, arithmetic order and update dtype remain native Muon choices.
    """

    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    num_attention_heads: int

    def __post_init__(self) -> None:
        """Reject missing or invalid logical matrix dimensions before optimizer setup."""
        values = (getattr(self, field.name) for field in fields(self))
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values):
            raise ValueError("JT Muon layout requires positive integer MLA dimensions")

    @classmethod
    def from_config(cls, config: Any) -> "JTMuonLayout":
        """Bind logical section sizes from a JT model configuration.

        Args:
            config: JT configuration containing the MLA ranks and head dimensions.

        Returns:
            A validated logical matrix layout shared by both optimizer callbacks.
        """
        return cls(**{field.name: getattr(config, field.name, None) for field in fields(cls)})

    def reshape_fn(self, name: str, tensor: torch.Tensor) -> list[torch.Tensor]:
        """Return independent fused blocks as views in the original HP storage.

        Args:
            name: Fully qualified JT parameter name.
            tensor: Complete parameter-shaped Newton-Schulz input.

        Returns:
            Logical matrices sharing storage with the input tensor.
        """
        if name.endswith(".self_attn.linear_qkv.weight"):
            sections = (self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim)
            if tensor.ndim != 2 or tensor.shape[0] != sum(sections):
                raise ValueError(f"Unexpected JT fused QKV shape for {name}: {tuple(tensor.shape)}")
            return list(tensor.split(sections, dim=0))
        if name.endswith(".mlp.experts.gate_up_proj"):
            if tensor.ndim != 3 or tensor.shape[1] % 2:
                raise ValueError(f"Unexpected JT expert gate/up shape for {name}: {tuple(tensor.shape)}")
            return list(tensor.chunk(2, dim=1))
        if name.endswith(".mlp.linear_fc1.weight") or name.endswith(".mlp.shared_experts.linear_fc1.weight"):
            if tensor.ndim != 2 or tensor.shape[0] % 2:
                raise ValueError(f"Unexpected JT MLP gate/up shape for {name}: {tuple(tensor.shape)}")
            return list(tensor.chunk(2, dim=0))
        return [tensor]

    def ns_transform_fn(self, name: str, tensor: torch.Tensor) -> NSInputTransform | None:
        """Group matching channels across all heads before their independent NS.

        Args:
            name: Fully qualified JT parameter name.
            tensor: Complete parameter-shaped Newton-Schulz input.

        Returns:
            A reversible transform with the corresponding MF parameter's scale shapes,
            or None to retain an unmatched parameter's native layout.
        """
        if name.endswith(".self_attn.q_b_proj.weight"):
            first, second = self.qk_nope_head_dim, self.qk_rope_head_dim
        elif name.endswith(".self_attn.kv_b_proj.weight"):
            first, second = self.qk_nope_head_dim, self.v_head_dim
        else:
            return self._fused_transform(name, tensor)
        heads = self.num_attention_heads
        width = first + second
        if tensor.ndim != 2 or tensor.shape[0] != heads * width:
            raise ValueError(f"Unexpected JT periodic attention shape for {name}: {tuple(tensor.shape)}")
        columns = tensor.shape[1]
        grouped = tensor.view(heads, width, columns)
        pieces = [grouped[:, :first].contiguous().view(heads * first, columns),
                  grouped[:, first:].contiguous().view(heads * second, columns)]

        def restore(updates: list[torch.Tensor], output: torch.Tensor) -> None:
            """Scatter the two whole-head-group updates back to their original rows.

            Args:
                updates: NoPE and RoPE/value matrix updates in logical group order.
                output: Destination in the original HP parameter layout.
            """
            destination = output.view(heads, width, columns)
            destination[:, :first].copy_(updates[0].view(heads, first, columns))
            destination[:, first:].copy_(updates[1].view(heads, second, columns))

        return NSInputTransform(tensors=pieces, restore=restore, scale_shapes=[tuple(pieces[-1].shape)] * 2)

    def _fused_transform(self, name: str, tensor: torch.Tensor) -> NSInputTransform | None:
        """Attach scale groups to contiguous fusions without changing their NS views.

        Args:
            name: Fully qualified JT parameter name.
            tensor: Complete parameter-shaped Newton-Schulz input.

        Returns:
            A fused logical transform, or None for an ordinary unsplit parameter.
        """
        pieces = self.reshape_fn(name, tensor)
        if len(pieces) == 1:
            return None
        shapes = [tuple(pieces[-1].shape[-2:])] * len(pieces)
        if name.endswith(".self_attn.linear_qkv.weight"):
            shapes[0] = tuple(pieces[0].shape[-2:])

        def restore(updates: list[torch.Tensor], output: torch.Tensor) -> None:
            """Write each logical block into the corresponding original storage view.

            Args:
                updates: Logical matrix updates in original fused-block order.
                output: Destination in the original HP parameter layout.
            """
            for target, update in zip(self.reshape_fn(name, output), updates):
                target.copy_(update)

        return NSInputTransform(tensors=pieces, restore=restore, scale_shapes=shapes)
