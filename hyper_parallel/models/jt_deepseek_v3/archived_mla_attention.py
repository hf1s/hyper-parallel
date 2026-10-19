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

"""Opt-in MF-archive MLA attention compatibility implementation."""

# This explicitly selected recipe is Torch/NPU-only.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.models.jt_deepseek_v3 import modeling_jt_deepseek_v3 as jt
from hyper_parallel.models.jt_deepseek_v3.archived_layers import ArchivedNorm
from hyper_parallel.models.replacement import module_replacement

__all__ = ["ArchivedMLAAttention", "SharedKeyHalfTree"]


class SharedKeyHalfTree(torch.autograd.Function):
    """Expand a singleton token-major key and sum head gradients in FP32."""

    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, heads: int) -> torch.Tensor:
        """Broadcast a singleton head while keeping sequence-major storage."""
        del ctx
        if value.ndim != 4 or value.shape[2] != 1 or heads < 1 or heads & (heads - 1):
            raise ValueError(
                "Expected singleton shared head and power-of-two head count: "
                f"{value.shape=}, {heads=}"
            )
        return value.expand(-1, -1, heads, -1)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Sum head contributions by a balanced FP32 tree, then round once to the key dtype."""
        del ctx
        reduced = gradient.float()
        while reduced.shape[2] > 1:
            half = reduced.shape[2] // 2
            reduced = reduced[:, :, :half] + reduced[:, :, half:]
        return reduced.to(gradient.dtype), None


class _SplitQKV(nn.Linear):
    """Run the two source projections while retaining the fused checkpoint weight."""

    def __init__(self, original: nn.Linear, q_rows: int) -> None:
        """Reuse the original fused parameter without copying or renaming its state."""
        super().__init__(original.in_features, original.out_features, bias=False)
        self.weight = original.weight
        self.q_rows = q_rows

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Run the two original projection GEMMs and restore the fused output layout."""
        return torch.cat((F.linear(values, self.weight[:self.q_rows]),  # pylint: disable=not-callable
                          F.linear(values, self.weight[self.q_rows:])), dim=-1)  # pylint: disable=not-callable


@module_replacement
class ArchivedMLAAttention(jt.JTDeepseekV3MLAAttention):
    """MF-archive MLA projection, latent-norm and shared-key backward behavior.

    This replacement is opt-in.  The standard ``JTDeepseekV3Attention`` and
    ``JTDeepseekV3MLAAttention`` implementations are intentionally unchanged.
    """

    def __init__(self, *, module: nn.Module, module_fqn: str = "", context: Any = None) -> None:
        """Compose the split projections and archived latent norms around the source parameters."""
        super().__init__(module=module, module_fqn=module_fqn, context=context)
        self.linear_qkv = _SplitQKV(self.linear_qkv, self.q_lora_rank)
        self.q_a_layernorm = ArchivedNorm(self.q_lora_rank, module.config.rms_norm_eps)
        self.kv_a_layernorm = ArchivedNorm(self.kv_lora_rank, module.config.rms_norm_eps)
        # Keep the source latent-norm parameters, as the product MLA keeps the source norm modules.
        self.q_a_layernorm.weight = module.q_a_layernorm.weight
        self.kv_a_layernorm.weight = module.kv_a_layernorm.weight

    def _project_attention_inputs(
        self, hidden_states: torch.Tensor, position_embeddings: Any, past_key_values: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project as the product MLA does, but reduce the shared rope key's head gradients in FP32."""
        if past_key_values is not None or position_embeddings is None:
            raise ValueError("Archived MLA requires RoPE tables and no KV cache")
        query_latent, kv_latent, key_rope = self.project_latent_inputs(hidden_states)
        batch, sequence = query_latent.shape[:2]
        query = self.q_b_proj(query_latent).reshape(batch, sequence, self.num_heads, self.qk_head_dim)
        query_pass, query_rope = query.split((self.qk_nope_head_dim, self.qk_rope_head_dim), dim=-1)
        cos, sin = position_embeddings
        # Assemble Q before expanding KV so its old projection views can be released.
        query = torch.cat(
            (query_pass, self.explicit_rotary(query_rope, cos, sin, unsqueeze_dim=2)), dim=-1
        )
        del query_pass, query_rope
        kv_latent = kv_latent.reshape(batch, 1, sequence, self.kv_lora_rank)
        kv_states = self.kv_b_proj(kv_latent).view(
            batch, sequence, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        key_pass, value = kv_states.split((self.qk_nope_head_dim, self.v_head_dim), dim=-1)
        key_rope = self.explicit_rotary(key_rope.unsqueeze(2), cos, sin, unsqueeze_dim=2)
        key = torch.cat((key_pass, SharedKeyHalfTree.apply(key_rope, self.num_heads)), dim=-1)
        return query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
