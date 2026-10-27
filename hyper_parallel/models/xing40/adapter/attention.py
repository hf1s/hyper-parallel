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
"""NPU multi-latent attention of Xing4.0: the framework ``MLAAttention`` with MindFormers numerics.

The recipe replaces the Transformers ``DeepseekV3Attention`` of every layer with
:class:`Xing40MLAAttention`. Importing this module requires ``torch_npu``.
"""

# This adapter uses the Torch runtime, like the framework attention modules.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch_npu
from torch import nn
from transformers.models.deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave

from hyper_parallel.components.functional.npu_fusion_attention import _prepare_fusion_attention_context
from hyper_parallel.components.modules.mla_attention import MLAAttention
from hyper_parallel.components.modules.rms_norm import RMSNorm
from hyper_parallel.models.replacement import module_replacement

__all__ = ["Xing40MLAAttention"]


def fusion_attention_with_logit_maxima(module: nn.Module, query: torch.Tensor, key: torch.Tensor,
                                       value: torch.Tensor, attention_mask: torch.Tensor | None,
                                       **kwargs: Any) -> tuple[torch.Tensor, None]:
    """Run causal TND fused attention and fold the per-head logit maxima into ``module.max_logits``.

    Args:
        module: Attention module; its ``max_logits`` holds the maxima since the last QK clipping.
        query: Query states, ``[batch, heads, sequence, head_dim]``.
        key: Key states, ``[batch, heads, sequence, head_dim]``.
        value: Value states, ``[batch, heads, sequence, v_head_dim]``.
        attention_mask: Unused; every sequence attends causally.
        **kwargs: ``scaling``, the attention score scale; the other options of the
            framework attention interface are unused.

    Returns:
        The attention output, ``[batch, sequence, heads, v_head_dim]``, and no attention weights.
    """
    del attention_mask
    batch, heads, sequence = query.shape[:3]
    lengths = tuple(range(sequence, batch * sequence + 1, sequence))
    context = _prepare_fusion_attention_context(module, query, key, value, None, {"actual_seq_len": lengths})
    output, softmax_max = torch_npu.npu_fusion_attention(
        context.query, context.key, context.value, heads, context.input_layout,
        pse=None, padding_mask=None, atten_mask=context.attention_mask, scale=kwargs["scaling"],
        pre_tockens=context.pre_tokens, next_tockens=context.next_tokens, keep_prob=1.0,
        inner_precise=0, sparse_mode=context.sparse_mode, actual_seq_qlen=context.query_lengths,
        actual_seq_kvlen=context.key_lengths, softmax_layout="TND")[:2]
    with torch.no_grad():
        maximum = softmax_max.amax(dim=(0, 2))
        module.max_logits = maximum if module.max_logits is None else torch.maximum(module.max_logits, maximum)
    return output.view(batch, sequence, heads, -1), None


@module_replacement
class Xing40MLAAttention(MLAAttention):
    """Framework MLA with the MindFormers FP32 RoPE that records the logit maxima QK clipping consumes.

    ``q_a_proj`` and ``kv_a_proj_with_mqa`` fuse into ``linear_qkv``, and the latent norms
    become the framework's NPU ``RMSNorm``: a replacement plan cannot also select these
    children of the replaced attention.
    """

    def __init__(self, *, module: nn.Module, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None) -> None:
        """Build the replacement from the Transformers ``DeepseekV3Attention`` it replaces.

        Args:
            module: Attention module being replaced.
            module_fqn: Fully qualified name of ``module``.
            context: Replacement context of the build pipeline.
        """
        super().__init__(module=module, module_fqn=module_fqn, context=context,
                         attention_interface=fusion_attention_with_logit_maxima)
        self.q_a_layernorm = RMSNorm(module=self.q_a_layernorm)
        self.kv_a_layernorm = RMSNorm(module=self.kv_a_layernorm)
        if not self.linear_qkv.weight.is_meta:
            with torch.no_grad():
                self.linear_qkv.weight.copy_(torch.cat((module.q_a_proj.weight, module.kv_a_proj_with_mqa.weight)))
        # Per-head logit maxima since the last QK clipping, folded in by the attention interface.
        self.max_logits: torch.Tensor | None = None

    def _project_attention_inputs(self, hidden_states: torch.Tensor,
                                  position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
                                  past_key_values: Any | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states to query, key and value, rotating in FP32 and rounding once as MindFormers.

        Args:
            hidden_states: Normalized layer input, ``[batch, sequence, hidden]``.
            position_embeddings: FP32 RoPE ``(cos, sin)`` tables.
            past_key_values: Must be None; training uses no KV cache.

        Returns:
            Query, key and value states, ``[batch, heads, sequence, head_dim]``.

        Raises:
            ValueError: If the RoPE tables are missing or a KV cache is given.
        """
        if position_embeddings is None or past_key_values is not None:
            raise ValueError("Xing4.0 MLA requires RoPE tables and no KV cache")
        latents = self._project_latents(hidden_states)
        cos, sin = position_embeddings
        query_rope, key_rope = apply_rotary_pos_emb_interleave(
            latents.query_rope.float(), latents.key_rope.float(), cos, sin)
        key_value = self.kv_b_proj(latents.kv_nope).view(
            latents.batch_size, latents.sequence_length, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        key_nope, value = key_value.transpose(1, 2).split((self.qk_nope_head_dim, self.v_head_dim), dim=-1)
        query_pass = latents.query_pass.transpose(1, 2)
        query = torch.cat((query_pass, query_rope.to(query_pass.dtype)), dim=-1)
        key = torch.cat((key_nope, key_rope.to(key_nope.dtype).expand(-1, self.num_heads, -1, -1)), dim=-1)
        return query, key, value
