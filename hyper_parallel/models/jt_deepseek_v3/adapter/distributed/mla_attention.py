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
"""Canonical-parameter JT MLA replacement with fused forward projection."""

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
import torch_npu

from hyper_parallel.components.checkpoint.conversion_ops import Split
from hyper_parallel.components.checkpoint.weight_conversion import WeightConverter
from hyper_parallel.components.functional.npu_fusion_attention import (
    _attention_options, _prepare_attention_inputs, resolve_packed_sequence_lengths,
)
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import ExplicitFP32RotaryEmbedding
from hyper_parallel.models.replacement import module_replacement

def observed_fusion_attention(module: nn.Module, query: torch.Tensor, key: torch.Tensor,
                              value: torch.Tensor, attention_mask: Any, dropout: float = 0.0,
                              scaling: float | None = None, **kwargs: Any) -> tuple[torch.Tensor, None]:
    """Run the fused attention kernel and collect QK clipping statistics."""
    batch_size, _, query_length, head_dim = query.shape
    query_lengths, key_lengths = resolve_packed_sequence_lengths(
        kwargs, batch_size * query_length, key.shape[0] * key.shape[2])
    pre_tokens, next_tokens, sparse_mode, window, causal = _attention_options(module, kwargs)
    query, key, value, layout, mask, sparse_mode = _prepare_attention_inputs(
        query, key, value, attention_mask, is_packed=query_lengths is not None,
        is_causal=causal, sliding_window=window, sparse_mode=sparse_mode)
    result = torch_npu.npu_fusion_attention(
        query, key, value, query.shape[1], layout,
        pse=None, padding_mask=None, atten_mask=mask,
        scale=head_dim**-0.5 if scaling is None else scaling,
        pre_tockens=pre_tokens, next_tockens=next_tokens,
        keep_prob=1.0 - dropout, inner_precise=0, sparse_mode=sparse_mode,
        actual_seq_qlen=query_lengths, actual_seq_kvlen=key_lengths)
    with torch.no_grad():
        maximum = result[1].amax(dim=(0, 2))
        if module.max_logits_val is None:
            module.max_logits_val = torch.zeros_like(maximum)
        module.max_logits_val.copy_(torch.maximum(module.max_logits_val, maximum))
    if query_lengths is not None:
        return result[0].reshape(batch_size, query_length, *result[0].shape[1:]), None
    return result[0].transpose(1, 2), None



@module_replacement
class JTDeepseekV3FusedMLAAttention(nn.Module):
    """Fuse canonical q/kv latent projections only for the forward computation."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "", context: Any = None) -> None:
        """Transfer canonical parameters without creating a fused trainable weight."""
        super().__init__()
        del module_fqn, context
        for name, child in module._modules.items():  # pylint: disable=protected-access
            self.add_module(name, child)
        for name, parameter in module._parameters.items():  # pylint: disable=protected-access
            self.register_parameter(name, parameter)
        for name, buffer in module._buffers.items():  # pylint: disable=protected-access
            self.register_buffer(
                name,
                buffer,
                persistent=name not in module._non_persistent_buffers_set,  # pylint: disable=protected-access
            )
        self.config = module.config
        self.layer_idx = module.layer_idx
        self.num_heads = module.num_heads
        self.q_lora_rank = module.q_lora_rank
        self.kv_lora_rank = module.kv_lora_rank
        self.qk_rope_head_dim = module.qk_rope_head_dim
        self.qk_nope_head_dim = module.qk_nope_head_dim
        self.v_head_dim = module.v_head_dim
        self.qk_head_dim = module.qk_head_dim
        self.scaling = module.scaling
        self.attention_dropout = module.attention_dropout
        self.is_causal = module.is_causal
        self.sliding_window = module.sliding_window
        self.explicit_rotary = ExplicitFP32RotaryEmbedding()
        self.key_rope_gather = nn.Identity()
        self.attention_interface = observed_fusion_attention
        self.train(module.training)

    def _project_attention_inputs(self, hidden_states: torch.Tensor, position_embeddings: Any,
                                  past_key_values: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one fused latent projection while retaining canonical parameters."""
        if past_key_values is not None or position_embeddings is None:
            raise ValueError("Reference MLA requires explicit positions and no KV cache")
        fused_weight = torch.cat(
            (self.q_a_proj.weight, self.kv_a_proj_with_mqa.weight),
            dim=0,
        )
        latent_states = F.linear(hidden_states, fused_weight)
        query_local, kv_local = latent_states.split(
            (self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim), dim=-1)
        kv_local, rope_local = kv_local.split((self.kv_lora_rank, self.qk_rope_head_dim), dim=-1)
        query_latent = self.q_a_layernorm(query_local)
        kv_latent = self.kv_a_layernorm(kv_local)
        key_rope = self.key_rope_gather(rope_local)
        batch, sequence = query_latent.shape[:2]
        query = self.q_b_proj(query_latent).reshape(batch, sequence, self.num_heads, self.qk_head_dim)
        query_pass, query_rope = query.split((self.qk_nope_head_dim, self.qk_rope_head_dim), dim=-1)
        kv_latent = kv_latent.reshape(batch, 1, sequence, self.kv_lora_rank)
        kv_states = self.kv_b_proj(kv_latent).view(
            batch, sequence, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        key_pass, value = kv_states.split((self.qk_nope_head_dim, self.v_head_dim), dim=-1)
        key_rope = key_rope.reshape(batch, 1, sequence, self.qk_rope_head_dim).expand(-1, self.num_heads, -1, -1)
        cos, sin = position_embeddings
        query = torch.cat((query_pass.transpose(1, 2),
                           self.explicit_rotary(query_rope.transpose(1, 2), cos, sin)), dim=-1)
        key = torch.cat((key_pass, self.explicit_rotary(key_rope, cos, sin)), dim=-1)
        return query, key, value

    def make_transforms(self) -> list[WeightConverter]:
        """Split legacy fused checkpoint weights into canonical target parameters."""
        return [
            WeightConverter(
                source_patterns="linear_qkv.weight",
                target_patterns=["q_a_proj.weight", "kv_a_proj_with_mqa.weight"],
                operations=[
                    Split(
                        (self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim),
                        dim=0,
                    )
                ],
            )
        ]



__all__ = ["JTDeepseekV3FusedMLAAttention"]
