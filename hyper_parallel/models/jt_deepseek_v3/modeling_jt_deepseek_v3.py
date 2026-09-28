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
"""Complete HF-derived causal model, prediction depths and reference training semantics."""

# This model uses the Torch/HF runtime, like the existing Trainer model families.
# pylint: disable=forbidden-backend-import
from __future__ import annotations
import copy
import functools
import logging
import os
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
from torch import nn
from torch.nn import functional as F
from transformers import DeepseekV32Config, DeepseekV32ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.deepseek_v32.modeling_deepseek_v32 import (
    DeepseekV32Attention, DeepseekV32DecoderLayer, DeepseekV32MLP,
    DeepseekV32MoE, DeepseekV32Model, DeepseekV32PreTrainedModel, DeepseekV32RMSNorm,
    DeepseekV32RotaryEmbedding, DeepseekV32TopkRouter,
)

from hyper_parallel.components.modules.mtp import DeepseekV3MTP
from hyper_parallel.components.losses._vocab_parallel_cross_entropy import vocab_parallel_cross_entropy_local
from hyper_parallel.core.tensor_parallel.loss_parallel import _get_loss_parallel_mesh
from hyper_parallel.distributed.expert_parallel.routing import MOE_ROUTER_ADAPTERS
logger = logging.getLogger(__name__)


class JTDeepseekV3RMSNorm(DeepseekV32RMSNorm):
    """Keep the HF scale and use FP32 reference normalization with an explicit cast."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Match the reference fused FP32 norm's output dtype boundary.

        Args:
            hidden_states: Input hidden states.
        """
        if hidden_states.device.type == "npu":
            normalized = torch_npu.npu_rms_norm(hidden_states.float(), self.weight, self.variance_epsilon)[0]
        else:
            values = hidden_states.float()
            normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.variance_epsilon)
            normalized = normalized * self.weight
        return normalized.to(hidden_states.dtype)


class ReferenceSwiGLU(torch.autograd.Function):
    """Retain the reference graph's SiLU derivative and BF16 rounding boundaries."""

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, rounded_up_gradient: bool) -> torch.Tensor:
        """Save gate/up inputs; MTP materializes SiLU before its up gradient.

        Args:
            ctx: Autograd context.
            inputs: Inputs retained for the custom derivative.
            rounded_up_gradient: Whether to round SiLU before the up-projection gradient.
        """
        ctx.save_for_backward(inputs)
        ctx.rounded_up_gradient = rounded_up_gradient
        if inputs.device.type == "npu":
            return torch_npu.npu_swiglu(inputs, dim=-1)
        gate, up = inputs.float().chunk(2, dim=-1)
        return (F.silu(gate) * up).to(inputs.dtype)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Follow the static graph's operation order, including its FP32 divisions.

        Args:
            ctx: Autograd context.
            gradient: Upstream gradient.
        """
        (inputs,) = ctx.saved_tensors
        gate, up = inputs.float().chunk(2, dim=-1)
        denominator = torch.exp(-gate) + 1
        sigmoid = 1 / denominator
        silu = gate / denominator
        derivative = (sigmoid + silu) - sigmoid * silu
        gate_gradient = (derivative * (up * gradient.float())).to(inputs.dtype)
        if ctx.rounded_up_gradient:
            silu = silu.to(inputs.dtype).float()
        up_gradient = (gradient.float() * silu).to(inputs.dtype)
        return torch.cat((gate_gradient, up_gradient), dim=-1), None


class JTDeepseekV3MLP(DeepseekV32MLP):
    """Retain HF gate/up/down projections and replace only the SwiGLU computation."""

    def __init__(self, config: Any, intermediate_size: int | None = None,
                 rounded_up_gradient: bool = False) -> None:
        """Construct HF projections with the configured SwiGLU backward boundary."""
        super().__init__(config, intermediate_size=intermediate_size)
        self.rounded_up_gradient = rounded_up_gradient

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Avoid materializing a rounded SiLU result before the gate/up product.

        Args:
            x: X.
        """
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        values = ReferenceSwiGLU.apply(torch.cat((gate, up), dim=-1), self.rounded_up_gradient)
        return self.down_proj(values)


class JTDeepseekV3Experts(nn.Module):
    """Keep routed expert parameters canonical while retaining grouped execution."""

    def __init__(self, config: Any) -> None:
        """Create separate gate, up, and down expert parameter tensors."""
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_proj = nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_dim, self.hidden_dim)
        )
        self.up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
        self.act_fn = F.silu



class ExplicitFP32RotaryEmbedding(nn.Module):
    """Preserve separate FP32 products and addition before the BF16 RoPE cast."""

    def forward(self, values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Reorder even/odd channels and rotate using the reference operation sequence.

        Args:
            values: Values to transform or reduce.
            cos: Cosine position frequencies.
            sin: Sine position frequencies.
        """
        ordered = torch.cat((values[..., ::2], values[..., 1::2]), dim=-1).float()
        first, second = ordered.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return (ordered * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)).to(values.dtype)






class JTDeepseekV3Attention(DeepseekV32Attention):
    """Full causal MLA semantics with HF projection names and no DSA indexer."""

    def __init__(self, config: Any, layer_idx: int) -> None:
        """Construct only the projections used by the configured causal model."""
        nn.Module.__init__(self)
        self.config, self.layer_idx = config, layer_idx
        self.num_heads = config.num_attention_heads
        self.q_lora_rank, self.kv_lora_rank = config.q_lora_rank, config.kv_lora_rank
        self.qk_rope_head_dim, self.qk_nope_head_dim = config.qk_rope_head_dim, config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        self.scaling = self.qk_head_dim ** -0.5
        self.attention_dropout, self.is_causal, self.sliding_window = config.attention_dropout, True, None
        self.q_a_proj = nn.Linear(config.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = JTDeepseekV3RMSNorm(self.q_lora_rank, config.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(config.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = JTDeepseekV3RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)
        self.explicit_rotary = ExplicitFP32RotaryEmbedding()
        self.key_rope_gather = nn.Identity()
        self.register_buffer("max_logits_val", None, persistent=False)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: Any = None,
                past_key_values: Any = None, attention_mask: Any = None, **kwargs: Any) -> tuple:
        """Run the complete causal attention before optional high-performance replacement.

        Args:
            hidden_states: Input hidden states.
            position_embeddings: Explicit rotary frequencies.
            past_key_values: Unsupported cached decoding state.
            attention_mask: Optional attention mask.
        """
        if past_key_values is not None or position_embeddings is None:
            raise ValueError("Reference attention requires explicit positions and no KV cache")
        batch, sequence = hidden_states.shape[:2]
        query = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        query = query.reshape(batch, sequence, self.num_heads, self.qk_head_dim).transpose(1, 2)
        kv, rope = self.kv_a_proj_with_mqa(hidden_states).split((self.kv_lora_rank, self.qk_rope_head_dim), dim=-1)
        rope = self.key_rope_gather(rope)
        kv = self.kv_b_proj(self.kv_a_layernorm(kv)).reshape(
            batch, sequence, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        key, value = kv.split((self.qk_nope_head_dim, self.v_head_dim), dim=-1)
        cos, sin = position_embeddings
        query = torch.cat((query[..., :self.qk_nope_head_dim],
                           self.explicit_rotary(query[..., self.qk_nope_head_dim:], cos, sin)), dim=-1)
        rope = self.explicit_rotary(rope.unsqueeze(1), cos, sin).expand(-1, self.num_heads, -1, -1)
        key = torch.cat((key, rope), dim=-1)
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask,
            is_causal=attention_mask is None, scale=self.scaling)
        output = output.transpose(1, 2)
        return self.o_proj(output.reshape(batch, sequence, -1)), None


class ExpertCombine(torch.autograd.Function):
    """Match the reference fused forward and BF16 routed-combine backward."""

    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, probabilities: torch.Tensor,
                rounded_probability: bool = False) -> torch.Tensor:
        """Accumulate selected expert outputs in FP32 before the activation cast.

        Args:
            ctx: Autograd context.
            values: Values to transform or reduce.
            probabilities: Selected routing probabilities.
            rounded_probability: Whether to round probabilities before the value gradient.
        """
        ctx.save_for_backward(values, probabilities)
        ctx.rounded_probability = rounded_probability
        return (values.float() * probabilities.unsqueeze(-1)).sum(1).to(values.dtype)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Retain the reference probability cast and product before reduction.

        Args:
            ctx: Autograd context.
            gradient: Upstream gradient.
        """
        values, probabilities = ctx.saved_tensors
        gradient = gradient.unsqueeze(1)
        factor = probabilities.to(values.dtype).float() if ctx.rounded_probability else probabilities
        value_gradient = (gradient.float() * factor.unsqueeze(-1)).to(values.dtype)
        probability_gradient = (gradient * values).sum(-1).to(probabilities.dtype)
        return value_gradient, probability_gradient, None


class _ModelParallelMean(torch.autograd.Function):
    """Replicate one global mean while differentiating each local contribution once."""

    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, group: Any) -> torch.Tensor:
        """Average equally weighted contributions across the model-parallel group."""
        ctx.world_size = dist.get_world_size(group)
        result = value.clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
        return result / ctx.world_size

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Scale the local derivative without summing identical output replicas."""
        return gradient / ctx.world_size, None


def _model_parallel_mean(value: torch.Tensor, group: Any = None) -> torch.Tensor:
    """Reduce a partitioned objective; absent an explicit group, keep it local.

    Contributions must be equally weighted, with the same upstream derivative
    on all ranks. This does not implement DDP averaging or independently
    consumed all-reduce outputs; uneven token partitions need explicit weights.
    """
    return value if group is None else _ModelParallelMean.apply(value, group)


class JTDeepseekV3Gate(DeepseekV32TopkRouter):
    """Expose HF gate logits to the public router and the model's auxiliary loss."""

    def __init__(self, config: Any) -> None:
        """Keep the HF parameters and bias without duplicating its top-k computation."""
        super().__init__(config)
        self.router_logits = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project once; retain full logits until the caller constructs auxiliary loss."""
        self.router_logits = F.linear(hidden_states.reshape(-1, self.hidden_dim).float(), self.weight.float())
        return self.router_logits


class JTDeepseekV3MoE(DeepseekV32MoE):
    """Own routing, balancing and combine semantics independently of EP binding."""

    def __init__(self, config: Any, *, is_mtp: bool = False) -> None:
        """Construct all expert branches and the standalone routing contract."""
        nn.Module.__init__(self)
        self.config = config
        self.reference_is_mtp = is_mtp
        self.padding = config.n_routed_experts if config.use_pad_tokens else 0
        self.experts = JTDeepseekV3Experts(config)
        self.gate = JTDeepseekV3Gate(config)
        self.shared_experts = JTDeepseekV3MLP(
            config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            rounded_up_gradient=is_mtp)
        self.ep_group, self.ep_world = None, 1
        self.ep_compute = self.local_routed_forward
        self.auxiliary_loss = None
        self.expert_load = None

    def route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select experts and compute the auxiliary sequence-balancing loss.

        Args:
            hidden: Hidden states including optional padding tokens.
        """
        group, world, padding = self.ep_group, self.ep_world, self.padding
        config = self.config
        hidden = hidden[:, padding:]
        with torch.autocast(hidden.device.type, enabled=False):
            indices, selected = MOE_ROUTER_ADAPTERS["deepseekv3"](self, hidden)
            scores = self.gate.router_logits.sigmoid()
            self.gate.router_logits = None
            frequency = torch.bincount(indices.flatten(), minlength=config.n_routed_experts).float()
            frequency = frequency / indices.numel()
            if group is not None:
                dist.all_reduce(frequency, group=group)
            frequency = frequency / world
            self.expert_load = frequency.detach()
            normalized = scores / (scores.sum(-1, keepdim=True) + 1e-20)
            self.auxiliary_loss = (normalized.mean(0) * frequency).sum() * scores.shape[-1] * config.moe_aux_loss_coeff
            self.auxiliary_loss = _model_parallel_mean(self.auxiliary_loss, group)
        if padding:
            pad_ids = torch.arange(padding * config.num_experts_per_tok, device=indices.device)
            pad_ids = pad_ids.reshape(padding, config.num_experts_per_tok) % padding
            indices = torch.cat((pad_ids, indices))
            selected = torch.cat((selected.new_zeros(padding, selected.shape[-1]), selected))
        return indices, selected

    def local_routed_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Execute the same router and ordered combine without distributed setup.

        Args:
            hidden: Hidden states.
        """
        indices, probabilities = self.route(hidden)
        flat = hidden.reshape(-1, hidden.shape[-1]).to(torch.bfloat16)
        outputs = flat.new_zeros(indices.shape[0], indices.shape[1], flat.shape[-1])
        for expert in range(self.experts.num_experts):
            tokens, slots = torch.where(indices == expert)
            gate = F.linear(flat[tokens], self.experts.gate_proj[expert].to(flat.dtype))
            up = F.linear(flat[tokens], self.experts.up_proj[expert].to(flat.dtype))
            values = ReferenceSwiGLU.apply(torch.cat((gate, up), dim=-1), False)
            values = F.linear(values, self.experts.down_proj[expert].to(flat.dtype))
            outputs = outputs.index_put((tokens, slots), values)
        return ExpertCombine.apply(outputs, probabilities, self.reference_is_mtp).reshape(hidden.shape).float()

    @staticmethod
    def combine_routed(owner: Any, hidden: torch.Tensor, routed: torch.Tensor) -> torch.Tensor:
        """Expose the routed branch before model-owned shared-expert composition.

        Args:
            owner: MoE module owning the execution.
            hidden: Hidden states.
            routed: Routed expert outputs.
        """
        del owner, hidden
        return routed.float()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run complete MoE semantics using the selected local or EP executor.

        Args:
            hidden_states: Input hidden states.
        """
        hidden = hidden_states.float()
        if self.padding:
            hidden = torch.cat((hidden.new_zeros(1, self.padding, hidden.shape[-1]), hidden), dim=1)
        routed = self.ep_compute(hidden)
        return routed[:, self.padding:] + self.shared_experts(hidden_states).float()


class JTDeepseekV3Decoder(DeepseekV32DecoderLayer):
    """Preserve HF submodules while matching FP32 residual accumulation boundaries."""

    def __init__(self, config: Any, layer_idx: int, *, is_mtp: bool = False) -> None:
        """Construct the complete specialized decoder without a later semantic patch."""
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = JTDeepseekV3Attention(config, layer_idx)
        self.mlp = (JTDeepseekV3MoE(config, is_mtp=is_mtp)
                    if config.mlp_layer_types[layer_idx] == "sparse" else JTDeepseekV3MLP(config))
        self.input_layernorm = JTDeepseekV3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = JTDeepseekV3RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Any = None,
                position_ids: Any = None, past_key_values: Any = None, use_cache: bool = False,
                position_embeddings: Any = None, **kwargs: Any) -> torch.Tensor:
        """Run the original HF children with the reference conversion order.

        Args:
            hidden_states: Input hidden states.
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs.
            past_key_values: Unsupported cached decoding state.
            use_cache: Whether cached decoding is requested.
            position_embeddings: Explicit rotary frequencies.
        """
        if past_key_values is not None or use_cache:
            raise ValueError("DeepSeek V3.2 JT does not support cached decoding")
        residual = hidden_states.float()
        branch, _ = self.self_attn(
            self.input_layernorm(residual).to(torch.bfloat16), attention_mask=attention_mask,
            position_ids=position_ids, position_embeddings=position_embeddings, **kwargs,
        )
        hidden_states = (residual + branch.float()).to(torch.bfloat16)
        residual = hidden_states.float()
        branch = self.mlp(self.post_attention_layernorm(residual).to(torch.bfloat16))
        return (residual + branch.float()).to(torch.bfloat16)


class JTDeepseekV3Model(DeepseekV32Model):
    """Construct specialized decoders directly while retaining the HF model contract."""

    def __init__(self, config: Any) -> None:
        """Build HF embeddings and positional state around complete specialized layers."""
        DeepseekV32PreTrainedModel.__init__(self, config)
        self.padding_idx, self.vocab_size = config.pad_token_id, config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([JTDeepseekV3Decoder(config, index) for index in range(config.num_hidden_layers)])
        self.norm = JTDeepseekV3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = DeepseekV32RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()



def masked_vocab_parallel_loss(logits: torch.Tensor, labels: torch.Tensor,
                               mask: torch.Tensor, *, vocab_size: int) -> torch.Tensor:
    """Compute the masked token mean from the public vocab-parallel CE."""
    mesh = _get_loss_parallel_mesh()
    targets = labels.masked_fill(labels < 0, -100)
    weights = mask.masked_fill(labels < 0, 0)
    values = logits.float().reshape(-1, logits.shape[-1])
    if mesh is None:
        if logits.shape[-1] != vocab_size:
            raise ValueError("Vocabulary shards require the public loss_parallel context")
        token_loss = F.cross_entropy(values, targets.reshape(-1), reduction="none", ignore_index=-100)
    else:
        token_loss = vocab_parallel_cross_entropy_local(
            values, targets.reshape(-1), vocab_size=vocab_size, mesh=mesh,
            ignore_index=-100, reduction="none")
    weighted_loss = token_loss.reshape_as(weights) * weights
    return weighted_loss.sum() / (weights.sum() + 1e-8)


class JTDeepseekV3ForCausalLM(DeepseekV32ForCausalLM):
    """Reuse HF construction and children; override the reference training orchestration."""

    config_class = DeepseekV32Config

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize canonical routed-expert parameters."""
        super()._init_weights(module)
        if isinstance(module, JTDeepseekV3Experts):
            std = self.config.initializer_range
            nn.init.normal_(module.gate_proj, mean=0.0, std=std)
            nn.init.normal_(module.up_proj, mean=0.0, std=std)
            nn.init.normal_(module.down_proj, mean=0.0, std=std)

    def __init__(self, config: Any) -> None:
        """Construct the HF skeleton with complete, unconditional JT adapters.

        Args:
            config: HF configuration carrying the explicit JT contract.
        """
        DeepseekV32PreTrainedModel.__init__(self, config)
        if config.hidden_act != "silu":
            raise ValueError("JT fused experts require hidden_act='silu'")
        if config.n_group != 1 or config.topk_group != 1:
            raise ValueError("JT routing currently requires n_group=topk_group=1")
        if config.rope_parameters["rope_type"] != "default":
            raise ValueError("JT rotary computation currently requires rope_type='default'")
        self.model = JTDeepseekV3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        mtp_config = copy.deepcopy(config)
        depth = config.num_nextn_predict_layers
        mtp_config.mlp_layer_types = ["sparse"] * depth
        self.mtp = DeepseekV3MTP(
            hidden_size=config.hidden_size, num_layers=depth,
            decoder_factory=lambda index: JTDeepseekV3Decoder(mtp_config, index, is_mtp=True),
            norm_factory=lambda size: JTDeepseekV3RMSNorm(size, eps=config.rms_norm_eps),
            output_norm_factory=lambda size: JTDeepseekV3RMSNorm(size, eps=config.rms_norm_eps),
        )
        self.loss_group = None
        self._step_loss_metrics = None
        self._metric_micro_batches = 0
        # The monitoring denominator counts trunk MoE layers, excluding MTP.
        moe_layers = sum(isinstance(layer.mlp, JTDeepseekV3MoE) for layer in self.model.layers)
        self._aux_loss_monitor_scale = moe_layers * config.moe_aux_loss_coeff
        self.post_init()

    def forward(self, input_ids: torch.Tensor, shift_labels: torch.Tensor | None = None,
                loss_mask: torch.Tensor | None = None, *, labels: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None,
                use_cache: bool = False) -> CausalLMOutputWithPast:
        """Use the same JT semantics for evaluation and Trainer backward.

        Args:
            input_ids: Unmodified token IDs.
            shift_labels: Already-shifted targets from the public text batch.
            loss_mask: Mask for the pre-shifted targets.
            labels: Public batch bookkeeping field; shift_labels owns supervision.
            position_ids: Public sequence positions used to construct RoPE.
            attention_mask: Must be None; this model builds its full causal attention internally.
            use_cache: Whether cached decoding is requested.
        """
        del labels
        if attention_mask is not None:
            raise ValueError("JT requires attention_mask=None for its full causal sequence")
        if shift_labels is None or loss_mask is None:
            raise ValueError("JT training requires explicit shift_labels and loss_mask")
        if shift_labels.shape != input_ids.shape or loss_mask.shape != input_ids.shape:
            raise ValueError("JT input_ids, shift_labels and loss_mask must have matching shapes")
        if use_cache:
            raise ValueError("JT does not support cached decoding")
        with torch.autocast(input_ids.device.type, dtype=torch.bfloat16, cache_enabled=False):
            losses = self.compute_jt_losses(input_ids, shift_labels, loss_mask, position_ids=position_ids)
        metrics = torch.stack([losses[name].detach() for name in ("lm_loss", "mtp_loss", "aux_loss")])
        self._step_loss_metrics = metrics if self._step_loss_metrics is None else self._step_loss_metrics + metrics
        self._metric_micro_batches += 1
        return CausalLMOutputWithPast(loss=losses["loss"])

    def get_logging_metrics(self) -> dict[str, torch.Tensor]:
        """Consume detached per-microbatch mean losses for the current TP-replicated JT objective.

        MTP and auxiliary values include their configured coefficients. The
        load-balancing monitor removes the coefficient times trunk MoE count.
        Its denominator intentionally excludes MTP, matching the JT callback.
        This is
        observation only: the combined objective remains the sole backward loss.
        The supported JT recipe has DP1/CP1 and one microbatch per optimizer step.
        """
        if self._step_loss_metrics is None:
            return {}
        values = self._step_loss_metrics / self._metric_micro_batches
        self._step_loss_metrics = None
        self._metric_micro_batches = 0
        metrics = dict(zip(("training/lm_loss", "training/mtp_loss", "training/aux_loss"), values.unbind()))
        metrics["training/load_balancing_loss"] = (
            values[2] / self._aux_loss_monitor_scale if self._aux_loss_monitor_scale > 0 else values[2].new_zeros(()))
        return metrics

    def compute_jt_losses(self, input_ids: torch.Tensor, labels: torch.Tensor,
                         loss_mask: torch.Tensor, *, position_ids: torch.Tensor | None = None
                         ) -> dict[str, torch.Tensor]:

        """Compute model-specific LM, MTP and router losses on pre-shifted labels.

        Args:
            input_ids: Unmodified token IDs.
            labels: Already-shifted target token IDs.
            loss_mask: Mask for the pre-shifted targets.
            position_ids: Optional public batch positions for RoPE.
        """
        cfg = self.config
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("JT sequence loss currently requires a two-dimensional batch-one input")
        sequence_length = input_ids.shape[1]
        if labels.shape != input_ids.shape or loss_mask.shape != input_ids.shape:
            raise ValueError("Tokens, pre-shifted labels and loss mask must have identical shapes")
        dim = cfg.qk_rope_head_dim
        inverse = 1.0 / (cfg.rope_parameters["rope_theta"] ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        inverse = torch.from_numpy(inverse.astype(np.float32)).to(input_ids.device)
        if position_ids is None:
            position_ids = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
        if position_ids.shape != input_ids.shape:
            raise ValueError("JT position_ids must match input_ids")
        frequency = position_ids.to(device=input_ids.device, dtype=torch.float32).unsqueeze(-1) * inverse
        frequency = torch.cat((frequency, frequency), dim=-1)
        attention_kwargs = {"position_embeddings": (frequency.cos(), frequency.sin()),
                            "actual_seq_len": (sequence_length,)}
        hidden = self.model.embed_tokens(input_ids).to(torch.bfloat16)
        auxiliary = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for layer in self.model.layers:
            hidden = layer(hidden, **attention_kwargs)
            if hasattr(layer.mlp, "auxiliary_loss"):
                auxiliary = auxiliary + layer.mlp.auxiliary_loss
        lm_loss = masked_vocab_parallel_loss(
            self.lm_head(self.model.norm(hidden)), labels, loss_mask,
            vocab_size=self.config.vocab_size)
        mtp_output = self.mtp(
            hidden, input_ids, embedding=self.model.embed_tokens, head=self.lm_head,
            labels=labels, loss_mask=loss_mask, loss_factor=cfg.mtp_loss_factor,
            decoder_kwargs=attention_kwargs,
            loss_fn=functools.partial(masked_vocab_parallel_loss, vocab_size=self.config.vocab_size),
            auxiliary_loss=auxiliary, auxiliary_fn=lambda decoder: decoder.mlp.auxiliary_loss,
        )
        losses = {"loss": (lm_loss + auxiliary) + mtp_loss, "lm_loss": lm_loss,
                  "mtp_loss": mtp_loss, "aux_loss": auxiliary}
        if os.getenv("JT_LOSS_TRACE") == "1":
            rank_logger = getattr(logger, "info_rank0", logger.info)
            rank_logger(
                "[JT_RAW_LOSS] lm=%.9e mtp=%.9e aux=%.9e total=%.9e",
                *(float(losses[name].detach().float().item())
                  for name in ("lm_loss", "mtp_loss", "aux_loss", "loss")),
            )
        return losses
