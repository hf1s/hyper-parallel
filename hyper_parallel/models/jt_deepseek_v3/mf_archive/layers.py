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

"""Named layer implementations for the explicitly selected archived MF recipe."""

# This explicitly selected recipe is Torch/NPU-only.
# pylint: disable=forbidden-backend-import

from typing import Any

import torch
import torch_npu
from torch import nn
from torch.nn import functional as F

from hyper_parallel.core.activation_memory.recompute_state import is_recomputing
from hyper_parallel.models.jt_deepseek_v3 import modeling_jt_deepseek_v3 as jt
from hyper_parallel.models.jt_deepseek_v3.mf_archive.legacy_gemm import head_weight_gradient
from hyper_parallel.models.jt_deepseek_v3.mf_archive.reductions import weighted_sum


class NormResidual(torch.autograd.Function):
    """Round once after joining the residual gradient and RMSNorm input gradient."""

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
        dx, dw = torch.ops.npu.npu_rms_norm_backward(norm_gradient.float(), inputs, weight.float(), rstd)
        return (dx + residual_gradient.float()).to(ctx.input_dtype), dw, None


class RouterRoutedInputPair(torch.autograd.Function):
    """Join FP32 router and BF16 expert derivatives before the shared-branch accumulation."""

    @staticmethod
    def forward(ctx: Any, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose the shared input at the expert and FP32 router boundaries."""
        ctx.input_dtype = hidden.dtype
        return hidden.view_as(hidden), hidden.float()

    @staticmethod
    def backward(ctx: Any, expert_gradient: torch.Tensor, router_gradient: torch.Tensor) -> torch.Tensor:
        """Accumulate the two consumers in FP32 before returning one gradient."""
        return (expert_gradient.float() + router_gradient).to(ctx.input_dtype)


class RoutingNormalization(torch.autograd.Function):
    """Keep original normalized weights while composing the measured derivative paths."""

    @staticmethod
    def forward(ctx: Any, scores: torch.Tensor, indices: torch.Tensor,
                scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize the selected and all-expert score branches independently."""
        selected = scores.gather(1, indices)
        selected_sum = selected.sum(-1, keepdim=True) + 1e-20
        score_sum = scores.sum(-1, keepdim=True) + 1e-20
        ctx.save_for_backward(scores, indices, selected, selected_sum, score_sum)
        ctx.scale = scale
        return selected / selected_sum * scale, scores / score_sum

    @staticmethod
    def backward(ctx: Any, coefficient_gradient: torch.Tensor,
                 auxiliary_gradient: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Differentiate both normalizations using the archive's accumulation order."""
        scores, indices, selected, selected_sum, score_sum = ctx.saved_tensors
        scaled = coefficient_gradient * ctx.scale
        top_numerator = scaled / selected_sum
        top_denominator = (-(selected / selected_sum / selected_sum) * scaled).sum(-1, keepdim=True)
        top_gradient = top_numerator + top_denominator
        selected_gradient = torch.zeros_like(scores).scatter_add_(1, indices, top_gradient)
        auxiliary_denominator = (-(scores / score_sum / score_sum) * auxiliary_gradient).sum(-1, keepdim=True)
        result = (selected_gradient + auxiliary_denominator) + auxiliary_gradient / score_sum
        return result, None, None


class ArithmeticSigmoid(torch.autograd.Function):
    """Isolate the derivative composition while preserving native sigmoid values."""

    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, mode: str) -> torch.Tensor:
        """Retain the native sigmoid output for the chosen derivative schedule."""
        result = values.sigmoid()
        ctx.save_for_backward(result)
        ctx.mode = mode
        return result

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Apply the specified algebraically equivalent derivative ordering."""
        result, = ctx.saved_tensors
        if ctx.mode == 'sub-square':
            value = (result - result * result) * gradient
        elif ctx.mode == 'factor-first':
            value = (result * (1 - result)) * gradient
        elif ctx.mode == 'gradient-first':
            value = (gradient * result) * (1 - result)
        else:
            raise ValueError('Unknown sigmoid derivative mode')
        return value, None


class SharedKeyHalfTree(torch.autograd.Function):
    """Expand token-major ``[B, S, 1, D]`` keys and sum head gradients in FP32."""

    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, heads: int) -> torch.Tensor:
        """Broadcast a singleton head while keeping sequence-major storage."""
        del ctx
        if value.ndim != 4 or value.shape[2] != 1 or heads < 1 or heads & (heads - 1):
            raise ValueError(f'Expected singleton shared head and power-of-two head count: {value.shape=}, {heads=}')
        return value.expand(-1, -1, heads, -1)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Sum head contributions by a balanced FP32 tree."""
        del ctx
        reduced = gradient.float()
        while reduced.shape[2] > 1:
            half = reduced.shape[2] // 2
            reduced = reduced[:, :, :half] + reduced[:, :, half:]
        return reduced.to(gradient.dtype), None


class ArchivedNorm(jt.DeepseekV32RMSNorm):
    """FP32 affine storage and RMS arithmetic, with an explicit output boundary."""

    keep_compute_in_fp32 = True

    def __init__(self, hidden_size: int, eps: float, *, residual: bool = False,
                 output_dtype: torch.dtype | None = None) -> None:
        """Choose the residual gradient join and the output cast boundary."""
        super().__init__(hidden_size, eps)
        self.residual = residual
        self.output_dtype = output_dtype

    def forward(self, hidden_states: torch.Tensor) -> Any:
        """Return the normalized tensor, or the normalization/residual pair."""
        if self.residual:
            return NormResidual.apply(hidden_states, self.weight, self.variance_epsilon)
        value = torch_npu.npu_rms_norm(hidden_states.float(), self.weight.float(),
                                     epsilon=self.variance_epsilon)[0]
        return value.to(self.output_dtype or hidden_states.dtype)


class ArchivedEmbedding(nn.Embedding):
    """Keep the shared main/MTP embedding parameter and gradient in FP32."""

    keep_compute_in_fp32 = True


class _HeadGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Project with the ordinary linear forward operation."""
        ctx.save_for_backward(values, weight)
        return F.linear(values, weight)

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
    """Cast each use separately, then join the shared parameter's gradients in FP32."""

    keep_compute_in_fp32 = True

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Use the archived GEMM route only for this head's weight derivative."""
        return _HeadGradient.apply(values, self.weight.to(values.dtype))


class _SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, gate: torch.Tensor, up: torch.Tensor, rounded_up: bool) -> torch.Tensor:
        """Evaluate fused SwiGLU, retaining the two source projections."""
        pair = torch.cat((gate, up), dim=-1)
        ctx.save_for_backward(pair)
        ctx.rounded_up = rounded_up
        return torch_npu.npu_swiglu(pair, dim=-1)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Preserve the archive's distinct main/MTP up-gradient cast boundaries."""
        pair, = ctx.saved_tensors
        gate, up = pair.chunk(2, dim=-1)
        sigmoid = gate.float().sigmoid()
        silu = F.silu(gate.float())
        derivative = (sigmoid + silu) - silu * sigmoid
        gate_gradient = ((gradient.float() * up.float()) * derivative).to(gate.dtype)
        up_gradient = (gradient * F.silu(gate) if ctx.rounded_up
                       else (gradient.float() * silu).to(up.dtype))
        return gate_gradient, up_gradient, None


class ArchivedMLP(jt.DeepseekV32MLP):
    """Interleaved FC1 with the archive's main/MTP SwiGLU rounding boundaries."""

    def __init__(self, config: Any, intermediate_size: int | None = None, *, mtp: bool = False) -> None:
        """Build independent gate/up parameters and select the derivative recipe."""
        super().__init__(config, intermediate_size=intermediate_size)
        self.mtp = mtp

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Preserve independent parameter ownership while combining their GEMM."""
        weights = torch.stack((self.gate_proj.weight, self.up_proj.weight), dim=1).flatten(0, 1)
        projected = F.linear(inputs, weights)
        activated = _SwiGLU.apply(projected[..., 0::2], projected[..., 1::2], self.mtp)
        return self.down_proj(activated)


class _RouterLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Apply the router's FP32 linear projection."""
        ctx.save_for_backward(values, weight)
        return F.linear(values, weight)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Make the left weight-gradient operand contiguous explicitly."""
        values, weight = ctx.saved_tensors
        return gradient @ weight, gradient.T.contiguous() @ values


class ArchivedRouter(jt.JTDeepseekV3Gate):
    """FP32 router storage, arithmetic and explicit weight-gradient operands."""

    keep_compute_in_fp32 = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits and retain the native consume-once logits contract."""
        values = hidden_states.reshape(-1, self.hidden_dim).float()
        self.router_logits = _RouterLinear.apply(values, self.weight.float())
        return self.router_logits


class _WeightedProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, outputs: torch.Tensor, weights: torch.Tensor, mtp: bool) -> torch.Tensor:
        """Multiply expert outputs and routing weights in FP32."""
        ctx.save_for_backward(outputs, weights)
        ctx.mtp = mtp
        return outputs.float() * weights.float()

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Reduce live weighted derivatives with the archived hidden-axis schedule."""
        outputs, weights = ctx.saved_tensors
        output_gradient = (gradient.to(outputs.dtype) * weights.to(outputs.dtype)
                           if ctx.mtp else gradient * weights.float())
        product = gradient.to(outputs.dtype) * outputs
        weight_gradient = weighted_sum(product).to(outputs.dtype).to(weights.dtype)
        return output_gradient, weight_gradient, None


def _expert_order(indices: torch.Tensor) -> torch.Tensor:
    token_count, top_k = indices.shape
    if indices.numel() >= 2 ** 24:
        raise ValueError("Archived FP32 expert sorting requires exactly representable integer indices")
    ids = indices.transpose(0, 1).reshape(1, -1).float()
    sorted_ids, first = torch.sort(ids, dim=1)
    second = torch.sort(sorted_ids, dim=1)[1]
    combined = first.gather(1, second).reshape(-1)
    return (combined % token_count) * top_k + combined // token_count


class ArchivedMoE(jt.JTDeepseekV3MoE):
    """Explicit single-card routing, FP32 joins and archived expert token order."""

    def __init__(self, config: Any, *, mtp: bool = False) -> None:
        """Compose the explicit router and shared expert implementations."""
        super().__init__(config)
        self.mtp = mtp
        self.gate = ArchivedRouter(config)
        self.shared_experts = ArchivedMLP(
            config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts, mtp=mtp)

    def route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing and consume the logits through the current JT interface."""
        config, padding = self.config, self.padding
        if config.n_group != 1 or config.topk_group != 1 or not config.norm_topk_prob:
            raise ValueError("The archived recipe requires one normalized routing group")
        if self.auxiliary_loss_group is not None:
            raise ValueError("The archived routing recipe is single-card only")
        logits = self.gate(hidden[:, padding:])
        self.gate.pop_router_logits()
        values = logits.view(-1, logits.shape[-1]).float()
        scores = ArithmeticSigmoid.apply(values, 'gradient-first')
        indices = (scores + self.gate.e_score_correction_bias).topk(
            config.num_experts_per_tok, dim=-1, sorted=False)[1]
        selected, normalized = RoutingNormalization.apply(scores, indices, float(config.routed_scaling_factor))
        flat_indices = indices.reshape(-1)
        counts = flat_indices.new_zeros(config.n_routed_experts, dtype=torch.float32).scatter_add_(
            0, flat_indices, flat_indices.new_ones(flat_indices.shape, dtype=torch.float32))
        load = counts / flat_indices.numel()
        auxiliary_loss = (normalized.mean(0) * load).sum() * config.n_routed_experts * config.moe_aux_loss_coeff
        if not is_recomputing():
            self.auxiliary_loss = auxiliary_loss
        self.tokens_per_expert = counts if self.tokens_per_expert is None else self.tokens_per_expert + counts
        if padding:
            pad_ids = torch.arange(padding * config.num_experts_per_tok, device=indices.device)
            pad_ids = pad_ids.reshape(padding, config.num_experts_per_tok) % padding
            indices = torch.cat((pad_ids, indices))
            selected = torch.cat((selected.new_zeros(padding, selected.shape[-1]), selected))
        return indices, selected

    def local_routed_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Keep native grouped experts while explicitly controlling the joins."""
        hidden, router_hidden = RouterRoutedInputPair.apply(hidden)
        indices, probabilities = self.route(router_hidden)
        flat = hidden.reshape(-1, hidden.shape[-1])
        expert_ids = indices.reshape(-1)
        order = _expert_order(indices)
        sources = torch.arange(flat.shape[0], device=flat.device).repeat_interleave(indices.shape[1])[order]
        counts = expert_ids.new_zeros(self.experts.num_experts).scatter_add_(
            0, expert_ids, torch.ones_like(expert_ids))
        outputs = self.experts.forward_expert_major(flat[sources], counts)
        weights = probabilities.reshape(-1)[order].unsqueeze(-1)
        weighted = _WeightedProduct.apply(outputs, weights, self.mtp)
        return torch.zeros_like(flat, dtype=torch.float32).index_add_(
            0, sources, weighted).to(hidden.dtype).reshape(hidden.shape)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Round the routed/shared/residual sum at the enclosing decoder boundary."""
        hidden = hidden_states
        if self.padding:
            hidden = torch.cat((hidden.new_zeros(1, self.padding, hidden.shape[-1]), hidden), dim=1)
        routed = self.ep_compute(hidden)[:, self.padding:]
        return routed.float() + self.shared_experts(hidden_states).float()


class ArchivedDecoder(jt.JTDeepseekV3Decoder):
    """FP32 normalization/residual derivative joins with native attention ownership."""

    def __init__(self, config: Any, layer_idx: int, *, mtp: bool = False, shared_output: bool = False) -> None:
        """Select explicit norms and feed-forward layers within native JT ownership."""
        super().__init__(config, layer_idx)
        self.shared_output = shared_output
        self.mlp = (ArchivedMoE(config, mtp=mtp) if config.mlp_layer_types[layer_idx] == "sparse"
                    else ArchivedMLP(config))
        self.input_layernorm = ArchivedNorm(config.hidden_size, config.rms_norm_eps, residual=True)
        self.post_attention_layernorm = ArchivedNorm(config.hidden_size, config.rms_norm_eps, residual=True)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Any = None, position_ids: Any = None,
                past_key_values: Any = None, use_cache: bool = False, position_embeddings: Any = None,
                **kwargs: Any) -> torch.Tensor:
        """Join branch derivatives before the activation-dtype cast."""
        if past_key_values is not None or use_cache:
            raise ValueError("The archived training recipe does not support a KV cache")
        normalized, residual = self.input_layernorm(hidden_states)
        branch, _ = self.self_attn(normalized, attention_mask=attention_mask, position_ids=position_ids,
                                   position_embeddings=position_embeddings, **kwargs)
        hidden_states = residual + branch
        normalized, residual = self.post_attention_layernorm(hidden_states)
        result = (residual + self.mlp(normalized)).to(hidden_states.dtype)
        return result.float() if self.shared_output else result


class _SplitQKV(nn.Linear):
    def __init__(self, original: nn.Linear, q_rows: int) -> None:
        """Reuse the original fused parameter without copying or renaming its state."""
        super().__init__(original.in_features, original.out_features, bias=False)
        self.weight = original.weight
        self.q_rows = q_rows

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Run the two original projection GEMMs and restore the fused output layout."""
        return torch.cat((F.linear(values, self.weight[:self.q_rows]),
                          F.linear(values, self.weight[self.q_rows:])), dim=-1)


class ArchivedMLAAttention(jt.JTDeepseekV3MLAAttention):
    """Native JT MLA with split projection GEMMs and the archived shared-key derivative."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "", context: Any = None) -> None:
        """Compose the archived projections with the native packed attention kernel."""
        super().__init__(module=module, module_fqn=module_fqn, context=context)
        self.linear_qkv = _SplitQKV(self.linear_qkv, self.q_lora_rank)
        self.q_a_layernorm = ArchivedNorm(self.q_lora_rank, module.config.rms_norm_eps)
        self.kv_a_layernorm = ArchivedNorm(self.kv_lora_rank, module.config.rms_norm_eps)

    def _project_attention_inputs(self, hidden_states, position_embeddings, past_key_values):
        if past_key_values is not None or position_embeddings is None:
            raise ValueError("Archived MLA requires RoPE tables and no KV cache")
        query_latent, kv_latent, key_rope = self.project_latent_inputs(hidden_states)
        batch, sequence = query_latent.shape[:2]
        query = self.q_b_proj(query_latent).reshape(batch, sequence, self.num_heads, self.qk_head_dim)
        query_pass, query_rope = query.split((self.qk_nope_head_dim, self.qk_rope_head_dim), dim=-1)
        cos, sin = position_embeddings
        query = torch.cat((query_pass, self.explicit_rotary(query_rope, cos, sin, unsqueeze_dim=2)), dim=-1)
        kv_latent = kv_latent.reshape(batch, 1, sequence, self.kv_lora_rank)
        kv_states = self.kv_b_proj(kv_latent).view(
            batch, sequence, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        key_pass, value = kv_states.split((self.qk_nope_head_dim, self.v_head_dim), dim=-1)
        key_rope = self.explicit_rotary(key_rope.unsqueeze(2), cos, sin, unsqueeze_dim=2)
        key = torch.cat((key_pass, SharedKeyHalfTree.apply(key_rope, self.num_heads)), dim=-1)
        return query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
