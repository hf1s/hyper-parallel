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

"""Opt-in archived Dense/MLP/Router/MoE arithmetic for JT DeepSeek-V3.

This module is intentionally not wired into the default JT model.  Callers that
need checkpoint-compatibility arithmetic select these classes explicitly.
"""

# This explicitly selected recipe is Torch/NPU-only.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from hyper_parallel.models.jt_deepseek_v3 import modeling_jt_deepseek_v3 as jt
from hyper_parallel.models.jt_deepseek_v3.archived_reductions import weighted_sum
from hyper_parallel.core.activation_memory.recompute_state import is_recomputing


class RouterRoutedInputPair(torch.autograd.Function):
    """Join FP32 router and activation-dtype expert derivatives."""

    @staticmethod
    def forward(ctx: Any, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the expert input in its own dtype and an FP32 copy for the router."""
        ctx.input_dtype = hidden.dtype
        return hidden.view_as(hidden), hidden.float()

    @staticmethod
    def backward(ctx: Any, expert_gradient: torch.Tensor, router_gradient: torch.Tensor) -> torch.Tensor:
        """Add the two consumers' derivatives in FP32, then round once to the input dtype."""
        return (expert_gradient.float() + router_gradient).to(ctx.input_dtype)


class RoutingNormalization(torch.autograd.Function):
    """Normalize selected routing scores and the all-expert auxiliary branch."""

    @staticmethod
    def forward(ctx: Any, scores: torch.Tensor, indices: torch.Tensor,
                scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the selected scores normalized and scaled by ``scale``, and all scores normalized."""
        selected = scores.gather(1, indices)
        selected_sum = selected.sum(-1, keepdim=True) + 1e-20
        score_sum = scores.sum(-1, keepdim=True) + 1e-20
        ctx.save_for_backward(scores, indices, selected, selected_sum, score_sum)
        ctx.scale = scale
        return selected / selected_sum * scale, scores / score_sum

    @staticmethod
    def backward(ctx: Any, coefficient_gradient: torch.Tensor,
                 auxiliary_gradient: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Differentiate both normalizations in the archive's accumulation order."""
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
    """Sigmoid whose backward follows the archived gradient-first schedule."""

    @staticmethod
    def forward(ctx: Any, values: torch.Tensor) -> torch.Tensor:
        """Return the native sigmoid and retain it for the derivative."""
        result = values.sigmoid()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> torch.Tensor:
        """Return ``(gradient * sigmoid) * (1 - sigmoid)``, multiplied in that order."""
        result, = ctx.saved_tensors
        return (gradient * result) * (1 - result)


class _SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, gate: torch.Tensor, up: torch.Tensor, rounded_up: bool) -> torch.Tensor:
        """Evaluate the fused SwiGLU on ``[gate, up]``, retaining both projections."""
        pair = torch.cat((gate, up), dim=-1)
        ctx.save_for_backward(pair)
        ctx.rounded_up = rounded_up
        # The fused kernel is the archive's forward operation on NPU. Like npu_grouped_swiglu, resolve
        # torch_npu at call time so importing this module needs nothing beyond the product modules.
        import torch_npu  # pylint: disable=import-outside-toplevel
        return torch_npu.npu_swiglu(pair, dim=-1)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Differentiate SwiGLU in FP32 with the archive's main/MTP rounding points.

        The gate gradient is rounded once to the input dtype. MTP (``rounded_up``) multiplies the
        gradient by a SiLU rounded to the input dtype; the main path rounds the FP32 product once.
        """
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
    """Dense SwiGLU MLP with archived projection and backward boundaries."""

    def __init__(self, config: Any, intermediate_size: int | None = None, *, mtp: bool = False) -> None:
        """Keep the product gate/up/down parameters; ``mtp`` selects the MTP up-gradient rounding."""
        super().__init__(config, intermediate_size=intermediate_size)
        self.mtp = mtp

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project gate and up in one GEMM over row-interleaved weights, then apply the fused SwiGLU."""
        weights = torch.stack((self.gate_proj.weight, self.up_proj.weight), dim=1).flatten(0, 1)
        projected = F.linear(inputs, weights)  # pylint: disable=not-callable
        activated = _SwiGLU.apply(projected[..., 0::2], projected[..., 1::2], self.mtp)
        return self.down_proj(activated)


class _RouterLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Apply the router's FP32 linear projection."""
        ctx.save_for_backward(values, weight)
        return F.linear(values, weight)  # pylint: disable=not-callable

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Make the left weight-gradient operand contiguous explicitly."""
        values, weight = ctx.saved_tensors
        return gradient @ weight, gradient.T.contiguous() @ values


class ArchivedRouter(jt.JTDeepseekV3Gate):
    """FP32 router storage, arithmetic and explicit weight-gradient operands."""

    keep_compute_in_fp32 = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return the FP32 logits ``[tokens, num_experts]`` and keep them for ``pop_router_logits``."""
        values = hidden_states.reshape(-1, self.hidden_dim).float()
        self.router_logits = _RouterLinear.apply(values, self.weight.float())
        return self.router_logits


class _WeightedProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, outputs: torch.Tensor, weights: torch.Tensor, mtp: bool) -> torch.Tensor:
        """Multiply expert outputs and routing weights in FP32.

        Raises:
            ValueError: If ``outputs`` are not BF16 with hidden width 5120, the only layout the
                backward weight reduction supports.
        """
        if outputs.shape[-1] != 5120 or outputs.dtype != torch.bfloat16:
            raise ValueError("The archived weighted reduction requires BF16 hidden width 5120")
        ctx.save_for_backward(outputs, weights)
        ctx.mtp = mtp
        return outputs.float() * weights.float()

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Differentiate the product with the archive's main/MTP rounding points.

        MTP rounds the gradient and the weights to the output dtype before multiplying them; the
        main path multiplies in FP32. The weight gradient reduces the output-dtype products
        ``gradient * outputs`` with ``weighted_sum`` and rounds the sum to the output dtype.
        """
        outputs, weights = ctx.saved_tensors
        output_gradient = (gradient.to(outputs.dtype) * weights.to(outputs.dtype)
                           if ctx.mtp else gradient * weights.float())
        product = gradient.to(outputs.dtype) * outputs
        weight_gradient = weighted_sum(product).to(outputs.dtype).to(weights.dtype)
        return output_gradient, weight_gradient, None


def _expert_order(indices: torch.Tensor) -> torch.Tensor:
    """Return the archive's expert-major order of the token-major positions ``t * top_k + k``.

    The expert ids are flattened top-k-major (slot ``k`` of every token before slot ``k + 1``),
    cast to FP32 and argsorted twice with ``torch.sort(..., stable=True)``. The result is always
    grouped by ascending expert id; ties within one expert keep the original top-k-major order.

    Both sorts were previously left unstable, matching the archive's compiled graph but leaving the
    within-expert order backend-defined: on CPU, forcing stability changes 8192 of 8208 positions for
    a realistic 4104-token/top-2 batch. A single-factor NPU ablation against the training trajectory
    (issue #252, arm R6: 200 steps from a shared checkpoint, same metrics and bitwise-equal loss curve
    with and without ``stable=True``) found it numerically inert on that backend and configuration.
    This is not a backend coincidence for ``num_experts_per_tok == 2``: every token then contributes
    exactly two terms to the later ``index_add_`` join, and IEEE 754 addition of two operands is
    exactly commutative regardless of which one is added first, so no within-expert tie-break order
    can change that sum. Stabilizing removes a dependency on backend sort behavior that could change
    silently across torch/torch_npu versions, at no measured numerical cost. The commutativity
    argument does not extend to ``num_experts_per_tok >= 3``, where three-or-more-term sums are
    order-sensitive; re-run the ablation before relying on this for other top-k values.

    Raises:
        ValueError: If ``indices`` holds ``2 ** 24`` or more entries, the archive's limit for exact FP32 sorting.
    """
    token_count, top_k = indices.shape
    if indices.numel() >= 2 ** 24:
        raise ValueError("Archived FP32 expert sorting requires exactly representable integer indices")
    ids = indices.transpose(0, 1).reshape(1, -1).float()
    sorted_ids, first = torch.sort(ids, dim=1, stable=True)
    second = torch.sort(sorted_ids, dim=1, stable=True)[1]
    combined = first.gather(1, second).reshape(-1)
    return (combined % token_count) * top_k + combined // token_count


class ArchivedMoE(jt.JTDeepseekV3MoE):
    """Opt-in sigmoid-routed MoE with archived ordering and FP32 joins."""

    def __init__(self, config: Any, *, mtp: bool = False) -> None:
        """Replace the product router and shared experts; ``mtp`` selects the MTP rounding points."""
        super().__init__(config)
        self.mtp = mtp
        self.gate = ArchivedRouter(config)
        self.shared_experts = ArchivedMLP(
            config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts, mtp=mtp)

    def route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select experts for the real tokens, record the auxiliary loss and count routed tokens.

        Args:
            hidden: Hidden states led by ``padding`` pad tokens, ``[1, padding + tokens, hidden]``.

        Returns:
            Expert indices and routing weights, ``[padding + tokens, top_k]``; pad token ``p`` uses
            experts ``(p * top_k + k) % padding`` with zero weight.

        Raises:
            ValueError: If the configuration is not single-group normalized top-k routing, or if
                the auxiliary loss is reduced over a process group.
        """
        config, padding = self.config, self.padding
        if config.n_group != 1 or config.topk_group != 1 or not config.norm_topk_prob:
            raise ValueError("The archived recipe requires one normalized routing group")
        if self.auxiliary_loss_group is not None:
            raise ValueError("The archived routing recipe is single-card only")
        logits = self.gate(hidden[:, padding:])
        self.gate.pop_router_logits()
        values = logits.view(-1, logits.shape[-1]).float()
        scores = ArithmeticSigmoid.apply(values)
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
        """Run the routed experts in the archive's expert order and sum each token's outputs in FP32.

        Args:
            hidden: Pad-prefixed hidden states, ``[1, padding + tokens, hidden]``.

        Returns:
            The weighted routed-expert output, shaped and typed like ``hidden``.
        """
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
        """Join the routed and shared expert outputs in FP32; the caller rounds the result.

        Args:
            hidden_states: Hidden states, ``[1, tokens, hidden]``.

        Returns:
            The FP32 MoE output, ``[1, tokens, hidden]``.
        """
        hidden = hidden_states
        if self.padding:
            hidden = torch.cat((hidden.new_zeros(1, self.padding, hidden.shape[-1]), hidden), dim=1)
        routed = self.ep_compute(hidden)[:, self.padding:]
        return routed.float() + self.shared_experts(hidden_states).float()


__all__ = [
    "ArithmeticSigmoid", "ArchivedMLP", "ArchivedMoE", "ArchivedRouter",
    "RouterRoutedInputPair", "RoutingNormalization",
]
