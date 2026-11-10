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
"""DeepSeek-V3.2 sequence balancing and expert loads."""

from __future__ import annotations

from typing import Any

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.dtensor.dtensor import DTensor, SkipDTensorDispatch, distribute_tensor
from hyper_parallel.core.utils.communication import differentiable_all_reduce


def sequence_load_balancing_loss(
        scores: torch.Tensor,
        selected_experts: torch.Tensor,
        sequence_ids: torch.Tensor,
        num_sequences: int,
        sequence_partition_groups: tuple[Any, ...] = (),
) -> torch.Tensor:
    """Average ``E * sum_i(fraction_i * mean_probability_i)`` over sequences.

    This is Megatron-Core's ``seq_aux_loss``: the fractions count the experts
    the router actually selected, with the correction bias and the group limit.

    Args:
        scores: Nonnegative, unbiased scores for all experts, shaped ``[T, E]``.
        selected_experts: Selected experts, shaped ``[T, K]``.
        sequence_ids: Sample index of each token, shaped ``[T]``.
        num_sequences: Number of samples in the microbatch.
        sequence_partition_groups: Groups partitioning the same samples' tokens,
            such as the context-parallel group. Probability sums use a
            differentiable SUM all-reduce.

    Returns:
        Unweighted mean over nonempty sequences.
    """
    num_tokens, num_experts = scores.shape
    top_k = selected_experts.shape[1]
    scores = scores.float()
    probabilities = scores / scores.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    indices = sequence_ids.unsqueeze(-1) * num_experts + selected_experts
    counts = scores.new_zeros(num_sequences * num_experts).scatter_add(
        0, indices.reshape(-1), scores.new_ones(num_tokens * top_k),
    ).view(num_sequences, num_experts)
    probability_sum = scores.new_zeros(num_sequences, num_experts).index_add(0, sequence_ids, probabilities)
    for group in sequence_partition_groups:
        dist.all_reduce(counts, group=group)
        probability_sum = differentiable_all_reduce(probability_sum, "sum", group)
    assignments = counts.sum(dim=-1, keepdim=True)
    token_count = (assignments / top_k).clamp_min(1.0)
    losses = num_experts * ((counts / assignments.clamp_min(1.0)) * (probability_sum / token_count)).sum(-1)
    return losses.sum() / (assignments > 0).sum().clamp_min(1)


class _AccumulateRouterLoad(torch.autograd.Function):
    """Count a training microbatch once, even when its forward is recomputed."""

    @staticmethod
    def forward(ctx: Any, weights: torch.Tensor, counts: torch.Tensor, accumulator: torch.Tensor) -> torch.Tensor:
        """Keep the forward unchanged; only a consumed backward records loads.

        Args:
            ctx: Autograd context holding the accumulator reference.
            weights: Routing-weight gradient carrier.
            counts: Immutable local microbatch counts.
            accumulator: Integer step-level expert counts.
        """
        ctx.save_for_backward(counts)
        ctx.accumulator = accumulator
        return weights

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Accumulate integer counts without retaining the model graph.

        Args:
            ctx: Context saved by forward.
            gradient: Unchanged gradient for the routing weights.
        """
        with SkipDTensorDispatch(), torch.no_grad():
            ctx.accumulator.add_(ctx.saved_tensors[0])
        return gradient, None, None


def accumulate_router_load(
        weights: torch.Tensor,
        selected_experts: torch.Tensor,
        accumulator: torch.Tensor,
) -> torch.Tensor:
    """Attach the expert-load accumulation to the router's backward path.

    Args:
        weights: Routing weights used by expert combination.
        selected_experts: Actual selected indices, shaped ``[T, K]``.
        accumulator: Integer ``[E]`` step counter.
    """
    flat = selected_experts.reshape(-1)
    counts = torch.zeros_like(accumulator).scatter_add(0, flat, torch.ones_like(flat, dtype=accumulator.dtype))
    if torch.is_grad_enabled() and not weights.requires_grad:
        weights = weights.detach().requires_grad_()
    return _AccumulateRouterLoad.apply(weights, counts, accumulator)


@torch.no_grad()
def update_correction_bias(bias: torch.Tensor, counts: torch.Tensor, update_rate: float) -> None:
    """Update the correction bias, including a sharded tensor.

    Counts must already include all DP and token partitions. Bias affects
    selection only.

    Args:
        bias: The router's ``e_score_correction_bias``, possibly sharded.
        counts: Global int64 expert assignment counts.
        update_rate: Nonnegative step size for the sign update.
    """
    num_experts = counts.shape[-1]
    total = counts.sum(dim=-1, keepdim=True)
    # Compare in integer arithmetic so rounding cannot erase small load
    # differences before the sign update.
    direction = (total - counts * num_experts).sign()
    delta = (update_rate * direction.float()).to(device=bias.device, dtype=bias.dtype)
    if isinstance(bias, DTensor):
        delta = distribute_tensor(delta, bias.device_mesh, bias.placements)
    bias.add_(delta)


__all__ = ["accumulate_router_load", "sequence_load_balancing_loss", "update_correction_bias"]
