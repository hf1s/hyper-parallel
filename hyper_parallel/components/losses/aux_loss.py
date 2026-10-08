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
"""MoE auxiliary (load-balancing) loss objectives."""

from typing import Any, Optional

# AutoModels loss components implement the Transformers/PyTorch Trainer API.
# pylint: disable-next=forbidden-backend-import
import torch
# pylint: disable-next=forbidden-backend-import
import torch.distributed as dist


class _ReplicatedGroupMean(torch.autograd.Function):
    """Replicate one group mean while differentiating each local contribution once."""

    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, group: Any) -> torch.Tensor:
        """Average equally weighted contributions across ``group``."""
        ctx.world_size = dist.get_world_size(group)
        result = value.clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
        return result / ctx.world_size

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Scale the local derivative without summing identical output replicas."""
        return gradient / ctx.world_size, None


def _replicated_group_mean(value: torch.Tensor, group: Any = None) -> torch.Tensor:
    """Reduce a partitioned objective; absent an explicit group, keep it local.

    Contributions must be equally weighted, with the same upstream derivative
    on all ranks. This does not implement DDP averaging or independently
    consumed all-reduce outputs; uneven token partitions need explicit weights.
    """
    return value if group is None else _ReplicatedGroupMean.apply(value, group)


def calculate_seq_aux_loss(
    scores: torch.Tensor,
    selected_experts: torch.Tensor,
    *,
    coeff: float,
    sequence_partition_group: Optional[Any] = None,
) -> torch.Tensor:
    """DeepSeek-V3 complementary sequence-wise aux loss ``coeff * E * sum_i(f_i * P_i)``.

    ``f_i`` is the fraction of routed slots assigned to expert ``i``, averaged
    over ``sequence_partition_group``; ``P_i`` is the mean over local tokens of
    each token's affinities normalized across all ``E`` experts. The returned
    value is the mean over the group and every rank differentiates only its own
    tokens. Unlike the top-k-weight load-balance loss of
    :class:`~hyper_parallel.components.modules.moe.MoE`, ``P_i`` uses the
    affinities of every expert.

    Args:
        scores: Router affinities of all experts, ``[tokens, num_experts]``,
            for example sigmoid scores before top-k selection.
        selected_experts: Selected expert indices, ``[tokens, top_k]``.
        coeff: Aux loss coefficient.
        sequence_partition_group: Optional process group spanning the
            sequence-partition dimension, whose ranks hold different token
            shards of the same sequence; ``None`` keeps the statistics local.

    Returns:
        The 0-d aux loss.
    """
    num_experts = scores.shape[-1]
    load = torch.bincount(selected_experts.flatten(), minlength=num_experts)
    load = load / selected_experts.numel()
    if sequence_partition_group is not None:
        dist.all_reduce(load, group=sequence_partition_group)
        load = load / dist.get_world_size(sequence_partition_group)
    normalized = scores / (scores.sum(-1, keepdim=True) + 1e-20)
    loss = (normalized.mean(0) * load).sum() * num_experts * coeff
    return _replicated_group_mean(loss, sequence_partition_group)


__all__ = ["calculate_seq_aux_loss"]
