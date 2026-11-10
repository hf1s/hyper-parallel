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
"""DeepSeek-V3.2 training router: sequence-wise balance loss and expert-load counts.

The Transformers router implements inference only. This router keeps its
expert selection and checkpoint names and adds the Megatron-Core
``seq_aux_loss`` and the counts of the loss-free expert-bias update.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from torch.nn import functional as F  # pylint: disable=forbidden-backend-import
from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32TopkRouter

from hyper_parallel.components.functional.aux_loss import aux_loss_auto_scale
from hyper_parallel.core.activation_memory.recompute_state import is_recomputing
from hyper_parallel.core.utils.communication import differentiable_all_reduce


class DeepseekV32Router(DeepseekV32TopkRouter):
    """Top-k router with the sequence-wise balance loss and expert-load counts.

    Expert selection and gate values match ``DeepseekV32TopkRouter``. In
    training, the balance loss is attached to the returned gate values and the
    tokens routed to each expert are counted. The Trainer's MoE monitor sums the
    counts over data-parallel ranks after each optimizer step and calls
    :meth:`update_expert_bias`.

    Args:
        config: DeepSeek-V3.2 configuration.
        aux_loss_coeff: Weight of the sequence-wise balance loss; 0 disables it.
        enable_expert_bias: Whether ``e_score_correction_bias`` follows the routed load.
        expert_bias_update_rate: Bias step per optimizer step.
    """

    def __init__(self, config: Any, *, aux_loss_coeff: float = 0.0, enable_expert_bias: bool = False,
                 expert_bias_update_rate: float = 1e-3) -> None:
        """Create the router state; the caller transfers the source weights."""
        super().__init__(config)
        if aux_loss_coeff < 0 or expert_bias_update_rate < 0:
            raise ValueError("aux_loss_coeff and expert_bias_update_rate must be nonnegative")
        self.aux_loss_coeff = float(aux_loss_coeff)
        self.enable_expert_bias = bool(enable_expert_bias)
        self.expert_bias_update_rate = float(expert_bias_update_rate)
        # Context-parallel group over which one sequence is split; bound after sharding.
        self.sequence_partition_group = None
        # TP ranks that compute this same loss and whose gradients the Trainer sums.
        self.aux_loss_tp_replicas = 1
        self.last_aux_loss: Optional[torch.Tensor] = None
        self._token_counts: Optional[torch.Tensor] = None

    @classmethod
    def from_router(cls, router: DeepseekV32TopkRouter, **options: Any) -> "DeepseekV32Router":
        """Wrap an existing router, sharing its weight and bias tensors.

        The Transformers router keeps no configuration, so the routing fields
        are read from the router itself.
        """
        config = SimpleNamespace(
            num_experts_per_tok=router.top_k, num_local_experts=router.num_experts,
            hidden_size=router.hidden_dim, routed_scaling_factor=router.routed_scaling_factor,
            n_group=router.num_group, topk_group=router.topk_group, norm_topk_prob=router.norm_topk_prob,
        )
        # The placeholder parameters are replaced below; build them where the source lives.
        with torch.device(router.weight.device):
            trained = cls(config, **options)
        trained.weight = router.weight
        trained.e_score_correction_bias = router.e_score_correction_bias
        trained.train(router.training)
        return trained

    def _select_experts(self, scores: torch.Tensor) -> torch.Tensor:
        """Return the biased, group-limited top-k expert indices."""
        scores_for_choice = scores + self.e_score_correction_bias
        group_scores = (
            scores_for_choice.view(-1, self.num_group, self.num_experts // self.num_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.num_group, self.num_experts // self.num_group)
            .reshape(-1, self.num_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        return torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return router logits, gate values and expert indices for ``[B, S, H]`` tokens."""
        batch_size = hidden_states.shape[0] if hidden_states.ndim == 3 else 1
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = F.linear(  # pylint: disable=not-callable
            hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        topk_indices = self._select_experts(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        if self.training:
            if self.aux_loss_coeff:
                topk_weights = self._attach_balance_loss(topk_weights, scores, topk_indices, batch_size)
            if self.enable_expert_bias and not is_recomputing():
                counts = self.tokens_per_expert
                counts.scatter_add_(0, topk_indices.reshape(-1),
                                    torch.ones_like(topk_indices.reshape(-1), dtype=counts.dtype))
        return router_logits, topk_weights, topk_indices

    def _attach_balance_loss(self, weights: torch.Tensor, scores: torch.Tensor, indices: torch.Tensor,
                             batch_size: int) -> torch.Tensor:
        """Attach ``coeff * mean_b sum_i f_i P_i`` over each sample of the micro-batch.

        ``f_i`` is the fraction of the sample's routed slots on expert ``i``
        times the expert count, from the selection actually used (biased and
        group-limited). ``P_i`` is the sample mean of the scores normalized over
        all experts, without the bias. A context-parallel split sums both
        statistics over the full sequence.
        """
        num_tokens = scores.shape[0]
        sequence_length = num_tokens // batch_size
        probabilities = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
        probability_sum = probabilities.view(batch_size, sequence_length, -1).sum(dim=1)
        routed = torch.zeros_like(scores).scatter_(1, indices, 1.0)
        counts = routed.view(batch_size, sequence_length, -1).sum(dim=1)
        group = self.sequence_partition_group
        if group is not None:
            dist.all_reduce(counts, group=group)
            probability_sum = differentiable_all_reduce(probability_sum, "sum", group)
            sequence_length *= dist.get_world_size(group)
        load = counts / (sequence_length * self.top_k / self.num_experts)
        loss = (load * (probability_sum / sequence_length)).sum(dim=-1).mean()
        self.last_aux_loss = loss.detach()
        return aux_loss_auto_scale(weights, loss * (self.aux_loss_coeff / self.aux_loss_tp_replicas))

    @property
    def tokens_per_expert(self) -> torch.Tensor:
        """Expert loads routed since the last bias update."""
        counts = self._token_counts
        if counts is None or counts.device != self.e_score_correction_bias.device:
            counts = torch.zeros(self.num_experts, dtype=torch.int64, device=self.e_score_correction_bias.device)
            self._token_counts = counts
        return counts

    @torch.no_grad()
    def update_expert_bias(self, lr: float, num_recomputations: int = 1) -> None:
        """Move each bias by ``lr`` toward the mean load and reset the counts.

        Recomputed forwards are not counted, so ``num_recomputations`` is unused.
        """
        del num_recomputations
        counts = self.tokens_per_expert
        offset = (counts.sum() - counts * self.num_experts).sign().to(self.e_score_correction_bias.dtype)
        self.e_score_correction_bias.add_(offset, alpha=lr)
        counts.zero_()


def bind_router_parallel_groups(model: nn.Module, mesh: Any) -> None:
    """Bind the context-parallel group and loss-parallel replicas of every training router.

    Under loss parallelism the Trainer sums replicated gradients over TP ranks,
    while each TP rank computes the full balance loss; the loss is therefore
    divided by the TP size.

    Args:
        model: Sharded DeepSeek-V3.2 model.
        mesh: Trainer ``MeshContext``; ``None`` leaves single-process groups.
    """
    if mesh is None:
        return
    cp_group = mesh.cp_mesh.get_group() if mesh.cp_size > 1 else None
    tp_replicas = mesh.tp_size if mesh.loss_parallel else 1
    for module in model.modules():
        if isinstance(module, DeepseekV32Router):
            module.sequence_partition_group = cp_group
            module.aux_loss_tp_replicas = tp_replicas


__all__ = ["DeepseekV32Router", "bind_router_parallel_groups"]
