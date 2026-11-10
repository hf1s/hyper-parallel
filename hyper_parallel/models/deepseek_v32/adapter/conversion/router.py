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
"""DeepSeek-V3.2 training router replacing the Transformers router."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32TopkRouter

from hyper_parallel.components.functional.aux_loss import aux_loss_auto_scale
from hyper_parallel.models.deepseek_v32.adapter.aux_loss import (
    accumulate_router_load,
    sequence_load_balancing_loss,
    update_correction_bias,
)


def _apply_router_training_state(
        module: nn.Module,
        hidden_states: torch.Tensor,
        scores: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
) -> torch.Tensor:
    """Attach the sequence auxiliary gradient and the deferred expert loads."""
    module.last_aux_loss = None
    if module.training and module.aux_loss_coeff:
        num_sequences = hidden_states.shape[0] if hidden_states.ndim == 3 else 1
        sequence_ids = torch.arange(num_sequences, device=scores.device).repeat_interleave(
            scores.shape[0] // num_sequences,
        )
        aux_loss = sequence_load_balancing_loss(
            scores, indices, sequence_ids, num_sequences, module.sequence_partition_groups,
        )
        module.last_aux_loss = aux_loss.detach()
        weights = aux_loss_auto_scale(weights, aux_loss * (module.aux_loss_coeff / module.aux_loss_tp_replicas))
    if module.training and module.enable_expert_bias:
        weights = accumulate_router_load(weights, indices, module.tokens_per_expert)
    return weights


class DeepseekV32Router(DeepseekV32TopkRouter):
    """``DeepseekV32TopkRouter`` with the sequence-wise balance loss and expert loads.

    Expert selection and gate values are the Transformers router's. In
    training, the balance loss is attached to the gate values and each
    microbatch's expert loads are counted once in backward. The Trainer's MoE
    monitor sums the loads over data-parallel ranks after each optimizer step
    and calls :meth:`update_expert_bias`.

    Args:
        config: DeepSeek-V3.2 routing configuration.
        aux_loss_coeff: Weight of the sequence-wise balance loss; 0 disables it.
        enable_expert_bias: Whether ``e_score_correction_bias`` follows the routed load.
        expert_bias_update_rate: Bias step per optimizer step.
    """

    def __init__(self, config: Any, *, aux_loss_coeff: float = 0.0, enable_expert_bias: bool = False,
                 expert_bias_update_rate: float = 1e-3) -> None:
        """Create the router state; the caller transfers the source weights."""
        super().__init__(config)
        if not math.isfinite(aux_loss_coeff) or aux_loss_coeff < 0:
            raise ValueError("aux_loss_coeff must be finite and nonnegative")
        if not math.isfinite(expert_bias_update_rate) or expert_bias_update_rate < 0:
            raise ValueError("expert_bias_update_rate must be finite and nonnegative")
        self.aux_loss_coeff = float(aux_loss_coeff)
        self.last_aux_loss: torch.Tensor | None = None
        self.enable_expert_bias = bool(enable_expert_bias)
        self.expert_bias_update_rate = float(expert_bias_update_rate)
        # Groups partitioning one microbatch's tokens and the TP ranks whose identical
        # balance-loss gradients the Trainer sums; both are bound after sharding.
        self.sequence_partition_groups: tuple[Any, ...] = ()
        self.aux_loss_tp_replicas = 1
        # The replacement keeps the Transformers router's parameter and buffer names,
        # so the load counter is a plain attribute created on the bias's device.
        self._token_counts: torch.Tensor | None = None

    @property
    def tokens_per_expert(self) -> torch.Tensor:
        """Expert loads accumulated since the last bias update."""
        counts = self._token_counts
        if counts is None or counts.device != self.e_score_correction_bias.device:
            counts = torch.zeros(self.num_experts, dtype=torch.long, device=self.e_score_correction_bias.device)
            self._token_counts = counts
        return counts

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

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return router logits, gate values and expert indices for ``[B, S, H]`` tokens."""
        router_logits, topk_weights, topk_indices = super().forward(hidden_states)
        topk_weights = _apply_router_training_state(
            self, hidden_states, router_logits.sigmoid(), topk_weights, topk_indices,
        )
        return router_logits, topk_weights, topk_indices

    @torch.no_grad()
    def update_expert_bias(self, lr: float | None = None, num_recomputations: int = 1) -> None:
        """Update the selection bias once after the optimizer step.

        The shared monitor synchronizes counts first. Backward counts each
        microbatch once, so checkpoint recomputation needs no correction.

        Args:
            lr: Optional step size, defaulting to this router's configured rate.
            num_recomputations: Shared monitor compatibility argument; unused.
        """
        del num_recomputations
        if lr is None:
            lr = self.expert_bias_update_rate
        update_correction_bias(self.e_score_correction_bias, self.tokens_per_expert, lr)
        self.tokens_per_expert.zero_()


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
    groups = (mesh.cp_mesh.get_group(),) if mesh.cp_size > 1 else ()
    tp_replicas = mesh.tp_size if mesh.loss_parallel else 1
    for module in model.modules():
        if isinstance(module, DeepseekV32Router):
            module.sequence_partition_groups = groups
            module.aux_loss_tp_replicas = tp_replicas


__all__ = ["DeepseekV32Router", "bind_router_parallel_groups"]
