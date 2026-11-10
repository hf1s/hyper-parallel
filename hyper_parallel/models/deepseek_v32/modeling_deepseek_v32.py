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
"""DeepSeek-V3.2 training model.

``DeepseekV32ForCausalLM`` in Transformers implements inference. The training
model keeps its modules and checkpoint names and adds the DeepSeek-V3 training
objectives with Megatron-Core definitions:

* every MoE router adds the sequence-wise balance loss and counts routed tokens
  for the loss-free expert-bias update;
* multi-token prediction (MTP) runs the extra decoder depth that the released
  checkpoint stores after the last trunk layer.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from torch.nn import functional as F  # pylint: disable=forbidden-backend-import
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.models.deepseek_v32.modeling_deepseek_v32 import (
    DeepseekV32DecoderLayer,
    DeepseekV32ForCausalLM as TransformersDeepseekV32ForCausalLM,
    DeepseekV32MoE,
    DeepseekV32RMSNorm,
    DeepseekV32TopkRouter,
)

from hyper_parallel.components.functional.aux_loss import aux_loss_auto_scale
from hyper_parallel.components.modules.mtp import MTPLayer, build_mtp
from hyper_parallel.core.activation_memory.recompute_state import is_recomputing
from hyper_parallel.core.utils.communication import differentiable_all_reduce, gather_counts_via_all_gather
from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.models.deepseek_v32.adapter.conversion.checkpoint_mapping import (
    register_training_checkpoint_mapping,
)

__all__ = [
    "DeepseekV32Router",
    "DeepseekV32ForCausalLM",
    "DeepseekV32TrainingOutput",
    "bind_parallel_groups",
]


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
    def from_router(cls, router: DeepseekV32TopkRouter, config: Any, **options: Any) -> "DeepseekV32Router":
        """Wrap an existing router, sharing its weight and bias tensors."""
        trained = cls(config, **options)
        trained.weight = router.weight
        trained.e_score_correction_bias = router.e_score_correction_bias
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


@dataclass
class DeepseekV32TrainingOutput(MoeCausalLMOutputWithPast):
    """Causal-LM output with log-only training metrics read by the Trainer."""

    extra_metrics: Optional[dict[str, torch.Tensor]] = None


def _mtp_layer_config(config: Any, layer_idx: int) -> Any:
    """Return a configuration whose per-layer lists also describe ``layer_idx``."""
    layer_config = copy.copy(config)
    for name in ("mlp_layer_types", "layer_types"):
        values = getattr(config, name, None)
        if isinstance(values, (list, tuple)) and len(values) <= layer_idx:
            setattr(layer_config, name, [*values, *([values[-1]] * (layer_idx + 1 - len(values)))])
    return layer_config


class DeepseekV32ForCausalLM(TransformersDeepseekV32ForCausalLM):
    """DeepSeek-V3.2 causal LM with MoE balancing, MTP and DSA indexer training.

    It keeps the released architecture name, so checkpoints it saves load like
    the released one.

    Args:
        config: DeepSeek-V3.2 configuration. ``num_nextn_predict_layers`` sets
            the MTP depth; 0 disables MTP.
        mtp_loss_scaling_factor: Weight of the MTP loss, divided by the depth.
        mtp_checkpoint_first_layer: Checkpoint layer index of the first MTP
            depth. Defaults to ``num_hidden_layers``; set it to the source
            checkpoint's trunk depth when loading a truncated trunk.
        moe_aux_loss_coeff: Weight of the sequence-wise balance loss.
        moe_router_enable_expert_bias: Whether routers update their load bias.
        moe_router_bias_update_rate: Load-bias step per optimizer step.
        dsa_loss_coeff: Weight of the DSA indexer KL loss; 0 keeps the indexer fixed.
        index_rotate_activation: Apply the Hadamard rotation to indexer queries
            and keys before scoring.
        dsa_dense_warm_up: Train the indexer against dense attention.
    """

    _no_split_modules = ["DeepseekV32DecoderLayer", "MTPLayer"]
    # MTP embedding and head are copies of the trunk's; the trunk weights are used.
    _keys_to_ignore_on_load_unexpected = [
        r"model\.layers\.\d+\.embed_tokens\.weight",
        r"model\.layers\.\d+\.shared_head\.head\.weight",
    ]

    def __init__(self, config: Any, *, mtp_loss_scaling_factor: float = 0.3,
                 mtp_checkpoint_first_layer: Optional[int] = None, moe_aux_loss_coeff: float = 0.0,
                 moe_router_enable_expert_bias: bool = False, moe_router_bias_update_rate: float = 1e-3,
                 dsa_loss_coeff: float = 0.0, index_rotate_activation: bool = False,
                 dsa_dense_warm_up: bool = False) -> None:
        """Build the trunk, the MTP depths and the training routers."""
        # The DSA replacement reads its options from the configuration.
        config.dsa_loss_coeff = float(dsa_loss_coeff)
        config.index_rotate_activation = bool(index_rotate_activation)
        config.dsa_dense_warm_up = bool(dsa_dense_warm_up)
        super().__init__(config)
        if mtp_loss_scaling_factor < 0:
            raise ValueError("mtp_loss_scaling_factor must be nonnegative")
        self.mtp_loss_scaling_factor = float(mtp_loss_scaling_factor)
        first_layer = config.num_hidden_layers
        self.mtp = build_mtp(getattr(config, "num_nextn_predict_layers", 0) or None,
                             layer_factory=lambda depth: MTPLayer(
                                 config.hidden_size,
                                 DeepseekV32DecoderLayer(_mtp_layer_config(config, first_layer + depth),
                                                         first_layer + depth),
                                 norm_factory=partial(DeepseekV32RMSNorm, eps=config.rms_norm_eps),
                             ))
        self.mtp_checkpoint_first_layer = (first_layer if mtp_checkpoint_first_layer is None
                                           else int(mtp_checkpoint_first_layer))
        router_options = {
            "aux_loss_coeff": moe_aux_loss_coeff,
            "enable_expert_bias": moe_router_enable_expert_bias,
            "expert_bias_update_rate": moe_router_bias_update_rate,
        }
        for module in list(self.modules()):
            if isinstance(module, DeepseekV32MoE):
                module.gate = DeepseekV32Router.from_router(module.gate, config, **router_options)
        # Context-parallel group of the trunk sequence; bound after sharding.
        self.sequence_group = None
        self._pre_norm_capture: Optional[tuple[nn.Module, Any]] = None
        self._pre_norm_hidden: Optional[torch.Tensor] = None
        self._capturing_pre_norm = False
        register_training_checkpoint_mapping(type(self).__name__, self.mtp_checkpoint_first_layer,
                                             0 if self.mtp is None else len(self.mtp.layers))
        self.post_init()

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> "DeepseekV32ForCausalLM":
        """Construct from a configuration through the shared custom-model builder."""
        return cls._from_config(config, **kwargs)

    def _capture_pre_norm_hidden(self) -> None:
        """Keep the next trunk output entering the final norm, which MTP consumes."""
        self._pre_norm_hidden = None
        self._capturing_pre_norm = True
        norm = self.model.norm
        if self._pre_norm_capture is not None and self._pre_norm_capture[0] is norm:
            return
        if self._pre_norm_capture is not None:
            self._pre_norm_capture[1].remove()

        def keep_input(_module: nn.Module, args: tuple[Any, ...]) -> None:
            """Store the final norm's input for the MTP depths during a capture."""
            if self._capturing_pre_norm:
                self._pre_norm_hidden = args[0]
                self._capturing_pre_norm = False

        self._pre_norm_capture = (norm, norm.register_forward_pre_hook(keep_input))

    def _shift_left(self, tokens: torch.Tensor, pad_value: int) -> torch.Tensor:
        """Shift ``[B, S]`` rows one position left; only the sequence tail is padded.

        Under context parallelism each rank holds a contiguous slice, so the new
        last column comes from the first column of the next rank.
        """
        group = self.sequence_group
        if group is None:
            return F.pad(tokens[:, 1:], (0, 1), value=pad_value)
        rank, size = dist.get_rank(group), dist.get_world_size(group)
        first = gather_counts_via_all_gather(tokens[:, 0].contiguous(), group, size)
        tail = first[rank + 1] if rank + 1 < size else torch.full_like(first[0], pad_value)
        return torch.cat((tokens[:, 1:], tail.unsqueeze(1)), dim=1)

    def _sequence_mean(self, loss_sum: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        """Average a token loss sum over the full sequence across context-parallel ranks."""
        group = self.sequence_group
        if group is None:
            return loss_sum / count.clamp_min(1)
        global_count = count.clone()
        dist.all_reduce(global_count, group=group)
        return differentiable_all_reduce(loss_sum, "sum", group) / global_count.clamp_min(1)

    def _mtp_loss(self, hidden: torch.Tensor, input_ids: torch.Tensor, targets: torch.Tensor,
                  attention_mask: Any, position_ids: torch.Tensor, kwargs: dict[str, Any],
                  ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return the weighted MTP loss and each depth's unweighted loss.

        Depth ``k`` fuses the previous depth's state at position ``i`` with the
        embedding of token ``i + k`` and predicts token ``i + k + 1``. Inputs and
        targets keep the trunk length; the positions shifted past the end use
        token 0 and are excluded from the loss.
        """
        position_embeddings = self.model.rotary_emb(hidden, position_ids=position_ids)
        total = torch.zeros((), device=hidden.device, dtype=torch.float32)
        depth_losses = []
        for layer in self.mtp.layers:
            input_ids = self._shift_left(input_ids, 0)
            targets = self._shift_left(targets, IGNORE_INDEX)
            hidden = layer(hidden, self.model.embed_tokens(input_ids), attention_mask=attention_mask,
                           position_ids=position_ids, position_embeddings=position_embeddings, **kwargs)
            loss_sum = self.loss_function(logits=self.lm_head(hidden), labels=None, vocab_size=self.config.vocab_size,
                                          num_items_in_batch=1, ignore_index=IGNORE_INDEX, shift_labels=targets)
            depth_loss = self._sequence_mean(loss_sum, targets.ne(IGNORE_INDEX).sum())
            depth_losses.append(depth_loss.detach())
            total = total + depth_loss
        return total * (self.mtp_loss_scaling_factor / len(self.mtp.layers)), depth_losses

    def _training_metrics(self) -> dict[str, torch.Tensor]:
        """Return the trunk's mean DSA indexer loss when the indexer is trained."""
        losses = [loss for layer in self.model.layers
                  if (loss := getattr(layer.self_attn, "last_indexer_loss", None)) is not None]
        return {"dsa_indexer_loss": torch.stack(losses).mean()} if losses else {}

    def forward(self, input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None, past_key_values: Any = None,
                inputs_embeds: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None, logits_to_keep: int | torch.Tensor = 0,
                **kwargs: Any) -> MoeCausalLMOutputWithPast:
        """Return the LM loss, plus the weighted MTP loss under ``foundation_loss/mtp``.

        Without labels, or outside training, this is the Transformers forward.
        """
        if labels is None or not self.training:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                   past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
                                   use_cache=use_cache, logits_to_keep=logits_to_keep, **kwargs)
        if self.mtp is not None:
            if input_ids is None:
                raise ValueError("DeepSeek-V3.2 MTP training requires input_ids")
            self._capture_pre_norm_hidden()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                             inputs_embeds=inputs_embeds, use_cache=False, **kwargs)
        logits = self.lm_head(outputs.last_hidden_state)
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        metrics = self._training_metrics()
        if self.mtp is not None:
            hidden, self._pre_norm_hidden = self._pre_norm_hidden, None
            targets = kwargs.get("shift_labels")
            if targets is None:
                targets = self._shift_left(labels, IGNORE_INDEX)
            if position_ids is None:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            mask = create_causal_mask(config=self.config, inputs_embeds=hidden, attention_mask=attention_mask,
                                      past_key_values=None, position_ids=position_ids,
                                      allow_is_causal_skip=False)
            mtp_loss, depth_losses = self._mtp_loss(hidden, input_ids, targets, mask, position_ids,
                                                    {k: v for k, v in kwargs.items() if k != "shift_labels"})
            # A context-parallel slice may hold no trunk target; its LM mean is then undefined.
            loss = torch.where(targets.ne(IGNORE_INDEX).any(), loss, torch.zeros_like(loss))
            loss = {"foundation_loss": loss, "foundation_loss/mtp": mtp_loss}
            metrics.update({f"mtp_{depth}_loss": value for depth, value in enumerate(depth_losses, start=1)})
        return DeepseekV32TrainingOutput(loss=loss, logits=logits, router_logits=None,
                                         extra_metrics=metrics or None)


def bind_parallel_groups(model: nn.Module, mesh: Any) -> None:
    """Bind the process groups the training objectives reduce over.

    Under loss parallelism the Trainer sums replicated gradients over TP ranks,
    while each TP rank computes the full balance and indexer losses; those
    losses are therefore divided by the TP size.

    Args:
        model: Constructed and parallelized DeepSeek-V3.2 training model.
        mesh: Trainer ``MeshContext``; ``None`` leaves single-process groups.
    """
    if mesh is None:
        return
    cp_group = mesh.cp_mesh.get_group() if mesh.cp_size > 1 else None
    tp_group = mesh.device_mesh["tp"].get_group() if mesh.tp_size > 1 else None
    tp_replicas = mesh.tp_size if mesh.loss_parallel else 1
    for module in model.modules():
        if isinstance(module, DeepseekV32ForCausalLM):
            module.sequence_group = cp_group
        elif isinstance(module, DeepseekV32Router):
            module.sequence_partition_group = cp_group
            module.aux_loss_tp_replicas = tp_replicas
        elif hasattr(module, "kl_head_group"):
            module.kl_head_group = tp_group
            module.kl_loss_tp_replicas = tp_replicas
