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
"""DeepSeek-V3.2 DSA attention with the indexer training options.

``DeepseekV32DSAAttention`` implements the released attention. The training
model adds, on the sparse path:

* ``index_rotate_activation``: the orthonormal Hadamard rotation that
  DeepSeek-V3.2 and MindSpeed-LLM apply to the indexer query and key after RoPE;
* the indexer KL teacher over all attention heads when tensor parallelism shards
  them, with the loss divided by the TP ranks whose identical gradients loss
  parallelism sums;
* ``last_indexer_loss``, the KL value of the latest forward, for logging.

The dense warm-up path keeps the released behaviour; its indexer loss is not
logged and rejects tensor-parallel heads.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Optional

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from torch.nn import functional as F  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.functional import aux_loss_auto_scale, dsa_kl_loss
from hyper_parallel.components.modules.dsa_attention import DeepseekV32DSAAttention


@lru_cache(maxsize=8)
def _hadamard_matrix(size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return the Sylvester-ordered ``size`` x ``size`` Hadamard matrix."""
    matrix = torch.ones(1, 1, dtype=torch.float32)
    while matrix.shape[0] < size:
        matrix = torch.cat((torch.cat((matrix, matrix), dim=1), torch.cat((matrix, -matrix), dim=1)), dim=0)
    return matrix.to(dtype=dtype, device=device)


def rotate_activation(states: torch.Tensor) -> torch.Tensor:
    """Apply the orthonormal Hadamard rotation over the last dimension."""
    size = states.shape[-1]
    if size & (size - 1):
        raise ValueError(f"Hadamard rotation requires a power-of-two width, got {size}")
    matrix = _hadamard_matrix(size, states.dtype, states.device)
    return F.linear(states, matrix) * size**-0.5  # pylint: disable=not-callable


def _gather_heads(tensor: torch.Tensor, dim: int, group: Any) -> torch.Tensor:
    """Concatenate tensor-parallel head shards along ``dim`` in rank order."""
    shards = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(shards, tensor.contiguous(), group=group)
    return torch.cat(shards, dim=dim)


class DeepseekV32TrainingDSAAttention(DeepseekV32DSAAttention):
    """``DeepseekV32DSAAttention`` with the indexer options of the training model."""

    def __init__(
        self,
        *,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Build the released replacement, then read the indexer training options."""
        super().__init__(module=module, module_fqn=module_fqn, context=context)
        self.index_rotate_activation = bool(getattr(self.config, "index_rotate_activation", False))
        # Group of the tensor-parallel attention-head shards and the TP ranks whose
        # identical KL gradients the Trainer sums; both are bound after sharding.
        self.kl_head_group = None
        self.kl_loss_tp_replicas = 1
        self.last_indexer_loss: Optional[torch.Tensor] = None

    def _project_index_states(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the indexer states, rotating the query and key when enabled."""
        index_query, index_key, merge_weight = super()._project_index_states(
            hidden_states, q_resid, position_embeddings,
        )
        if self.index_rotate_activation:
            index_query, index_key = rotate_activation(index_query), rotate_activation(index_key)
        return index_query, index_key, merge_weight

    def _indexer_loss_enabled(self) -> bool:
        """Return whether this forward trains the indexer."""
        return bool(self.training and not self.freeze_dsa and self.dsa_loss_coeff)

    def _kl_head_group(self) -> Any:
        """Return the group holding the other heads of the KL teacher when TP shards them."""
        full_heads = getattr(self, "_hp_full_head_counts", {}).get("num_heads")
        if full_heads is None or full_heads == self.num_heads:
            return None
        if self.kl_head_group is None:
            raise RuntimeError(
                "Tensor parallelism shards the attention heads; the DSA indexer loss "
                "needs kl_head_group set to the tensor-parallel group"
            )
        return self.kl_head_group

    def check_dense_indexer_loss(self) -> None:
        """Prepare a dense warm-up forward, which neither logs nor gathers the KL teacher."""
        self.last_indexer_loss = None
        if self._indexer_loss_enabled() and self._kl_head_group() is not None:
            raise NotImplementedError(
                "The dense warm-up indexer loss does not gather tensor-parallel attention heads"
            )

    def _compute_attention_output(self, hidden_states, position_embeddings, actual_seq_len, kwargs):
        """Run the selected dense or sparse stage."""
        if self.dsa_dense_warm_up:
            self.check_dense_indexer_loss()
        return super()._compute_attention_output(hidden_states, position_embeddings, actual_seq_len, kwargs)

    def _apply_auxiliary_loss(
        self,
        attn_output: torch.Tensor,
        index_states: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attention_states: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        topk_indices: torch.Tensor,
        softmax_stats: tuple[torch.Tensor, torch.Tensor],
        actual_q_len: torch.Tensor,
        actual_kv_len: torch.Tensor,
    ) -> torch.Tensor:
        """Attach the DSA KL auxiliary loss against the attention of all heads."""
        if not self._indexer_loss_enabled():
            self.last_indexer_loss = None
            return attn_output
        query_tnd, key_tnd, q_rot_tnd, k_rot_tnd = (
            tensor.reshape(-1, tensor.shape[2], tensor.shape[3])
            for tensor in attention_states
        )
        group = self._kl_head_group()
        if group is not None:
            query_tnd, q_rot_tnd = (_gather_heads(tensor.detach(), 1, group) for tensor in (query_tnd, q_rot_tnd))
            softmax_stats = tuple(_gather_heads(tensor.detach(), -1, group) for tensor in softmax_stats)
        aux_loss = dsa_kl_loss(  # pylint: disable=not-callable
            *index_states, query_tnd, key_tnd,
            topk_indices, *softmax_stats, q_rot_tnd, k_rot_tnd,
            actual_q_len, actual_kv_len, self.scaling, self.dsa_loss_coeff,
            backend="cann",
        )
        self.last_indexer_loss = aux_loss.detach()
        return aux_loss_auto_scale(attn_output, aux_loss / self.kl_loss_tp_replicas)


__all__ = ["DeepseekV32TrainingDSAAttention", "rotate_activation"]
