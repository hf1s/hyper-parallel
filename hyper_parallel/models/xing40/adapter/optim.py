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
"""Muon for Xing4.0: MindFormers matrix splits and QK clipping after each step."""

# This adapter uses the Torch runtime, like the existing optimizer builders.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.components.optim.builders import Muon
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.core.optimizer.muon import NSInputTransform

__all__ = ["build_optimizer", "clip_qk"]


def build_optimizer(*, model: nn.Module, muon_config: dict[str, Any], adamw_config: dict[str, Any],
                    qk_clip_threshold: float = 0.0) -> Muon:
    """Build Muon with the MindFormers Newton-Schulz block splits and AdamW decay groups.

    As in MindFormers, AdamW does not decay one-dimensional parameters (norm weights, mHC
    gates and biases) or biases.

    Args:
        model: Model whose parameters are optimized.
        muon_config: Muon hyperparameters.
        adamw_config: Hyperparameters of the AdamW fallback.
        qk_clip_threshold: Maximum attention logit enforced by QK clipping after every
            optimizer step; 0 disables clipping.

    Returns:
        The optimizer builder whose chained optimizer the Trainer steps.
    """
    # The AdamW grouping matches name fragments; full names select exactly the 1-D parameters.
    no_decay = ["bias", *(name for name, parameter in model.named_parameters() if parameter.ndim == 1)]
    # The GroupedExperts replacement stores the expert matrices transposed, [E, H, 2I].
    transposed_experts = frozenset(name for name, module in model.named_modules()
                                   if getattr(module, "is_transposed", False))
    builder = Muon(muon_config={**muon_config, "ns_transform_fn": _ns_transform(model.config, transposed_experts)},
                   adamw_config=dict(adamw_config), model=model, no_decay_params=no_decay)
    if qk_clip_threshold > 0:
        # The MLA modules: the NPU Xing40MLAAttention or, unreplaced, the Transformers attention.
        attention = [module for module in model.modules()
                     if hasattr(module, "q_b_proj") and hasattr(module, "kv_b_proj")]
        chained = builder.get_optimizer()
        step_leaves = chained.step

        def step_and_clip(closure: Any = None) -> None:
            """Step every leaf optimizer, then clip the attention logits before any main-weight copy-back."""
            step_leaves(closure)
            clip_qk(attention, qk_clip_threshold)

        # The Trainer keeps this chained optimizer; optimizer-state swap replaces its leaves, not it.
        chained.step = step_and_clip
    return builder


def _periodic_split(matrix: torch.Tensor, first: int, second: int) -> list[torch.Tensor]:
    """Split per-head ``[first; second]`` row blocks into the two stacked matrices."""
    heads = matrix.T.reshape(matrix.shape[1], -1, first + second)
    return [heads[..., :first].reshape(matrix.shape[1], -1).T, heads[..., first:].reshape(matrix.shape[1], -1).T]


def _periodic_merge(parts: list[torch.Tensor], first: int, second: int) -> torch.Tensor:
    """Invert :func:`_periodic_split`."""
    columns = parts[0].shape[1]
    heads = torch.cat((parts[0].T.reshape(columns, -1, first), parts[1].T.reshape(columns, -1, second)), dim=-1)
    return heads.reshape(columns, -1).T


def _ns_transform(config: Any, transposed_experts: frozenset[str] = frozenset()
                  ) -> Callable[[str, torch.Tensor], NSInputTransform | None]:
    """Return the MindFormers Muon schema, which orthogonalizes these blocks separately.

    * ``linear_qkv``: the query latent, KV latent and shared RoPE-key row blocks; the
      unfused Transformers layout keeps ``q_a_proj`` whole and splits
      ``kv_a_proj_with_mqa`` into its KV latent and RoPE-key rows;
    * ``q_b_proj``/``kv_b_proj``: the per-head no-RoPE rows and the RoPE or value rows;
    * ``experts.gate_up_proj``: each expert's gate and up halves, along dimension 1 of
      ``[E, 2I, H]`` or dimension 2 of the transposed ``[E, H, 2I]`` of the modules
      named in ``transposed_experts``.
    """
    nope, rope, value = config.qk_nope_head_dim, config.qk_rope_head_dim, config.v_head_dim
    latent_blocks = (config.q_lora_rank, config.kv_lora_rank, rope)

    def block_split(update: torch.Tensor, sizes: Sequence[int], dim: int) -> NSInputTransform:
        """Split contiguous blocks of ``sizes`` along ``dim``; restore into the matching views."""
        def restore(parts: list[torch.Tensor], output: torch.Tensor) -> None:
            """Copy each block's update into its view of ``output``."""
            for view, part in zip(output.split(list(sizes), dim=dim), parts):
                view.copy_(part)

        return NSInputTransform(tensors=list(update.split(list(sizes), dim=dim)), restore=restore)

    def head_split(update: torch.Tensor, second: int) -> NSInputTransform:
        """Split per-head no-RoPE rows from the ``second`` rows that follow them."""
        return NSInputTransform(
            tensors=_periodic_split(update, nope, second),
            restore=lambda parts, output: output.copy_(_periodic_merge(parts, nope, second)))

    def transform(name: str, update: torch.Tensor) -> NSInputTransform | None:
        """Split one parameter's update for Newton-Schulz, or return None to keep it whole.

        Args:
            name: Parameter FQN.
            update: Full momentum update of the parameter.
        """
        if name.endswith(".self_attn.linear_qkv.weight"):
            return block_split(update, latent_blocks, dim=0)
        if name.endswith(".self_attn.kv_a_proj_with_mqa.weight"):
            return block_split(update, latent_blocks[1:], dim=0)
        if name.endswith(".self_attn.q_b_proj.weight"):
            return head_split(update, rope)
        if name.endswith(".self_attn.kv_b_proj.weight"):
            return head_split(update, value)
        if name.endswith(".mlp.experts.gate_up_proj"):
            dim = 2 if name.removesuffix(".gate_up_proj") in transposed_experts else 1
            return block_split(update, (config.moe_intermediate_size,) * 2, dim=dim)
        return None

    return transform


@torch.no_grad()
def clip_qk(attention: Sequence[nn.Module], threshold: float) -> None:
    """Rescale the query/key projections of heads whose maximum logit exceeded ``threshold``.

    MuonClip: with ``s = threshold / max(max_logit, threshold)`` per head, the query
    no-RoPE rows and the key no-RoPE rows scale by ``sqrt(s)`` and the query RoPE rows
    by ``s``; value rows are unchanged. The maxima are reduced over every rank, which
    all hold every head. The recorded maxima are then cleared.

    Args:
        attention: MLA modules; the NPU fused attention records their ``max_logits`` each step.
        threshold: Maximum attention logit.

    Raises:
        RuntimeError: If a module recorded no maxima, as without the NPU fused attention.
    """
    maxima = [getattr(module, "max_logits", None) for module in attention]
    if any(maximum is None for maximum in maxima):
        raise RuntimeError("QK clipping requires the logit maxima recorded by the NPU fused attention")
    flat = torch.cat(maxima)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(flat, op=dist.ReduceOp.MAX)
    for module, maximum in zip(attention, flat.split([maximum.numel() for maximum in maxima])):
        scale = threshold / maximum.clamp_min(threshold)
        _scale_head_rows(module.q_b_proj.weight, scale, module.qk_nope_head_dim, scale, module.qk_rope_head_dim)
        _scale_head_rows(module.kv_b_proj.weight, scale, module.qk_nope_head_dim, torch.ones_like(scale),
                         module.v_head_dim)
        module.max_logits = None


def _scale_head_rows(weight: torch.Tensor, scale: torch.Tensor, nope_dim: int, tail_scale: torch.Tensor,
                     tail_dim: int) -> None:
    """Scale each head's ``nope_dim`` rows by ``sqrt(scale)`` and its next ``tail_dim`` rows by ``tail_scale``.

    As in MindFormers, only the FP32 main copy is scaled when there is one; the optimizer then
    copies it back into the parameter.
    """
    factors = torch.cat((scale.sqrt()[:, None].expand(-1, nope_dim), tail_scale[:, None].expand(-1, tail_dim)),
                        dim=-1).reshape(-1, 1)
    tensor = getattr(weight, "main_param", None)
    tensor = weight if tensor is None else tensor
    if isinstance(tensor, DTensor):
        local = distribute_tensor(factors.to(tensor.dtype), tensor.device_mesh, tensor.placements).to_local()
        tensor.to_local().mul_(local)
    else:
        tensor.mul_(factors.to(tensor.dtype))
