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
"""JT DeepSeek-V3 optimizer: public Muon/AdamW with the QK-clip and routing-bias post-step hook."""

# This adapter uses the Torch runtime, like the existing model and Trainer modules.
# pylint: disable=forbidden-backend-import

import math
from functools import partial
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.components.optim.builders import Muon
from hyper_parallel.core.utils.moe_utils import sync_and_update_expert_bias
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (
    JTDeepseekV3MLAAttention,
    JTDeepseekV3MoE,
)


def _replica_maxima(modules: list[JTDeepseekV3MLAAttention], group: Any) -> list[torch.Tensor]:
    """Return each module's per-head QK maxima over every replica of its heads.

    Ranks in ``group`` hold the same attention heads but see different tokens, so
    clipping with a rank-local maximum would rescale the replicas differently.

    Args:
        modules: Attention modules in model order, identical on every rank.
        group: DP+CP process group, or ``None`` when the heads have no replica.

    Returns:
        Per-module maxima reduced with MAX over ``group``.
    """
    maxima = [module.max_logits_val for module in modules]
    if group is None or not maxima:
        return maxima
    flat = torch.cat([maximum.reshape(-1) for maximum in maxima])
    dist.all_reduce(flat, op=dist.ReduceOp.MAX, group=group)
    parts = flat.split([maximum.numel() for maximum in maxima])
    return [part.view_as(maximum) for part, maximum in zip(parts, maxima)]


@torch.no_grad()
def clip_qk(model: torch.nn.Module, threshold: float) -> None:
    """Clip coupled query/key projections after each optimizer update.

    Args:
        model: JT model with MLA statistics and the DP+CP ``qk_clip_group`` of its heads.
        threshold: Positive clipping threshold from the optimizer adapter configuration.
    """
    modules = [module for module in model.modules() if isinstance(module, JTDeepseekV3MLAAttention)]
    for module, maximum in zip(modules, _replica_maxima(modules, model.qk_clip_group)):
        scale = threshold / maximum.clamp_min(threshold)
        query = module.q_b_proj.weight.view(
            module.num_heads, module.qk_nope_head_dim + module.qk_rope_head_dim, -1)
        query[:, :module.qk_nope_head_dim].mul_(scale.sqrt()[:, None, None])
        query[:, module.qk_nope_head_dim:].mul_(scale[:, None, None])
        key_value = module.kv_b_proj.weight.view(module.num_heads, module.qk_nope_head_dim + module.v_head_dim, -1)
        key_value[:, :module.qk_nope_head_dim].mul_(scale.sqrt()[:, None, None])
        module.max_logits_val.zero_()


@torch.no_grad()
def _after_update(model: torch.nn.Module, threshold: float, optimizer: Any, args: tuple, kwargs: dict) -> None:
    """Clip the QK projections and update the routing biases after an optimizer step."""
    del optimizer, args, kwargs
    clip_qk(model, threshold)
    config = model.config
    if config.moe_router_enable_expert_bias:
        for module in model.modules():
            if isinstance(module, JTDeepseekV3MoE):
                sync_and_update_expert_bias(
                    module, lr=config.moe_router_bias_update_rate,
                    tp_group=module.sequence_partition_group, dp_group=model.expert_load_group)


def build_optimizer(*, model: torch.nn.Module, qk_clip_threshold: float, **kwargs: Any) -> Muon:
    """Build the Muon/AdamW optimizer and register the JT post-step hook.

    Args:
        model: Model whose final FSDP parameter layouts are already prepared.
        qk_clip_threshold: Positive clipping threshold for QK projections.
        **kwargs: Muon builder options from the training recipe.

    Returns:
        The Muon builder.

    Raises:
        ValueError: If ``qk_clip_threshold`` is not finite and positive.
    """
    if not math.isfinite(qk_clip_threshold) or qk_clip_threshold <= 0:
        raise ValueError("qk_clip_threshold must be finite and positive")
    muon_config = dict(kwargs["muon_config"])
    builder = Muon(model=model, muon_config=muon_config, **{
        name: value for name, value in kwargs.items() if name != "muon_config"
    })
    optimizer = builder.get_optimizer()
    optimizer.chained_optimizers[-1].register_step_post_hook(partial(_after_update, model, qk_clip_threshold))
    return builder
