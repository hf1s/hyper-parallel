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
"""JT model hooks around the public Muon optimizer."""

# This adapter uses the Torch runtime, like the existing model and Trainer modules.
# pylint: disable=forbidden-backend-import

import math
from functools import partial
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.components.optim.builders import Muon
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3MLAAttention


def _value_copies(parameter: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return every tensor that holds ``parameter``'s value.

    With ``optimizer.fp32_main_params`` the leaf optimizers update an FP32
    ``main_param`` that the mixed-precision wrapper copies back into the model
    parameter after the step, so a post-update edit must change both copies.

    Args:
        parameter: Model parameter, possibly carrying an optimizer ``main_param``.

    Returns:
        The distinct FP32 main parameter (if any) followed by the model parameter.
    """
    main_param = getattr(parameter, "main_param", None)
    if main_param is None or main_param is parameter:
        return (parameter,)
    return main_param, parameter


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


def _global_expert_loads(modules: list[torch.nn.Module], group: Any) -> list[torch.Tensor]:
    """Return each MoE module's expert load averaged over the global batch.

    The forward averages loads only over the shards of one sequence. Every replica of
    the expert bias must take the same update, so the batch shards are averaged here.

    Args:
        modules: MoE modules in model order, identical on every rank.
        group: DP+CP process group, or ``None`` when the batch is not partitioned.

    Returns:
        Per-module expert loads averaged over ``group``.
    """
    loads = [module.expert_load for module in modules]
    if group is None or not loads:
        return loads
    stacked = torch.stack(loads)
    dist.all_reduce(stacked, group=group)
    stacked /= dist.get_world_size(group)
    return list(stacked.unbind())


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
        for weight in _value_copies(module.q_b_proj.weight):
            query = weight.view(module.num_heads, module.qk_nope_head_dim + module.qk_rope_head_dim, -1)
            query[:, :module.qk_nope_head_dim].mul_(scale.sqrt()[:, None, None])
            query[:, module.qk_nope_head_dim:].mul_(scale[:, None, None])
        for weight in _value_copies(module.kv_b_proj.weight):
            key_value = weight.view(module.num_heads, module.qk_nope_head_dim + module.v_head_dim, -1)
            key_value[:, :module.qk_nope_head_dim].mul_(scale.sqrt()[:, None, None])
        module.max_logits_val.zero_()


@torch.no_grad()
def _after_update(model: torch.nn.Module, threshold: float, optimizer: Any, args: tuple, kwargs: dict) -> None:
    """Apply model-owned updates after all public optimizer leaves complete."""
    del optimizer, args, kwargs
    clip_qk(model, threshold)
    config = model.config
    if config.moe_router_enable_expert_bias:
        modules = [module for module in model.modules() if getattr(module, "expert_load", None) is not None]
        for module, load in zip(modules, _global_expert_loads(modules, model.expert_load_group)):
            direction = (1 / config.n_routed_experts - load).sign()
            module.gate.e_score_correction_bias.add_(direction, alpha=config.moe_router_bias_update_rate)
            module.expert_load.zero_()




def build_optimizer(*, model: torch.nn.Module, qk_clip_threshold: float, **kwargs: Any) -> Muon:
    """Build public Muon/AdamW and attach the JT-specific post-update hooks.

    Args:
        model: Model whose final FSDP parameter layouts are already prepared.
        qk_clip_threshold: Positive clipping threshold for QK projections.
        **kwargs: Public Muon Builder options from the training recipe.

    Returns:
        The unmodified public Muon Builder.
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
