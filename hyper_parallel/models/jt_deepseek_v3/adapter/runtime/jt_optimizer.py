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

import logging
import math
import os
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.components.optim.builders import Muon
from hyper_parallel.models.jt_deepseek_v3.adapter.distributed.mla_attention import (
    JTDeepseekV3FusedMLAAttention,
)
logger = logging.getLogger(__name__)


def _vector_trace_category(parameter_name: str, parameter: torch.Tensor) -> str | None:
    """Classify matrix parameters for cross-branch first-step comparison."""
    if parameter.ndim < 2:
        return None
    if ".self_attn." in parameter_name:
        for projection in ("q_a_proj", "kv_a_proj_with_mqa", "linear_qkv",
                           "q_b_proj", "kv_b_proj", "o_proj"):
            if f".{projection}." in parameter_name:
                return f"attention/{projection}"
    if ".shared_experts." in parameter_name:
        return "shared_expert"
    if ".mlp.experts." in parameter_name:
        return "routed_expert"
    if ".mlp." in parameter_name:
        return "mlp"
    return "other_matrix"


def _vector_trace_local(value: torch.Tensor) -> torch.Tensor:
    """Read local storage from a DTensor without changing the parameter."""
    to_local = getattr(value, "to_local", None)
    return to_local() if callable(to_local) else value


def _vector_trace_before_step(
        model: torch.nn.Module,
        optimizer: Any,
        args: tuple,
        kwargs: dict,
) -> None:
    """Capture first-step gradients and parameter snapshots."""
    del optimizer, args, kwargs
    if getattr(model, "_jt_vector_trace_started", False):
        return
    parameters = {}
    gradients = {}
    snapshots = {}
    categories = {}
    for name, parameter in model.named_parameters():
        category = _vector_trace_category(name, parameter)
        if category is None or parameter.grad is None:
            continue
        parameters[name] = parameter
        categories[name] = category
        gradients[name] = _vector_trace_local(parameter.grad.detach()).cpu().clone()
        snapshots[name] = _vector_trace_local(parameter.detach()).clone()
    model._jt_vector_trace_started = True
    model._jt_vector_trace_parameters = parameters
    model._jt_vector_trace_categories = categories
    model._jt_vector_trace_gradients = gradients
    model._jt_vector_trace_snapshots = snapshots


@torch.no_grad()
def _vector_trace_after_muon(
        model: torch.nn.Module,
        optimizer: Any,
        args: tuple,
        kwargs: dict,
) -> None:
    """Persist first-step gradients and pre-post-hook Muon updates."""
    del optimizer, args, kwargs
    output_path = os.getenv("JT_MUON_VECTOR_TRACE_PATH")
    snapshots = getattr(model, "_jt_vector_trace_snapshots", {})
    parameters = getattr(model, "_jt_vector_trace_parameters", {})
    if not output_path or not snapshots:
        return
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    payload = {
        "stage": "muon",
        "parameters": {},
    }
    for name, before in snapshots.items():
        current = _vector_trace_local(parameters[name].detach())
        payload["parameters"][name] = {
            "category": model._jt_vector_trace_categories[name],
            "gradient": model._jt_vector_trace_gradients[name],
            "muon_delta": (current - before).cpu(),
        }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    rank_logger = getattr(logger, "info_rank0", logger.info)
    rank_logger("[JT_MUON_VECTOR_TRACE] wrote %s", str(path))


@torch.no_grad()
def clip_qk(model: torch.nn.Module, threshold: float) -> dict[str, torch.Tensor]:
    """Clip coupled query/key projections and return detached global maxima.

    Args:
        model: JT model with MLA statistics.
        threshold: Positive clipping threshold from the optimizer adapter configuration.
    """
    metrics = {}
    for name, module in model.named_modules():
        if not isinstance(module, JTDeepseekV3FusedMLAAttention):
            continue
        maximum = module.max_logits_val
        observed = maximum.detach().amax().clone()
        dist.all_reduce(observed, op=dist.ReduceOp.MAX, group=model.loss_group)
        metrics[f"optimizer/qkclip_maxlogits/{name}"] = observed
        scale = threshold / maximum.clamp_min(threshold)
        query = module.q_b_proj.weight.view(
            module.num_heads, module.qk_nope_head_dim + module.qk_rope_head_dim, -1)
        query[:, :module.qk_nope_head_dim].mul_(scale.sqrt()[:, None, None])
        query[:, module.qk_nope_head_dim:].mul_(scale[:, None, None])
        key_value = module.kv_b_proj.weight.view(module.num_heads, module.qk_nope_head_dim + module.v_head_dim, -1)
        key_value[:, :module.qk_nope_head_dim].mul_(scale.sqrt()[:, None, None])
        maximum.zero_()
    if metrics:
        metrics["optimizer/qkclip_maxlogits"] = torch.stack(list(metrics.values())).amax()
    return metrics



@torch.no_grad()
def _after_update(model: torch.nn.Module, threshold: float, optimizer: Any, args: tuple, kwargs: dict) -> None:
    """Apply model-owned updates after all public optimizer leaves complete."""
    del optimizer, args, kwargs
    model.jt_optimizer_metrics = clip_qk(model, threshold)
    config = model.config
    if config.moe_router_enable_expert_bias:
        for module in model.modules():
            if getattr(module, "expert_load", None) is not None:
                direction = (1 / config.n_routed_experts - module.expert_load).sign()
                module.gate.e_score_correction_bias.add_(direction, alpha=config.moe_router_bias_update_rate)
                module.expert_load.zero_()


def _take_metrics(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return QK clipping metrics collected after the previous optimizer step."""
    result = model.jt_optimizer_metrics
    model.jt_optimizer_metrics = {}
    return result


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
    core_optimizer = optimizer.chained_optimizers[-1]
    vector_trace_path = os.getenv("JT_MUON_VECTOR_TRACE_PATH")
    if vector_trace_path:
        core_optimizer.register_step_pre_hook(partial(_vector_trace_before_step, model))
        core_optimizer.register_step_post_hook(partial(_vector_trace_after_muon, model))
    core_optimizer.register_step_post_hook(partial(_after_update, model, qk_clip_threshold))
    model.jt_optimizer_metrics = {}
    optimizer.get_logging_metrics = partial(_take_metrics, model)
    return builder
