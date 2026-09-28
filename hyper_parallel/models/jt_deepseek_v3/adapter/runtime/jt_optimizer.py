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

import json
import logging
import math
import os
from functools import partial
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.components.optim.builders import Muon
from hyper_parallel.core.optimizer.muon import NSInputTransform
from hyper_parallel.models.jt_deepseek_v3.adapter.distributed.mla_attention import (
    JTDeepseekV3FusedMLAAttention,
)
logger = logging.getLogger(__name__)


def _trace_category(parameter_name: str, parameter: torch.Tensor) -> str | None:
    """Assign first-step optimizer tracing to a semantic matrix family."""
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


def _trace_local(value: torch.Tensor) -> torch.Tensor:
    """Use local storage for DTensor-compatible diagnostic norms."""
    to_local = getattr(value, "to_local", None)
    return to_local() if callable(to_local) else value


def _trace_norm_sq(value: torch.Tensor) -> float:
    """Return a local squared L2 norm without changing training tensors."""
    local = _trace_local(value.detach())
    return float(local.float().square().sum().item())


def _trace_before_step(model: torch.nn.Module, optimizer: Any, args: tuple, kwargs: dict) -> None:
    """Capture first-step gradients and parameter storage before Muon."""
    del optimizer, args, kwargs
    if getattr(model, "_jt_trace_started", False):
        return
    parameters = {}
    snapshots = {}
    gradient_norms: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        category = _trace_category(name, parameter)
        if category is None or parameter.grad is None:
            continue
        parameters[name] = parameter
        snapshots[name] = _trace_local(parameter.detach()).clone()
        gradient_norms[category] = (
            gradient_norms.get(category, 0.0)
            + _trace_norm_sq(parameter.grad)
        )
    model._jt_trace_started = True
    model._jt_trace_parameters = parameters
    model._jt_trace_snapshots = snapshots
    model._jt_trace_gradient_norms = gradient_norms


def _trace_emit(model: torch.nn.Module, stage: str) -> None:
    """Emit first-step gradient/update norms around the QK clip hook."""
    snapshots = getattr(model, "_jt_trace_snapshots", {})
    parameters = getattr(model, "_jt_trace_parameters", {})
    if not snapshots:
        return
    current_by_name = {
        name: _trace_local(parameter.detach())
        for name, parameter in parameters.items()
    }
    if stage == "muon":
        update_norms: dict[str, float] = {}
        after_muon = {}
        for name, before in snapshots.items():
            current = current_by_name[name]
            after_muon[name] = current.clone()
            category = _trace_category(name, parameters[name])
            update = current - before
            update_norms[category] = update_norms.get(category, 0.0) + _trace_norm_sq(update)
        model._jt_trace_after_muon = after_muon
        payload = {
            "stage": stage,
            "gradient_norm": {
                key: math.sqrt(value)
                for key, value in model._jt_trace_gradient_norms.items()
            },
            "muon_delta_norm": {
                key: math.sqrt(value)
                for key, value in update_norms.items()
            },
        }
    else:
        after_muon = getattr(model, "_jt_trace_after_muon", {})
        clip_norms: dict[str, float] = {}
        total_norms: dict[str, float] = {}
        for name, before in snapshots.items():
            category = _trace_category(name, parameters[name])
            current = current_by_name[name]
            clip_delta = current - after_muon[name]
            total_delta = current - before
            clip_norms[category] = clip_norms.get(category, 0.0) + _trace_norm_sq(clip_delta)
            total_norms[category] = total_norms.get(category, 0.0) + _trace_norm_sq(total_delta)
        payload = {
            "stage": stage,
            "qk_clip_delta_norm": {
                key: math.sqrt(value) for key, value in clip_norms.items()
            },
            "total_delta_norm": {
                key: math.sqrt(value) for key, value in total_norms.items()
            },
        }
    rank_logger = getattr(logger, "info_rank0", logger.info)
    rank_logger("[JT_MUON_TRACE] %s", json.dumps(payload, sort_keys=True))


def _trace_after_muon(model: torch.nn.Module, optimizer: Any, args: tuple, kwargs: dict) -> None:
    """Trace the core optimizer update before model-owned postprocessing."""
    del optimizer, args, kwargs
    _trace_emit(model, "muon")


def _trace_after_clip(model: torch.nn.Module, optimizer: Any, args: tuple, kwargs: dict) -> None:
    """Trace the final first-step delta after model-owned postprocessing."""
    del optimizer, args, kwargs
    _trace_emit(model, "post_clip")


def _periodic_muon_transform(
        update: torch.Tensor,
        first_size: int,
        second_size: int,
) -> NSInputTransform:
    """Split interleaved per-head rows into reference Muon matrices."""
    section_size = first_size + second_size
    if update.ndim != 2 or update.shape[0] % section_size:
        raise ValueError(
            "JT Muon periodic split expects a 2D matrix whose row count is "
            f"divisible by {section_size}, got {tuple(update.shape)}"
        )
    blocks = update.shape[0] // section_size
    hidden_size = update.shape[1]
    grouped = update.view(blocks, section_size, hidden_size)
    first = grouped[:, :first_size, :].reshape(-1, hidden_size).contiguous()
    second = grouped[:, first_size:, :].reshape(-1, hidden_size).contiguous()

    def restore(updates: list[torch.Tensor], output: torch.Tensor) -> None:
        """Restore the separately orthogonalized blocks."""
        target = output.view(blocks, section_size, hidden_size)
        target[:, :first_size, :].copy_(updates[0].view(blocks, first_size, hidden_size))
        target[:, first_size:, :].copy_(updates[1].view(blocks, second_size, hidden_size))

    return NSInputTransform(tensors=[first, second], restore=restore)


def _build_jt_mla_ns_transform(config: Any):
    """Restore JT's logical MLA Muon splits on canonical parameter names."""
    kv_lora_rank = int(config.kv_lora_rank)
    qk_rope_head_dim = int(config.qk_rope_head_dim)
    qk_nope_head_dim = int(config.qk_nope_head_dim)
    v_head_dim = int(config.v_head_dim)

    def transform(parameter_name: str, update: torch.Tensor) -> NSInputTransform | None:
        if parameter_name.endswith(".kv_a_proj_with_mqa.weight"):
            expected_rows = kv_lora_rank + qk_rope_head_dim
            if update.ndim != 2 or update.shape[0] != expected_rows:
                raise ValueError(
                    "JT Muon kv_a split expects "
                    f"{expected_rows} rows, got {tuple(update.shape)}"
                )
            kv_update = update[:kv_lora_rank]
            rope_update = update[kv_lora_rank:]

            def restore(updates: list[torch.Tensor], output: torch.Tensor) -> None:
                """Restore latent and rotary updates into the canonical matrix."""
                output[:kv_lora_rank].copy_(updates[0])
                output[kv_lora_rank:].copy_(updates[1])

            return NSInputTransform(
                tensors=[kv_update, rope_update],
                restore=restore,
            )

        if parameter_name.endswith(".q_b_proj.weight"):
            return _periodic_muon_transform(update, qk_nope_head_dim, qk_rope_head_dim)

        if parameter_name.endswith(".kv_b_proj.weight"):
            return _periodic_muon_transform(update, qk_nope_head_dim, v_head_dim)

        return None

    return transform


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
    muon_config["ns_transform_fn"] = _build_jt_mla_ns_transform(model.config)
    builder = Muon(model=model, muon_config=muon_config, **{
        name: value for name, value in kwargs.items() if name != "muon_config"
    })
    optimizer = builder.get_optimizer()
    core_optimizer = optimizer.chained_optimizers[-1]
    trace_enabled = os.getenv("JT_MUON_TRACE_FIRST_STEP") == "1"
    if trace_enabled:
        core_optimizer.register_step_pre_hook(partial(_trace_before_step, model))
        core_optimizer.register_step_post_hook(partial(_trace_after_muon, model))
    core_optimizer.register_step_post_hook(partial(_after_update, model, qk_clip_threshold))
    if trace_enabled:
        core_optimizer.register_step_post_hook(partial(_trace_after_clip, model))
    model.jt_optimizer_metrics = {}
    optimizer.get_logging_metrics = partial(_take_metrics, model)
    return builder
