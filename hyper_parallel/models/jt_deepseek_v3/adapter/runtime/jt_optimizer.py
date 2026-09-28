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

import math
from functools import partial
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.components.optim.builders import Muon
from hyper_parallel.core.optimizer.muon import NSInputTransform
from hyper_parallel.models.jt_deepseek_v3.adapter.distributed.mla_attention import (
    JTDeepseekV3FusedMLAAttention,
)


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


def _reshape_jt_expert_projection(
        parameter_name: str,
        update: torch.Tensor,
) -> list[torch.Tensor]:
    """Match the old gate/up expert Muon matrix orientation."""
    if parameter_name.endswith((".experts.gate_proj.weight", ".experts.up_proj.weight")):
        return [update.mT]
    return [update]


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
    muon_config["reshape_fn"] = _reshape_jt_expert_projection
    builder = Muon(model=model, muon_config=muon_config, **{
        name: value for name, value in kwargs.items() if name != "muon_config"
    })
    optimizer = builder.get_optimizer()
    optimizer.chained_optimizers[-1].register_step_post_hook(partial(_after_update, model, qk_clip_threshold))
    model.jt_optimizer_metrics = {}
    optimizer.get_logging_metrics = partial(_take_metrics, model)
    return builder
