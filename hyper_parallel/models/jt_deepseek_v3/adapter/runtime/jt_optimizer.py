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
from typing import Any, Optional, Union

import torch
import torch.distributed as dist

from hyper_parallel.components.optim.builders import Muon
from hyper_parallel.components.optim.parameter_groups import get_adamw_param_groups, split_muon_adamw_params
from hyper_parallel.core.optimizer import _build_configured_optimizer, _filter_optimizer_config
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.dtensor_compat import detect_dtensor_backend
from hyper_parallel.core.optimizer.muon import Muon as CoreMuon, NSInputTransform
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.core.utils.moe_utils import sync_and_update_expert_bias
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (
    JTDeepseekV3Attention,
    JTDeepseekV3MLAAttention,
    JTDeepseekV3MoE,
)


def _reference_newton_schulz(inputs: torch.Tensor, steps: int) -> torch.Tensor:
    """Reproduce native Graph O1 BF16 NS arithmetic on logical 2D/3D matrices.

    The graph fuses multiply-add expressions in FP32, with BF16 scalar
    coefficients and BF16 matmul boundaries. Only 2D normalization retains
    FP32 intermediates; expert normalization rounds the norm to BF16. Its 2D path
    evaluates ``(c * A) @ A``, while the expert path evaluates ``c * (A @ A)``.
    """
    if inputs.ndim not in (2, 3) or inputs.dtype != torch.bfloat16:
        raise ValueError("reference Newton-Schulz requires logical BF16 matrices of rank 2 or 3")
    transposed = inputs.shape[-2] > inputs.shape[-1]
    value = inputs.mT if transposed else inputs
    normalization_input = value.float() if inputs.ndim == 2 else value
    value = (normalization_input / (normalization_input.norm(dim=(-2, -1), keepdim=True) + 1e-7)).bfloat16()
    # These are the legacy coefficients rounded to BF16 before graph fusion.
    coeff_a, coeff_b, coeff_c = 3.4375, -4.78125, 2.03125
    for _ in range(steps):
        gram = value @ value.mT
        if inputs.ndim == 2:
            polynomial = coeff_b * gram.float() + ((coeff_c * gram) @ gram).float()
        else:
            polynomial = coeff_b * gram.float() + coeff_c * (gram @ gram).float()
        value = (coeff_a * value.float() + (polynomial.bfloat16() @ value).float()).bfloat16()
    return value.mT if transposed else value


class _JTReferenceMuon(CoreMuon):
    """Preserve logical matrix rank only for JT's Graph O1 numerical policy."""

    def _compute_batched_ns_outputs_for_tensors(
            self, tensor_list, ns_steps, ns_variant="asym5", ns_coefficients=None, ns_epsilon=1e-10):
        del ns_variant, ns_coefficients, ns_epsilon
        return [self.zeropower_fn(tensor, steps=ns_steps) for tensor in tensor_list]


class _JTReferenceMuonBuilder:
    """Compose standard AdamW with the JT-local Muon scheduling specialization."""

    def __init__(self, muon_config: dict, adamw_config: dict, model: torch.nn.Module,
                 extra_adamw_name_keywords: Optional[list[str]] = None,
                 no_decay_params: Optional[list[str]] = None) -> None:
        """Reuse shared parameter grouping and compose the JT reference leaf."""
        self.muon_config, self.adamw_config, self.model = muon_config, adamw_config, model
        matrices, others, _, _ = split_muon_adamw_params(model, extra_adamw_name_keywords or ())
        if not matrices:
            raise ValueError("Muon requires at least one eligible matrix parameter")
        adamw_groups, _ = get_adamw_param_groups(
            model, weight_decay=adamw_config.get("adamw_weight_decay", 1e-2),
            no_decay_params=no_decay_params, allowed_param_ids=[id(parameter) for parameter in others])
        detect_dtensor_backend(adamw_groups, matrices)
        optimizers = {}
        # Reuse the factory's key normalization and defaults; only the Muon class varies.
        if adamw_groups:
            optimizers["adamw"] = _build_configured_optimizer(
                "adamw", AdamW, adamw_groups, _filter_optimizer_config("adamw", AdamW, adamw_config))
        optimizers["muon"] = _build_configured_optimizer(
            "muon", _JTReferenceMuon, matrices,
            _filter_optimizer_config("muon", _JTReferenceMuon, muon_config))
        self.optimizer = ChainedOptimizer(model, optimizers, flatten=bool(adamw_groups))

    def get_optimizer(self) -> ChainedOptimizer:
        """Return the standard chained optimizer runtime."""
        return self.optimizer


def _reference_muon_transform(param_name: str, tensor: torch.Tensor, *, config: Any,
                              matched_adamw_rms: float) -> NSInputTransform:
    """Expose reference logical matrices through the public reversible NS interface.

    Packed projections are storage layouts, not single Muon matrices. The
    reference also shares the last logical matrix's scale across each packed
    parameter. This optional compatibility policy leaves the core optimizer's
    default per-matrix scaling unchanged.
    """
    join_dim, transpose, periodic_shape = 0, False, None
    if param_name.endswith("experts.gate_up_proj"):
        parts = list(tensor.transpose(-1, -2).chunk(2, dim=-1))
        join_dim, transpose = -1, True
    elif param_name.endswith("experts.down_proj"):
        parts = [tensor.transpose(-1, -2)]
        transpose = True
    elif param_name.endswith("self_attn.q_b_proj.weight"):
        periodic_shape = (config.num_attention_heads, config.qk_nope_head_dim + config.qk_rope_head_dim, -1)
        pair = tensor.reshape(periodic_shape).split((config.qk_nope_head_dim, config.qk_rope_head_dim), dim=1)
        parts = [part.reshape(-1, tensor.shape[-1]) for part in pair]
    elif param_name.endswith("self_attn.kv_b_proj.weight"):
        periodic_shape = (config.num_attention_heads, config.qk_nope_head_dim + config.v_head_dim, -1)
        pair = tensor.reshape(periodic_shape).split((config.qk_nope_head_dim, config.v_head_dim), dim=1)
        parts = [part.reshape(-1, tensor.shape[-1]) for part in pair]
    elif param_name.endswith("self_attn.kv_a_proj_with_mqa.weight"):
        parts = list(tensor.split((config.kv_lora_rank, config.qk_rope_head_dim), dim=0))
    elif param_name.endswith("self_attn.linear_qkv.weight"):
        parts = list(tensor.split((config.q_lora_rank, config.kv_lora_rank, config.qk_rope_head_dim), dim=0))
    elif param_name.endswith("linear_fc1.weight"):
        parts = list(tensor.chunk(2, dim=0))
    else:
        parts = [tensor]
    scale = math.sqrt(max(parts[-1].shape[-2:])) * matched_adamw_rms

    def restore(updates: list[torch.Tensor], output: torch.Tensor) -> None:
        """Restore projection/expert storage after independent matrix updates."""
        if periodic_shape is not None:
            values = [update.reshape(config.num_attention_heads, -1, tensor.shape[-1]) for update in updates]
            restored = torch.cat(values, dim=1).reshape_as(output)
        else:
            restored = torch.cat(updates, dim=join_dim) if len(updates) > 1 else updates[0]
            if transpose:
                restored = restored.transpose(-1, -2)
        output.copy_(restored * scale)

    return NSInputTransform(tensors=parts, restore=restore)


def _replica_maxima(modules: list[JTDeepseekV3Attention | JTDeepseekV3MLAAttention],
                    group: Any) -> list[torch.Tensor]:
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
    modules = [module for module in model.modules()
               if isinstance(module, (JTDeepseekV3Attention, JTDeepseekV3MLAAttention))]
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
    """Apply model-owned updates after all public optimizer leaves complete."""
    del optimizer, args, kwargs
    clip_qk(model, threshold)
    config = model.config
    if config.moe_router_enable_expert_bias:
        for module in model.modules():
            if isinstance(module, JTDeepseekV3MoE):
                sync_and_update_expert_bias(
                    module, lr=config.moe_router_bias_update_rate,
                    tp_group=module.sequence_partition_group, dp_group=model.expert_load_group)


def build_optimizer(*, model: torch.nn.Module, qk_clip_threshold: float,
                    reference_muon: bool = False, **kwargs: Any) -> Union[Muon, _JTReferenceMuonBuilder]:
    """Build public Muon/AdamW and attach the JT-specific post-update hooks.

    Args:
        model: Model whose final FSDP parameter layouts are already prepared.
        qk_clip_threshold: Positive clipping threshold for QK projections.
        reference_muon: Match native Graph O1 logical matrices, BF16 NS arithmetic and shared scaling.
        **kwargs: Public Muon Builder options from the training recipe.

    Returns:
        The standard builder, or its JT-local reference specialization.
    """
    if not math.isfinite(qk_clip_threshold) or qk_clip_threshold <= 0:
        raise ValueError("qk_clip_threshold must be finite and positive")
    if not isinstance(reference_muon, bool):
        raise ValueError("reference_muon must be a boolean")
    muon_config = dict(kwargs["muon_config"])
    if reference_muon:
        if any(muon_config.get(name) is not None for name in ("ns_transform_fn", "reshape_fn", "zeropower_fn")):
            raise ValueError("reference_muon cannot be combined with another Muon layout or NS callback")
        matched_rms = muon_config.get("matched_adamw_rms", 0.2)
        muon_config.update(
            ns_transform_fn=partial(_reference_muon_transform, config=model.config, matched_adamw_rms=matched_rms),
            zeropower_fn=_reference_newton_schulz,
            matched_adamw_rms=0.0, zero_rms_scale_mode="use_lr")
    builder_type = _JTReferenceMuonBuilder if reference_muon else Muon
    builder = builder_type(model=model, muon_config=muon_config, **{
        name: value for name, value in kwargs.items() if name != "muon_config"
    })
    optimizer = builder.get_optimizer()
    optimizer.chained_optimizers[-1].register_step_post_hook(partial(_after_update, model, qk_clip_threshold))
    return builder
