# Copyright 2026 Huawei Technologies Co., Ltd.
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

"""Explicit optimizer objects for archived MF continuation, without global overrides."""

# This explicitly selected recipe is Torch/NPU-only.
# pylint: disable=forbidden-backend-import

import math
from functools import partial
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from hyper_parallel.components.optim.parameter_groups import get_adamw_param_groups, split_muon_adamw_params
from hyper_parallel.core.optimizer import _filter_optimizer_config
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.muon import Muon, NSInputTransform
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.models.jt_deepseek_v3.adapter.runtime.jt_optimizer import _after_update
from hyper_parallel.models.jt_deepseek_v3.mf_archive.reductions import normalize_ns


def _local(value: torch.Tensor) -> torch.Tensor:
    return value.to_local() if hasattr(value, "to_local") else value


def _logical_matrices(name: str, value: torch.Tensor, config: Any) -> NSInputTransform:
    if name.endswith(".self_attn.linear_qkv.weight"):
        pieces = list(value.split((config.q_lora_rank, config.kv_lora_rank, config.qk_rope_head_dim), dim=0))

        def restore(updates: list[torch.Tensor], output: torch.Tensor) -> None:
            """Restore the row-concatenated archived Q/KV/RoPE layout."""
            output.copy_(torch.cat(updates, dim=0))

    elif name.endswith((".self_attn.q_b_proj.weight", ".self_attn.kv_b_proj.weight")):
        first = config.qk_nope_head_dim
        second = config.qk_rope_head_dim if name.endswith("q_b_proj.weight") else config.v_head_dim
        columns = value.shape[1]
        shaped = value.reshape(config.num_attention_heads, first + second, columns)
        pieces = [shaped[:, :first].reshape(-1, columns).contiguous(),
                  shaped[:, first:].reshape(-1, columns).contiguous()]

        def restore(updates: list[torch.Tensor], output: torch.Tensor) -> None:
            """Restore the interleaved per-head projection layout."""
            left = updates[0].reshape(config.num_attention_heads, first, columns)
            right = updates[1].reshape(config.num_attention_heads, second, columns)
            output.copy_(torch.cat((left, right), dim=1).reshape_as(output))

    elif name.endswith(".mlp.experts.gate_up_proj"):
        pieces = list(value.chunk(2, dim=1))

        def restore(updates: list[torch.Tensor], output: torch.Tensor) -> None:
            """Restore fused expert gate/up storage after independent updates."""
            output.copy_(torch.cat(updates, dim=1))

    else:
        pieces = [value]

        def restore(updates: list[torch.Tensor], output: torch.Tensor) -> None:
            """Return an unsplit matrix to its original parameter."""
            output.copy_(updates[0])

    return NSInputTransform(pieces, restore)


def archived_ns(value: torch.Tensor, steps: int) -> torch.Tensor:
    """Apply the archived 2D/3D scalar placement, casts and Gram storage choices."""
    current = normalize_ns(value)
    for _ in range(steps):
        if current.ndim == 2:
            gram = current @ current.mT.contiguous()
            square = (gram * 2.03125) @ gram
            polynomial = (gram.float() * -4.78125 + square.float()).bfloat16()
        else:
            gram = torch.bmm(current, current.mT)
            square = gram @ gram
            polynomial = (gram.float() * -4.78125 + square.float() * 2.03125).bfloat16()
        product = polynomial @ current
        current = (current.float() * 3.4375 + product.float()).bfloat16()
    return (current.mT if value.shape[-2] > value.shape[-1] else current).float()


class ArchivedMuon(Muon):
    """Single-card continuation recipe; native Muon remains the independent default.

    The source archive uses a sum-form momentum, a BF16 NS iteration and a
    last-piece Q-up scale. The latter is retained only to reproduce this archive,
    not asserted to be the correct shape scaling for a new training run.
    """

    def __init__(self, params: Any, *, model_config: Any, **kwargs: Any) -> None:
        """Build native Muon state while explicitly selecting the archived recipe."""
        if not dist.is_initialized() or dist.get_world_size() != 1:
            raise ValueError("The archived optimizer is restricted to one initialized rank")
        self.model_config = model_config
        self.restored_lr_folding: list[bool] = []
        kwargs["apply_lr_in_update"] = True
        super().__init__(params, **kwargs)

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore all native tensor state, selecting this recipe's update representation.

        ``apply_lr_in_update`` describes where LR is folded, not a moment or a
        training step. Converted native checkpoints use False; this explicit
        recipe uses True. No tensor state, LR, momentum or step is substituted.
        """
        super().load_state_dict(state_dict)
        self.restored_lr_folding = [group["apply_lr_in_update"] for group in self.param_groups]
        for group in self.param_groups:
            group["apply_lr_in_update"] = True

    def _update_muon_momentum(self, group, params):
        beta1, beta2 = group["momentum"]
        if beta1 != beta2 or group["ns_variant"] != "legacy" or group["ns_epsilon"] != 1e-7:
            raise ValueError("The archived recipe requires equal momentum coefficients and legacy NS epsilon 1e-7")
        updates = {}
        for parameter in params:
            if parameter.grad is None:
                continue
            gradient = _local(parameter.grad)
            if gradient.dtype != torch.float32:
                raise ValueError("The archived momentum requires FP32 gradients")
            state = self.state[parameter]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(parameter.grad)
            buffer = _local(state["momentum_buffer"])
            buffer.mul_(beta2).add_(gradient)
            updates[parameter] = gradient + buffer * beta1 if group["nesterov"] else buffer.clone()
        return updates

    def _compute_batched_ns_updates(self, p_list, ns_inputs, group, no_shard=False):
        del no_shard
        if not group["apply_lr_in_update"] or group["ns_coefficients"] is not None:
            raise ValueError("The archived NS recipe requires internal LR folding and the legacy coefficient schedule")
        restored = {}
        for parameter in p_list:
            name = parameter.model_name
            source = ns_inputs[parameter].view(parameter.shape)
            working = source.contiguous()
            transform = _logical_matrices(name, working, self.model_config)
            updates = []
            for value in transform.tensors:
                if name.endswith(("q_b_proj.weight", "kv_b_proj.weight", "gate_proj.weight", "up_proj.weight")):
                    value = value.bfloat16().float()
                if value.ndim == 3:
                    value = value.bfloat16()
                elif value.shape[-2] > value.shape[-1] or tuple(value.shape) in ((512, 5120), (64, 5120)):
                    value = value.bfloat16().float()
                update = archived_ns(value, group["ns_steps"])
                dimension = max(update.shape[-2:])
                if name.endswith(".self_attn.q_b_proj.weight"):
                    dimension = max(transform.tensors[-1].shape[-2:])
                scale = np.float32(np.float32(math.sqrt(float(dimension))) * np.float32(group["matched_adamw_rms"]))
                scale = np.float32(scale * np.float32(-group["lr"]))
                updates.append(update.mul_(float(scale)))
            transform.restore(updates, working)
            if working.data_ptr() != source.data_ptr():
                source.copy_(working)
            restored[parameter] = source
        return restored


def archived_adamw(parameter: torch.Tensor, gradient: torch.Tensor, moment: torch.Tensor,
                   variance: torch.Tensor, group: dict) -> None:
    """Update one FP32 parameter with the archive's scalar and addcmul boundaries."""
    values = [_local(value) for value in (parameter, gradient, moment, variance)]
    if any(value.dtype != torch.float32 or not value.is_contiguous() for value in values):
        raise ValueError("Archived AdamW requires contiguous FP32 parameter, gradient and moment state")
    weight, grad, first, second = [value.view(-1) for value in values]
    beta1, beta2, lr, decay, eps = [torch.tensor([number], device=weight.device, dtype=torch.float32)
                                  for number in (*group["betas"], group["lr"], group["weight_decay"], group["eps"])]
    step_size = lr / (1 - beta1.pow(group["step"]))
    correction = 1 - beta2.pow(group["step"])
    variance_scale = (1 - beta2).item()
    decay_scale = 1 - lr * decay
    for start in range(0, weight.numel(), 1048576):
        end = start + 1048576
        current_weight, current_grad = weight[start:end], grad[start:end]
        current_m, current_v = first[start:end], second[start:end]
        next_weight = current_weight * decay_scale
        current_m.copy_(current_m * beta1 + current_grad * (1 - beta1))
        current_v.copy_(torch.addcmul(current_v * beta2, current_grad, current_grad, value=variance_scale))
        denominator = (current_v / correction).sqrt() + eps
        current_weight.copy_(next_weight - (current_m / denominator) * step_size)


class ArchivedAdamW(AdamW):
    """Native checkpoint contract with explicit archived arithmetic on each tensor."""

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        """Advance each native group once and update its live parameter and moment tensors."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["amsgrad"] or group["maximize"]:
                raise ValueError("The archived recipe does not support AMSGrad or maximization")
            group["step"] = (group.get("step") or 0) + 1
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["exp_avg"] = torch.zeros_like(parameter.grad)
                    state["exp_avg_sq"] = torch.zeros_like(parameter.grad)
                archived_adamw(parameter, parameter.grad, state["exp_avg"], state["exp_avg_sq"], group)
        return loss


class ArchivedOptimizerBuilder:
    """YAML factory that composes explicit archive optimizer objects and native JT hooks."""

    def __init__(self, *, model: torch.nn.Module, muon_config: dict, adamw_config: dict,
                 qk_clip_threshold: float, **kwargs: Any) -> None:
        """Reuse native grouping, checkpoint composition and post-update JT hooks."""
        if kwargs:
            raise ValueError(f"Unsupported archive optimizer options: {sorted(kwargs)}")
        if not math.isfinite(qk_clip_threshold) or qk_clip_threshold <= 0:
            raise ValueError("QK clipping threshold must be positive and finite")
        muon_params, adamw_params, _, _ = split_muon_adamw_params(model)
        adamw_groups, _ = get_adamw_param_groups(
            model, weight_decay=adamw_config.get("adamw_weight_decay", .1),
            allowed_param_ids=[id(parameter) for parameter in adamw_params])
        adamw = ArchivedAdamW(adamw_groups, **_filter_optimizer_config("adamw", AdamW, adamw_config))
        muon = ArchivedMuon(muon_params, model_config=model.config,
                            **_filter_optimizer_config("muon", Muon, muon_config))
        self.optimizer = ChainedOptimizer(model, optimizers={"adamw": adamw, "muon": muon}, flatten=True)
        muon.register_step_post_hook(partial(_after_update, model, qk_clip_threshold))

    def get_optimizer(self) -> ChainedOptimizer:
        """Return the standard chained checkpoint and training interface."""
        return self.optimizer
