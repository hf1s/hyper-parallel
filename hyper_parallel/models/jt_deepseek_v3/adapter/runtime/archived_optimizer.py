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

"""Archived MF optimizer recipe: ``ArchivedMuon`` + ``ArchivedAdamW``.

Selected explicitly with the YAML optimizer target ``ArchivedOptimizerBuilder``;
the default JT optimizer target (``jt_optimizer.build_optimizer``) never imports
this module. Everything here replaces native Muon/AdamW arithmetic for the
explicit archive continuation only; the framework optimizers are unchanged.

NS norms use the shared CANN 8.5 schedules in ``archived_reductions``. They are
reproducibility operations, not more accurate replacements for native
reductions; small partial sums cross to the host.
"""

# This explicitly selected recipe is Torch/NPU-only.
# pylint: disable=forbidden-backend-import

import math
from collections.abc import Callable
from functools import partial
from typing import Any, List, Optional

import numpy as np
import torch
import torch.distributed as dist

from hyper_parallel.components.optim.parameter_groups import get_adamw_param_groups, split_muon_adamw_params
from hyper_parallel.core.optimizer import _filter_optimizer_config
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.dtensor_compat import detect_dtensor_backend, to_local_if_dtensor
from hyper_parallel.core.optimizer.muon import Muon, NSInputTransform
from hyper_parallel.core.optimizer.muon_shard import chunk_update_by_layout
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
# The QK-clip / expert-bias post-step hook is deliberately shared with the default JT optimizer target.
from hyper_parallel.models.jt_deepseek_v3.adapter.runtime.jt_optimizer import _after_update
from hyper_parallel.models.jt_deepseek_v3.archived_reductions import (
    adjacent_tree, core_sum, gradient_square_sum, half_tree, tiled_square_sum,
)


def normalize_ns(value: torch.Tensor) -> torch.Tensor:
    """Return the archived BF16 NS input, including its documented cast boundaries."""
    shape = tuple(value.shape)
    current = value.mT if value.shape[-2] > value.shape[-1] else value
    epsilon = 1.0011717677116394e-7
    if value.ndim == 3:
        if value.dtype != torch.bfloat16:
            raise ValueError("Archived expert NS requires BF16 input")
        if shape == (8, 5120, 1536):
            rows = current.float().square().reshape(8, 768, 160, 64).transpose(-1, -2)
            partials = adjacent_tree(half_tree(rows)).cpu().numpy()
            totals = np.asarray([core_sum(row, 5) for row in partials], dtype=np.float32)
            norm = torch.from_numpy(totals).to(value.device).sqrt().reshape(8, 1, 1)
        else:
            # The compiled archive adds epsilon before its one BF16 conversion;
            # rounding the norm earlier changes values near a BF16 midpoint.
            norm = current.float().norm(dim=(-2, -1), keepdim=True)
        return (current / (norm + 1e-7).bfloat16()).bfloat16()
    if value.ndim != 2 or value.dtype != torch.float32:
        raise ValueError("Archived matrix NS requires FP32 two-dimensional input")
    if shape in ((1536, 5120), (5120, 16384), (5120, 10240), (5120, 12288)):
        total = tiled_square_sum(current, 6984, 24)
        norm = torch.tensor(total, device=value.device).sqrt()
    elif shape in ((16384, 1536), (8192, 1536), (12288, 5120), (5120, 3072)):
        total = tiled_square_sum(current, 9776, 16)
        norm = torch.tensor(total, device=value.device).sqrt()
    elif shape == (8, 5120):
        norm = torch.tensor(gradient_square_sum(current), device=value.device).sqrt()
        epsilon = 1e-7
    else:
        norm = current.norm(dim=(-2, -1), keepdim=True)
    return (current / (norm + epsilon)).bfloat16()


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

    # Framework Muon hooks whose role this recipe's fixed arithmetic takes over.
    _REPLACED_HOOKS = ("zeropower_fn", "momentum_update_fn", "reshape_fn", "ns_transform_fn")

    def __init__(self, params: Any, *, model_config: Any, **kwargs: Any) -> None:
        """Build native Muon state while explicitly selecting the archived recipe.

        Raises:
            ValueError: Outside one initialized rank, if LR folding is disabled, if
                a replaced framework hook is supplied, or for momentum/NS options the
                archived arithmetic does not implement.
        """
        if not dist.is_initialized() or dist.get_world_size() != 1:
            raise ValueError("The archived optimizer is restricted to one initialized rank")
        if kwargs.get("apply_lr_in_update", True) is not True:
            raise ValueError("The archived recipe folds LR into the update; apply_lr_in_update cannot be disabled")
        replaced = sorted(name for name in self._REPLACED_HOOKS if kwargs.get(name) is not None)
        if replaced:
            raise ValueError(f"The archived recipe replaces these framework Muon hooks: {replaced}")
        self.model_config = model_config
        self.restored_lr_folding: list[bool] = []
        kwargs["apply_lr_in_update"] = True
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            beta1, beta2 = group["momentum"]
            if beta1 != beta2 or group["ns_variant"] != "legacy" or group["ns_epsilon"] != 1e-7:
                raise ValueError("The archived recipe requires equal momentum coefficients and legacy NS epsilon 1e-7")

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
            gradient = to_local_if_dtensor(parameter.grad)
            if gradient.dtype != torch.float32:
                raise ValueError("The archived momentum requires FP32 gradients")
            state = self.state[parameter]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(parameter.grad)
            buffer = to_local_if_dtensor(state["momentum_buffer"])
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

    def _fused_broadcast_and_apply(self, valid_params, my_updates, param_compute_coord, group, hsdp_assign):
        """Apply the computed update directly; a single rank never needs to broadcast.

        The framework's ``Muon._fused_broadcast_and_apply`` always packs updates
        into a BF16 buffer before any possible broadcast, which rounds away the
        FP32 precision this recipe computes even when, as here, no communication
        ever happens. Overriding only this method keeps the fix local to the
        archived recipe without changing framework Muon.
        """
        del param_compute_coord  # only used to coordinate which rank computes which shard
        _, _, _, total_shard_size = self._get_shard_info(hsdp_assign)
        if total_shard_size != 1:
            raise ValueError("The archived optimizer requires a singleton shard group")
        local_params = []
        local_updates = []
        for parameter in valid_params:
            local_param = to_local_if_dtensor(parameter.data)
            update = chunk_update_by_layout(
                my_updates[parameter], parameter, hsdp_assign.layout_spec, self._param_shard_metadata.get(parameter))
            local_params.append(local_param)
            local_updates.append(update.view(local_param.shape))
        if not local_params:
            return
        if group["weight_decay"] != 0.0:
            torch._foreach_mul_(local_params, 1 - group["lr"] * group["weight_decay"])  # pylint: disable=protected-access
        apply_alpha = 1.0 if group["apply_lr_in_update"] else -group["lr"]
        torch._foreach_add_(local_params, local_updates, alpha=apply_alpha)  # pylint: disable=protected-access


def archived_adamw(parameter: torch.Tensor, gradient: torch.Tensor, moment: torch.Tensor,
                   variance: torch.Tensor, group: dict) -> None:
    """Update one FP32 parameter with the archive's scalar and addcmul boundaries."""
    values = [to_local_if_dtensor(value) for value in (parameter, gradient, moment, variance)]
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
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
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
    """YAML optimizer target chaining ``ArchivedAdamW`` and ``ArchivedMuon`` with the native JT post-step hook.

    Parameter routing, ``ChainedOptimizer`` composition and the QK-clip / expert-bias
    hook are the same as the default ``jt_optimizer.build_optimizer`` target, which
    stays unchanged; only the two leaf optimizers differ.
    """

    def __init__(
            self,
            *,
            model: torch.nn.Module,
            muon_config: dict,
            adamw_config: dict,
            qk_clip_threshold: float,
            extra_adamw_name_keywords: Optional[List[str]] = None,
            no_decay_params: Optional[List[str]] = None,
    ) -> None:
        """Route parameters like the native builder, chain the archived optimizers and register the JT hook.

        Raises:
            ValueError: If ``qk_clip_threshold`` is not finite and positive, or a parameter class is empty.
        """
        if not math.isfinite(qk_clip_threshold) or qk_clip_threshold <= 0:
            raise ValueError("qk_clip_threshold must be finite and positive")
        muon_params, adamw_params, _, _ = split_muon_adamw_params(
            model, extra_adamw_name_keywords=extra_adamw_name_keywords or ())
        if not muon_params or not adamw_params:
            raise ValueError("The archived optimizer recipe requires both Muon and AdamW parameters")
        adamw_groups, _ = get_adamw_param_groups(
            model, weight_decay=adamw_config.get("adamw_weight_decay", 1e-2), no_decay_params=no_decay_params,
            allowed_param_ids=[id(parameter) for parameter in adamw_params])
        detect_dtensor_backend(adamw_groups, muon_params)
        adamw = ArchivedAdamW(adamw_groups, **_filter_optimizer_config("adamw", AdamW, adamw_config))
        muon = ArchivedMuon(muon_params, model_config=model.config,
                            **_filter_optimizer_config("muon", Muon, muon_config))
        self.optimizer = ChainedOptimizer(model, optimizers={"adamw": adamw, "muon": muon}, flatten=True)
        muon.register_step_post_hook(partial(_after_update, model, qk_clip_threshold))

    def get_optimizer(self) -> ChainedOptimizer:
        """Return the chained optimizer runtime."""
        return self.optimizer
