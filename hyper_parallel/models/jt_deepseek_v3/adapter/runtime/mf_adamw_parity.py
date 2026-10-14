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
"""MindFormers-parity AdamW for the JT DeepSeek-V3 optimizer's non-Muon parameters.

HP's AdamW (``core/optimizer/adamw.py``) calls ``torch._fused_adamw_``, a single fused
kernel. MindFormers' Muon optimizer runs the same AdamW algorithm as a sequence of
separate elementwise ops for its non-Muon parameters (embedding, RMSNorm, router, bias):
``muon.py:159-198`` casts the parameter and gradient to FP32 once, applies decoupled
weight decay, updates the running moments, computes the bias-corrected step and casts
back once. Both compute the textbook decoupled-AdamW update; a fused kernel's internal
op grouping rounds a long chain of FP32 operations differently from a fused kernel's,
which issue #179's ``native-adamw`` factor measures (200-step MAE 0.059, first-step
delta 0, judged reference-compatible, not a default-changing fix).

This module ports MF's exact op sequence as an alternative AdamW for alignment
experiments; it is not selected by default and is not a product change.
"""

from typing import Any

import torch


class MFParityAdamW(torch.optim.Optimizer):
    """AdamW computed as MindFormers' explicit op sequence instead of a fused kernel.

    Matches ``core.optimizer.adamw.AdamW``'s parameter-group schema and FP32 state
    convention (``exp_avg``/``exp_avg_sq`` take the gradient's dtype, ``step`` is one
    shared counter per group) so it drops into the same param groups. ``amsgrad`` and
    ``maximize`` are not part of MF's formula and are rejected rather than silently
    ignored.
    """

    def __init__(self, params: Any, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                eps: float = 1e-8, weight_decay: float = 0.01, amsgrad: bool = False,
                maximize: bool = False) -> None:
        """Build the optimizer over ``params``, same defaults as ``core.optimizer.adamw.AdamW``.

        Args:
            params: Parameters or per-group dicts, as for any ``torch.optim.Optimizer``.
            lr: Default learning rate.
            betas: Default first and second moment decay rates.
            eps: Default denominator epsilon.
            weight_decay: Default decoupled weight decay.
            amsgrad: Must be ``False``; MF's formula has no AMSGrad variant.
            maximize: Must be ``False``; MF's formula does not negate the gradient.

        Raises:
            ValueError: If a group requests ``amsgrad`` or ``maximize``.
        """
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay,
                    "amsgrad": amsgrad, "maximize": maximize}
        super().__init__(params, defaults)
        for group in self.param_groups:
            if group["amsgrad"] or group["maximize"]:
                raise ValueError("MFParityAdamW does not implement amsgrad or maximize")

    def step(self, closure: Any = None) -> Any:
        """Run one update, computing each op in the order MF's ``_apply_adamw_update`` does."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = [p for p in group["params"] if p.grad is not None]
            if not params_with_grad:
                continue
            if any(p.grad.is_sparse for p in params_with_grad):
                raise RuntimeError("MFParityAdamW does not support sparse gradients")
            group["step"] = (group.get("step") or 0) + 1
            step = group["step"]
            device = params_with_grad[0].device
            # Every scalar constant below is computed as an FP32 tensor, as MF's muon.py
            # casts them explicitly (``op_cast(beta1, mstype.float32)``, ...); a Python float
            # scalar computed in double precision and narrowed later can round a value such
            # as ``1 - beta2`` to a different FP32 bit pattern than computing it in FP32.
            beta1, beta2, lr, eps = (
                torch.tensor(value, device=device, dtype=torch.float32)
                for value in (*group["betas"], group["lr"], group["eps"]))
            one_minus_beta1, one_minus_beta2 = 1 - beta1, 1 - beta2
            bias_correction1, bias_correction2 = 1 - beta1 ** step, 1 - beta2 ** step
            step_size = lr / bias_correction1
            decay_factor = 1 - lr * group["weight_decay"]

            for parameter in params_with_grad:
                state = self.state[parameter]
                if not state:
                    state["exp_avg"] = torch.zeros_like(parameter.grad, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(parameter.grad, memory_format=torch.preserve_format)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                next_param = parameter.float() * decay_factor
                gradient_fp32 = parameter.grad.float()

                exp_avg.copy_(exp_avg * beta1 + gradient_fp32 * one_minus_beta1)
                exp_avg_sq.copy_(torch.addcmul(exp_avg_sq * beta2, gradient_fp32, gradient_fp32,
                                               value=one_minus_beta2.item()))

                denom = (exp_avg_sq / bias_correction2).sqrt() + eps
                next_param = next_param - (exp_avg / denom) * step_size
                parameter.data.copy_(next_param.to(parameter.dtype))

        return loss


def build_optimizer(*, model: torch.nn.Module, qk_clip_threshold: float, **kwargs: Any) -> Any:
    """Build the JT optimizer with MF-parity AdamW for the non-Muon parameters.

    Builds the ordinary JT optimizer first, keeping its Muon instance (and the
    QK-clip/expert-bias hook already registered on it) unchanged, then wraps that same
    Muon instance and a fresh :class:`MFParityAdamW`, built from the resolved AdamW
    param groups, in a new chained optimizer.

    Args:
        model: Model whose final FSDP parameter layouts are already prepared.
        qk_clip_threshold: Positive clipping threshold for QK projections.
        **kwargs: Muon builder options from the training recipe.

    Returns:
        The chained optimizer.

    Raises:
        ValueError: If the JT optimizer has no AdamW parameter group to replace.
    """
    from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer  # pylint: disable=C0415
    from hyper_parallel.models.jt_deepseek_v3.adapter.runtime.jt_optimizer import (  # pylint: disable=C0415
        build_optimizer as build_jt_optimizer,
    )

    builder = build_jt_optimizer(model=model, qk_clip_threshold=qk_clip_threshold, **kwargs)
    chained = builder.get_optimizer()
    if "adamw" not in chained.optimizers_dict:
        raise ValueError("mf_adamw_parity requires an AdamW parameter group to replace")
    mf_adamw = MFParityAdamW(chained.optimizers_dict["adamw"].param_groups)
    optimizers = dict(chained.optimizers_dict)
    optimizers["adamw"] = mf_adamw
    builder.optimizer = ChainedOptimizer(model, optimizers=optimizers, flatten=chained.flatten)
    return builder
