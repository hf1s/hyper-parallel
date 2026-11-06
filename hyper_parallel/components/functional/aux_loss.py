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
"""Reusable auxiliary-loss functions."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.dtensor.dtensor import DTensor, SkipDTensorDispatch


@dataclass
class _AuxLossScaleState:
    """Per-forward scale, shared with its backward and recomputation."""

    scale: torch.Tensor | float = 1.0
    active: bool = False
    tp_replicas: int = 1
    handles: list[Any] = field(default_factory=list)


_AUX_LOSS_STATE: ContextVar[_AuxLossScaleState | None] = ContextVar("aux_loss_state", default=None)


@contextmanager
def aux_loss_scale_context(*, loss_parallel: bool = False, tp_size: int = 1) -> Iterator[None]:
    """Isolate auxiliary gradient scaling for one forward/backward microstep.

    Call :func:`bind_aux_loss_scale` on the local scalar task loss before
    weighting it. Keep this context active through backward so checkpoint
    recomputation observes the same scale. Nested contexts are independent.

    Args:
        loss_parallel: Whether the task loss has partitioned vocabulary gradients.
        tp_size: Positive tensor-parallel degree. Only explicitly replicated
            auxiliary means use this degree for gradient normalization.
    """
    state = _AuxLossScaleState(tp_replicas=tp_size if loss_parallel else 1)
    token = _AUX_LOSS_STATE.set(state)
    try:
        yield
    finally:
        for handle in state.handles:
            handle.remove()
        _AUX_LOSS_STATE.reset(token)


def bind_aux_loss_scale(loss: torch.Tensor) -> None:
    """Make injected losses follow the scalar task loss's backward multiplier.

    This is a no-op outside :func:`aux_loss_scale_context` or when no auxiliary
    loss was attached. The hook runs before upstream model backward, including
    checkpoint recomputation, and never changes the task-loss gradient.

    Args:
        loss: Unweighted scalar foundation loss connected to the model output.
    """
    state = _AUX_LOSS_STATE.get()
    if state is None or not state.active or not loss.requires_grad:
        return
    if loss.numel() != 1:
        raise ValueError("Auxiliary gradient scaling requires a scalar task loss")

    def save_scale(gradient: torch.Tensor) -> None:
        """Capture the actual multiplier after token and parallel weighting.

        Args:
            gradient: Upstream scalar loss multiplier, possibly a DTensor.
        """
        with SkipDTensorDispatch():
            local_gradient = gradient.to_local() if isinstance(gradient, DTensor) else gradient
            state.scale = local_gradient.detach()

    state.handles.append(loss.register_hook(save_scale))


class _AuxLossAutoScaler(torch.autograd.Function):
    """Inject auxiliary-loss gradients without changing the forward loss value."""

    main_loss_backward_scale = torch.tensor(1.0)

    @staticmethod
    def forward(ctx: Any, output: torch.Tensor, aux_loss: torch.Tensor,
                tp_replicated: bool = False) -> torch.Tensor:
        """Save the auxiliary loss and return the output unchanged.

        Args:
            ctx: Autograd context for the auxiliary loss and scale state.
            output: Forward tensor carrying the injected gradient.
            aux_loss: Scalar objective whose gradient is injected in backward.
            tp_replicated: Whether the auxiliary mean is replicated across TP ranks.
        """
        ctx.save_for_backward(aux_loss)
        ctx.scale_state = _AUX_LOSS_STATE.get()
        ctx.tp_replicated = tp_replicated
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """Pass through the output gradient and inject the auxiliary-loss gradient.

        Args:
            ctx: Context saved by forward.
            grad_output: Unchanged gradient returned to the forward tensor.
        """
        (aux_loss,) = ctx.saved_tensors
        scale = (
            ctx.scale_state.scale if ctx.scale_state is not None
            else _AuxLossAutoScaler.main_loss_backward_scale
        )
        if isinstance(scale, torch.Tensor):
            scale = scale.to(device=aux_loss.device, dtype=aux_loss.dtype)
        tp_replicas = ctx.scale_state.tp_replicas if ctx.tp_replicated and ctx.scale_state is not None else 1
        if tp_replicas > 1:
            scale = scale / tp_replicas
        aux_loss_grad = torch.ones_like(aux_loss) * scale
        gradients = (grad_output, aux_loss_grad)
        return gradients + (None,) if len(ctx.needs_input_grad) == 3 else gradients

    @staticmethod
    def set_loss_scale(scale: torch.Tensor) -> None:
        """Set the auxiliary-loss gradient scale."""
        _AuxLossAutoScaler.main_loss_backward_scale = scale


def aux_loss_auto_scale(output: torch.Tensor, aux_loss: torch.Tensor, *,
                        tp_replicated: bool = False) -> torch.Tensor:
    """Attach an auxiliary-loss gradient to an unchanged forward tensor.

    Args:
        output: Main forward output.
        aux_loss: Scalar auxiliary loss whose gradient should be injected.
        tp_replicated: The auxiliary mean represents the same objective on each
            TP rank. Average its injected gradients when loss parallelism keeps
            task-loss gradients partitioned; leave partial auxiliary objectives
            at the default.

    Returns:
        ``output`` unchanged in value, with an autograd edge to ``aux_loss``.
    """
    state = _AUX_LOSS_STATE.get()
    if state is not None:
        state.active = True
    return _AuxLossAutoScaler.apply(output, aux_loss, tp_replicated)


def set_aux_loss_scale(scale: torch.Tensor) -> None:
    """Set the gradient multiplier used by :func:`aux_loss_auto_scale`.

    Args:
        scale: Tensor used to scale the injected auxiliary-loss gradient.
    """
    _AuxLossAutoScaler.set_loss_scale(scale)
