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
"""Adamw optimizer."""

from typing import List

import torch


def adamw(
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        max_exp_avg_sqs: List[torch.Tensor],
        step: int,
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float,
        maximize: bool
) -> None:
    r"""Functional API that performs AdamW algorithm computation.
    See :class:`~torch.optim.AdamW` for details.
    """
    step_tensor = torch.tensor(step, dtype=torch.int64, device=params[0].device)
    state_steps = [step_tensor] * len(params)

    torch._fused_adamw_(  # pylint: disable=protected-access
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs if amsgrad else [],
        state_steps,
        amsgrad=amsgrad,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize
    )


class AdamW(torch.optim.Optimizer):
    """AdamW with bias correction based on each parameter's effective updates.

    A missing gradient skips that parameter's counter as well as its moments.
    Legacy checkpoints without parameter counters use the saved group step for
    initialized moments; their historical missing-gradient counts are unavailable.
    """

    PARAMETER_STEP_MODE = '_hp_adamw_parameter_steps'

    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=False,
            maximize=False
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
            "maximize": maximize
        }
        super().__init__(params, defaults)

    def add_param_group(self, param_group: dict) -> None:
        """Register the explicit checkpoint policy for this optimizer family."""
        super().add_param_group(param_group)
        self.param_groups[-1][self.PARAMETER_STEP_MODE] = True

    def __setstate__(self, state):
        """Set optimizer state."""
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group[self.PARAMETER_STEP_MODE] = True
            for param in group['params']:
                param_state = self.state.get(param)
                if param_state:
                    # Torch's flat loader reconstructs absent legacy fields as
                    # empty dictionaries. A never-active parameter stays lazy.
                    if all(isinstance(param_state.get(key), dict) and not param_state[key]
                           for key in ('exp_avg', 'exp_avg_sq')):
                        param_state.clear()
                    else:
                        param_state['step'] = self.get_parameter_step(param_state, group.get('step') or 0)

    @staticmethod
    def get_parameter_step(state: dict, group_step: int) -> int:
        """Read an effective update count, including legacy checkpoint defaults.

        ``None`` or an empty dictionary represents a legacy entry with moments
        but no saved counter after DCP or Torch flat-state reconstruction.
        Tensor counters imported from Torch are normalized once, outside the
        ordinary integer-counter path.

        Args:
            state: This parameter's optimizer state.
            group_step: Saved group count used only for initialized legacy state.

        Returns:
            The nonnegative number of effective updates already completed.
        """
        value = state.get('step')
        if value is None or (isinstance(value, dict) and not value):
            value = group_step if any(isinstance(state.get(key), torch.Tensor)
                                     for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq')) else 0
        if isinstance(value, torch.Tensor):
            value = value.item()
        try:
            step = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('AdamW parameter step must be a nonnegative integer') from exc
        if step < 0 or step != value:
            raise ValueError('AdamW parameter step must be a nonnegative integer')
        return step

    def __str__(self):
        return super().__repr__()

    __repr__ = __str__

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            previous_group_step = group.get('step') or 0
            group['step'] = previous_group_step + 1
            params_by_step = {}

            current_rank_params = group['params']
            for p in current_rank_params:
                if p.grad is None:
                    continue

                if p.grad.data.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                state = self.state[p]
                completed_steps = self.get_parameter_step(state, previous_group_step)

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)

                state['step'] = completed_steps + 1
                params_by_step.setdefault(state['step'], []).append(p)

            for parameter_step, params_with_grad in params_by_step.items():
                states = [self.state[param] for param in params_with_grad]
                adamw(
                    params_with_grad,
                    [param.grad for param in params_with_grad],
                    [state['exp_avg'] for state in states],
                    [state['exp_avg_sq'] for state in states],
                    [state['max_exp_avg_sq'] for state in states] if amsgrad else [],
                    parameter_step,
                    amsgrad=amsgrad,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    eps=group['eps'],
                    maximize=group['maximize']
                )

        return loss
