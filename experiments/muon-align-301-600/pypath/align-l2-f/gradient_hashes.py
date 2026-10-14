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
"""Per-step gradient and parameter fingerprints in the shared alignment format.

Fingerprints cover the optimizer's named parameters only and are taken from the raw
gradients exactly as the backward pass produced them: the capture sits in front of the
trainer's gradient clipping, which is the first code that reads those gradients, so no
clipping, rounding or rescaling has happened yet. Only hashes and scalar statistics are
written; no gradient tensor is ever persisted.
"""

import hashlib
import json
from pathlib import Path

import torch

from hyper_parallel.core.optimizer.dtensor_compat import to_local_if_dtensor


def _optimizer_parameters(trainer) -> dict:
    """Return the optimizer-owned parameters keyed by their model name."""
    named = {}
    for optimizer in trainer.base.optimizer.optimizers_dict.values():
        for group in optimizer.param_groups:
            for parameter in group['params']:
                name = getattr(parameter, 'model_name', None)
                if not name:
                    raise ValueError('Optimizer parameter without a model_name')
                named[name.replace('_checkpoint_wrapped_module.', '')] = parameter
    if not named:
        raise ValueError('The optimizer exposes no named parameters')
    return named


def _local(tensor: torch.Tensor) -> torch.Tensor:
    """Detach a possibly distributed tensor and return its local value."""
    return to_local_if_dtensor(tensor).detach()


def _gradient(parameter: torch.Tensor):
    """Return the parameter's local gradient through the trainer's own accessor."""
    from hyper_parallel.core.utils.clip_grad import _get_local_grad  # pylint: disable=C0415

    return _get_local_grad(parameter)


def _float32_bytes(tensor: torch.Tensor) -> bytes:
    """Return little-endian float32 raw bytes of a contiguous, flattened tensor."""
    value = _local(tensor).to(torch.float32).contiguous().cpu().reshape(-1)
    return value.view(torch.uint8).numpy().tobytes()


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    """Return a parameter's bytes in its own dtype, without any conversion."""
    return _local(tensor).contiguous().cpu().view(torch.uint8).numpy().tobytes()


def _digest(payload: bytes) -> str:
    """Hash raw little-endian bytes."""
    return hashlib.sha256(payload).hexdigest()


def _gradient_records(named: dict) -> tuple:
    """Fingerprint every named parameter's raw pre-clipping gradient."""
    records, missing, total = [], [], 0.0
    for name in sorted(named):
        gradient = _gradient(named[name])
        if gradient is None:
            missing.append(name)
            continue
        value = _local(gradient)
        squares = value.to(torch.float64).cpu().reshape(-1)
        square_sum = float((squares * squares).sum())
        total += square_sum
        records.append({'name': name, 'dtype': str(gradient.dtype),
                        'numel': int(value.numel()),
                        'sha256': _digest(_float32_bytes(value)),
                        'frobenius_fp64': square_sum ** 0.5,
                        'min': float(squares.min()), 'max': float(squares.max())})
    return records, missing, total ** 0.5


def _write_name_mapping(named: dict, model: torch.nn.Module, output: Path) -> None:
    """Record how every HP parameter name maps onto its native counterpart."""
    from convert_state import mapping_for  # pylint: disable=C0415

    model_parameters = {id(parameter) for parameter in model.parameters()}
    parameters = {}
    for name in sorted(named):
        try:
            origins, transform = mapping_for(name)
        except Exception as error:  # pylint: disable=broad-except
            parameters[name] = {'error': f'{type(error).__name__}: {error}'}
            continue
        parameters[name] = {'native_names': origins, 'transform': transform}
    report = {
        'rule': ('convert_state.mapping_for maps an HP JT state-dict name to its native '
                 'MindFormers parameter name(s); a packed HP parameter lists every native name it '
                 "holds, and 'transform' names the layout change between the two layouts"),
        'index': ('names are the optimizer parameters\' own model_name attributes; '
                  f"{sum(id(parameter) in model_parameters for parameter in named.values())} of "
                  f'{len(named)} optimizer parameters are the model parameters themselves'),
        'parameter_count': len(parameters),
        'parameters': parameters,
    }
    (output / 'name_mapping.json').write_text(json.dumps(report, indent=2) + '\n')


def attach(trainer, output: str | Path) -> dict:
    """Fingerprint raw gradients and post-step parameters for every training step.

    Args:
        trainer: Built HP trainer whose optimizer owns the named parameters.
        output: Directory receiving the JSONL artifacts.

    Returns:
        The shared stash holding the current step's float64 gradient norm.
    """
    import hyper_parallel.trainer.base as trainer_base  # pylint: disable=C0415
    from hyper_parallel.trainer.callbacks.base import Callback  # pylint: disable=C0415

    output = Path(output)
    (output / 'grad_hashes').mkdir(parents=True, exist_ok=True)
    named = _optimizer_parameters(trainer)
    _write_name_mapping(named, trainer.base.model, output)
    print(f'[GRAD-HASH] optimizer parameters: {len(named)}', flush=True)
    pending = {}
    original = trainer_base.clip_grad_norm_

    def clip_grad_norm_(*args, **kwargs):
        """Fingerprint the raw gradients the trainer is about to clip."""
        records, missing, norm = _gradient_records(named)
        pending.update(records=records, missing=missing, norm_fp64=norm)
        return original(*args, **kwargs)

    trainer_base.clip_grad_norm_ = clip_grad_norm_

    class StepHashes(Callback):
        """Persist the stashed fingerprints once the optimizer has updated the weights."""

        def on_step_end(self, state, **kwargs):
            """Write this step's gradient hashes and the resulting parameter hashes."""
            del kwargs
            records = pending.pop('records', None)
            if records is None:
                return
            step = int(state.global_step)
            norm = pending['norm_fp64']
            missing = pending.pop('missing')
            payload = ''.join(json.dumps(record) + '\n' for record in records)
            (output / 'grad_hashes' / f'step-{step:06d}.jsonl').write_text(payload)
            if not (output / 'grad_hashes.jsonl').exists():
                (output / 'grad_hashes.jsonl').write_text(payload)
            (output / 'grad_norms.jsonl').write_text(json.dumps(
                {'step': step, 'param_count': len(records), 'grad_norm_fp64': norm,
                 'grad_norm_fp64_17g': '%.17g' % norm}) + '\n')
            parameters = [{'name': name, 'dtype': str(named[name].dtype),
                           'sha256': _digest(_raw_bytes(named[name]))} for name in sorted(named)]
            (output / f'step{step}_param_hashes.jsonl').write_text(
                ''.join(json.dumps(record) + '\n' for record in parameters))
            print(f'[GRAD-HASH] step={step} gradients={len(records)} parameters={len(parameters)} '
                  f'missing_gradients={json.dumps(missing)}', flush=True)
            print(f'[GRAD-HASH] step {step}: grad_norm_raw_fp64_17g={"%.17g" % norm}', flush=True)

    trainer.base._callbacks.append(StepHashes(trainer.base))
    return pending
