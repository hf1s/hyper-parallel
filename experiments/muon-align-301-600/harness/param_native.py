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
"""Fingerprint the 46 native-layout **fp32 main parameters** after a step.

The Muon update adds into the mixed-precision optimizer's fp32 main parameters
(``Float16OptimizerWithFloat16Params.fp32_from_float16_groups``), which is the only place where
the update is visible in full precision — the bf16 model parameters are a rounded copy of them.
The native layout comes from the same recipe used for the gradients
(``mf_dvm_clip.native_recipes`` / ``materialize``), so an updated parameter can be compared with
MindFormers' ``param_next`` chain one-to-one.

Usage from the entry::

    import param_native
    param_native.install(trainer.base, Path(os.environ['PR180_PARAM_DUMP']), step=301)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

BLOCK = 1 << 23


def _main_params(optimizer):
    """Return ``{model_name: fp32 main parameter}`` of a mixed-precision optimizer."""
    from hyper_parallel.core.optimizer.dtensor_compat import to_local_if_dtensor  # pylint: disable=C0415

    named = {}
    for attribute in ('fp32_from_float16_groups', 'fp32_from_fp32_groups'):
        for group in getattr(optimizer, attribute, []) or []:
            for parameter in group:
                name = getattr(parameter, 'model_name', None)
                if name is None:
                    raise SystemExit('param_native: a fp32 main parameter has no model_name')
                named[name] = to_local_if_dtensor(parameter)
    if not named:
        raise SystemExit('param_native: no fp32 main parameters found on the optimizer')
    return named


def native_params(optimizer):
    """Return ``{native name: fp32 tensor on the host}`` in MindFormers' coordinates."""
    import mf_dvm_clip  # pylint: disable=C0415

    # the transforms in ``materialize`` slice and transpose, so the Dtensor has to be resolved to
    # its local tensor first (the FSDP2 shard is the whole parameter on a single rank)
    named = {}
    for name, parameter in _main_params(optimizer).items():
        local = parameter.detach().cpu()
        named[name] = local.to_local() if hasattr(local, 'to_local') else local
    recipes = mf_dvm_clip.native_recipes(named)
    out = {}
    for spec in mf_dvm_clip.NATIVE_SCHEDULE:
        tensor = mf_dvm_clip.materialize(recipes[spec['name']], spec['shape'])
        out[spec['name'].removeprefix('model.')] = tensor.detach().cpu()
        del tensor
    return out


def fingerprint(optimizer, directory, save_npy=True):
    """Record every native fp32 main parameter; return the per-parameter records."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for name, tensor in native_params(optimizer).items():
        flat = tensor.reshape(-1).contiguous()
        if flat.dtype != torch.float32:
            flat = flat.float()
        digest = hashlib.sha256()
        total = 0.0
        smallest = float('inf')
        largest = float('-inf')
        for offset in range(0, flat.numel(), BLOCK):
            block = flat[offset:offset + BLOCK]
            digest.update(block.numpy().astype('<f4', copy=False).tobytes())
            values = block.double()
            total += float((values * values).sum())
            smallest = min(smallest, float(values.min()))
            largest = max(largest, float(values.max()))
        record = {'name': name, 'dtype': 'float32', 'shape': list(tensor.shape),
                  'numel': int(flat.numel()), 'sha256': digest.hexdigest(),
                  'frobenius_fp64': total ** 0.5, 'min': smallest, 'max': largest}
        if save_npy:
            import numpy as np  # pylint: disable=C0415

            path = directory / f'{name}.npy'
            np.save(str(path), flat.numpy().astype('<f4', copy=False).reshape(tensor.shape))
            record['npy'] = str(path)
        records.append(record)
        print(f'[PR180] param {name:60s} {record["sha256"][:16]} '
              f'fro={record["frobenius_fp64"]:.6g}', flush=True)
        del tensor, flat
    (directory / 'hp-params.jsonl').write_text(
        ''.join(__import__('json').dumps(record, sort_keys=True) + '\n' for record in records))
    print(f'[PR180] {len(records)} native fp32 parameters -> {directory}', flush=True)
    return records


def install(trainer, directory, step=None, save_npy=True):
    """Fingerprint the fp32 main parameters once, after the first optimizer step."""
    import hyper_parallel.trainer.base as trainer_base  # pylint: disable=C0415

    original = trainer_base.BaseTrainer.step_optimizers_and_schedulers
    state = {'done': False}

    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if not state['done']:
            state['done'] = True
            try:
                optimizer = getattr(self, 'optimizer', None)
                if optimizer is None:
                    raise SystemExit('param_native: the trainer has no optimizer attribute')
                fingerprint(optimizer, directory, save_npy=save_npy)
            except Exception as error:  # pylint: disable=broad-except
                print(f'[PR180] parameter fingerprint failed: {type(error).__name__}: {error}',
                      flush=True)
        return result

    trainer_base.BaseTrainer.step_optimizers_and_schedulers = wrapped
    print(f'[PR180] native fp32 parameter fingerprint driver installed (step={step}, '
          f'{directory})', flush=True)
    return state
