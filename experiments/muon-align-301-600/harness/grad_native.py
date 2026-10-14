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
"""Fingerprint the 46 native-layout parameter gradients of the live HP backward pass.

The HP model keeps its parameters in its own packing (``qkv`` fused, gate/up interleaved,
``experts`` transposed), so a gradient can only be compared with MindFormers after undoing that
packing.  ``/alignment/tools/mf_dvm_clip.py`` already carries the exact recipe — its
``native_recipes``/``materialize`` pair maps live HP gradients onto the 46 native MF layouts used
by the measured DVM norm schedule — so this module reuses it rather than re-deriving the mapping.

Each parameter is recorded one at a time (the largest is a 3.1 GB fp32 tensor) as

* ``sha256`` of the little-endian fp32 C-order payload, the convention of
  ``/home/zth/tmp/align-cmp/mf-grads.jsonl``;
* ``frobenius_fp64``/``min``/``max``/``numel``/``shape``;
* optionally the payload itself (``PR180_GRAD_RAW=1``) for the norm-convention experiment.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

BLOCK = 1 << 23


def named_native_gradients(model):
    """Return ``({native name: recipe}, named_hp_gradients)`` in MindFormers' coordinates."""
    import mf_dvm_clip  # pylint: disable=C0415
    from hyper_parallel.core.utils.clip_grad import _get_local_grad  # pylint: disable=C0415

    named = {}
    for name, parameter in model.named_parameters():
        gradient = _get_local_grad(parameter)
        if gradient is None:
            continue
        key = getattr(parameter, 'model_name', name).replace('_checkpoint_wrapped_module.', '')
        # keep everything on the host: the transforms below would need temporary device buffers of
        # the same size as the gradient, and the model plus the 46 gradients already fill the card
        named[key] = gradient.detach().to('cpu')
    try:  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=C0415

        if hasattr(torch, 'npu'):
            torch.npu.empty_cache()
    except Exception:  # pylint: disable=broad-except
        pass
    return named, mf_dvm_clip


def fingerprint(model, directory, save_raw=False):
    """Record every native-layout parameter gradient; return the per-parameter records."""
    import mf_dvm_clip  # pylint: disable=C0415

    named, _ = named_native_gradients(model)
    recipes = mf_dvm_clip.native_recipes(named)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in mf_dvm_clip.NATIVE_SCHEDULE:
        native = spec['name'].removeprefix('model.')
        gradient = mf_dvm_clip.materialize(recipes[spec['name']], spec['shape'])
        tensor = gradient.detach().cpu()
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        flat = tensor.reshape(-1).contiguous()
        digest = hashlib.sha256()
        total = 0.0
        smallest = float('inf')
        largest = float('-inf')
        norm_sq = torch.zeros((), dtype=torch.float32)
        for offset in range(0, flat.numel(), BLOCK):
            block = flat[offset:offset + BLOCK]
            digest.update(block.numpy().astype('<f4', copy=False).tobytes())
            values = block.double()
            total += float((values * values).sum())
            smallest = min(smallest, float(values.min()))
            largest = max(largest, float(values.max()))
        with torch.no_grad():
            for offset in range(0, flat.numel(), BLOCK):
                block = flat[offset:offset + BLOCK]
                norm_sq = norm_sq + (block * block).sum()
        record = {'name': native, 'dtype': 'float32', 'shape': list(tensor.shape),
                  'numel': int(flat.numel()), 'sha256': digest.hexdigest(),
                  'frobenius_fp64': total ** 0.5, 'ssum_fp64': total,
                  'norm_fp32': float(norm_sq.sqrt()), 'min': smallest, 'max': largest}
        if save_raw:
            path = directory / f'{native}.raw'
            flat.numpy().astype('<f4', copy=False).tofile(path)
            record['raw'] = str(path)
        records.append(record)
        print(f'[PR180] grad {native:60s} {record["sha256"][:16]} fro={record["frobenius_fp64"]:.6g}',
              flush=True)
        del tensor, flat, gradient
    (directory / 'hp-grads.jsonl').write_text(
        ''.join(json.dumps(record, sort_keys=True) + '\n' for record in records))
    return records


def install(trainer, directory, save_raw=False):
    """Fingerprint the gradients once, just before the first optimizer/clip step."""
    import hyper_parallel.trainer.base as trainer_base  # pylint: disable=C0415

    original = trainer_base.BaseTrainer.prepare_optimizer_step
    state = {'done': False}

    def wrapped(self, *args, **kwargs):
        if not state['done']:
            state['done'] = True
            try:
                fingerprint(self.model, directory, save_raw)
                print(f'[PR180] 46 native gradient fingerprints -> {directory}', flush=True)
            except Exception as error:  # pylint: disable=broad-except
                print(f'[PR180] gradient fingerprint failed: {type(error).__name__}: {error}',
                      flush=True)
        return original(self, *args, **kwargs)

    trainer_base.BaseTrainer.prepare_optimizer_step = wrapped
    print(f'[PR180] native-gradient fingerprint driver installed ({directory})', flush=True)
    if os.environ.get('PR180_GRAD_RAW') == '1':
        print('[PR180] raw fp32 gradient payloads will be written', flush=True)
    return state
