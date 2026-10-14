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
"""Fix 5 — apply the Muon slice scale / last-block factor / lr in float32 (MindFormers' semantics).

``OptimizerParityAudit`` localised the update-step divergence to two float32→bf16 quantisations on
HP's side of the Muon update:

* ``hyper_parallel/core/optimizer/muon.py:974`` (native path) and ``:1044`` (transform path) run
  ``update.mul_(slice_scale)`` **on the bfloat16 Newton–Schulz output**, so the scaled update is
  quantised to bf16 before it ever reaches the parameter;
* ``.../jt_deepseek_v3/adapter/runtime/mf_muon_parity.py:200-202`` multiplies the last-block
  ``factor`` in fp32 and then ``copy_``s the result into the **bf16** working tensor, i.e. it
  rounds to bf16 again.

MindFormers keeps the whole update in fp32 (``mindformers/core/optim/muon.py:150-153``:
``update_with_lr = adjusted_lr * x_ret`` then a fp32 parameter add).

Both quantisations exist only because the *Newton–Schulz input buffer* is bf16: the update is
scaled, restored and applied **through that same buffer**.  The NS arithmetic itself must stay
bfloat16 to match MindFormers (its NS inputs are bf16 too).  So the minimal faithful patch is

1. keep the NS **input buffer** in float32 (``_update_muon_momentum`` no longer casts its result to
   bfloat16) — the buffer then carries the exact bf16 Newton–Schulz values in an fp32 container;
2. cast to bf16 only *inside* the central Newton–Schulz routine
   (``_compute_batched_ns_outputs_for_tensors``), whose results are cast back to the caller's
   dtype — so the NS math is bit-identical to before while its output stays fp32.

Every downstream step (``update.mul_(slice_scale)``, the adapter's ``update.float() * factor``,
``target.copy_``, and ``torch._foreach_add_(fp32 params, updates, alpha=...)``) then runs in fp32,
exactly as MindFormers does.  Nothing in the repository is modified: both patches are runtime
attribute replacements on the optimiser class.
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path

import torch

_STATE = {'notes': 0, 'shard_path_calls': 0, 'ns_calls': 0, 'captured': [],
          'storage_names': {}}


def _optimizer_class():
    """Return the class that owns ``_update_muon_momentum``."""
    from hyper_parallel.core.optimizer import muon as muon_module  # pylint: disable=C0415

    for _, value in vars(muon_module).items():
        if inspect.isclass(value) and '_update_muon_momentum' in value.__dict__:
            return value, muon_module
    raise SystemExit('Fix 5: could not find the Muon optimizer class')


def apply(xret_numel=None, verbose=True):
    """Install both patches; returns the list of patched attributes."""
    cls, muon_module = _optimizer_class()
    patched = []

    original_momentum = cls._update_muon_momentum

    def momentum(self, group, params):
        """Return NS inputs in float32 (the bf16 cast happens inside the NS routine instead)."""
        updates = original_momentum(self, group, params)
        out = {}
        for parameter, tensor in updates.items():
            out[parameter] = tensor.float() if tensor.dtype != torch.float32 else tensor
        if _STATE['notes'] < 2 and updates:
            dtypes = sorted({str(t.dtype) for t in updates.values()})
            print(f'[PR180] Fix 5: NS inputs {dtypes} -> float32 ({len(out)} tensors)', flush=True)
            _STATE['notes'] += 1
        return out

    cls._update_muon_momentum = momentum
    patched.append(f'{cls.__name__}._update_muon_momentum')

    original_ns = cls._compute_batched_ns_outputs_for_tensors

    def ns_outputs(self, tensor_list, *args, **kwargs):
        """Run the Newton–Schulz step in bfloat16, hand the result back in float32."""
        if not tensor_list:
            return original_ns(self, tensor_list, *args, **kwargs)
        work = tensor_list
        upcast = tensor_list[0].dtype == torch.float32
        if upcast:
            work = [tensor.to(torch.bfloat16) for tensor in tensor_list]
        outputs = original_ns(self, work, *args, **kwargs)
        if not upcast:
            return outputs
        if _STATE['ns_calls'] < 2:
            print(f'[PR180] Fix 5: NS on {[str(t.dtype) for t in tensor_list]} via bf16, '
                  f'{len(outputs)} outputs back to float32', flush=True)
            _STATE['ns_calls'] += 1
        results = []
        for output in outputs:
            tensor = output.to(torch.float32)
            if xret_numel is not None and tensor.numel() == int(xret_numel):
                name = _STATE['storage_names'].get(
                    tensor_list[0].untyped_storage().data_ptr(), 'unknown')
                _STATE['captured'].append((name, tensor.detach().to('cpu')))
            results.append(tensor)
        if xret_numel is not None and _STATE['captured']:
            print(f'[PR180] Fix 5: captured NS outputs: '
                  f'{[(n, tuple(t.shape)) for n, t in _STATE["captured"][-4:]]}', flush=True)
            _STATE['captured'] = _STATE['captured'][-8:]
        return results

    cls._compute_batched_ns_outputs_for_tensors = ns_outputs
    patched.append(f'{cls.__name__}._compute_batched_ns_outputs_for_tensors')

    for attribute in ('_reshape_ns_input', '_prepare_ns_transform'):
        original_prepare = getattr(cls, attribute)

        def prepare(self, param, ns_input, _original=original_prepare):
            """Record ``{storage pointer: parameter name}`` so a capture can be attributed."""
            fqn = getattr(param, 'model_name', None) or repr(param)
            _STATE['storage_names'][ns_input.untyped_storage().data_ptr()] = fqn
            result = _original(self, param, ns_input)
            working = result[0] if isinstance(result, tuple) else result
            _STATE['storage_names'][working.untyped_storage().data_ptr()] = fqn
            return result

        setattr(cls, attribute, prepare)
        patched.append(f'{cls.__name__}.{attribute} (name map)')

    if hasattr(cls, '_fused_broadcast_and_apply'):
        original_fused = cls._fused_broadcast_and_apply

        def fused(self, *args, **kwargs):
            """Warn once: the sharded relay path packs updates into bf16 buffers."""
            _STATE['shard_path_calls'] += 1
            if _STATE['shard_path_calls'] == 1:
                print('[PR180] Fix 5 WARNING: the sharded relay path '
                      '(_fused_broadcast_and_apply) is in use; its bf16 pack buffers are NOT '
                      'patched, so the update would be quantised there', flush=True)
            return original_fused(self, *args, **kwargs)

        cls._fused_broadcast_and_apply = fused
        patched.append(f'{cls.__name__}._fused_broadcast_and_apply (instrumented)')

    print(f'[PR180] Fix 5 applied: {patched}', flush=True)
    if verbose:
        print('[PR180] Fix 5 = fp32 NS-input buffers + bf16-only NS arithmetic '
              '(scale/factor/lr and the parameter add stay fp32)', flush=True)
    return patched


def captured_summary():
    """Return the number of sharded-relay calls and of captured NS outputs."""
    return {'shard_path_calls': _STATE['shard_path_calls'],
            'captured_ns_outputs': len(_STATE['captured'])}


def dump_captured(directory, name='xret'):
    """Write the captured Newton–Schulz outputs as ``<name>__<i>.npy`` (float32 on disk)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not _STATE['captured']:
        print(f'[PR180] Fix 5: nothing captured for {name}', flush=True)
        return []
    import numpy as np  # pylint: disable=C0415

    written = []
    for index, (fqn, tensor) in enumerate(_STATE['captured']):
        safe = fqn.replace('model.', '').replace('/', '_')
        path = directory / f'{name}__{index}__{safe}.npy'
        np.save(str(path), tensor.numpy())
        written.append(str(path))
        print(f'[PR180] Fix 5: {path} {fqn} shape={tuple(tensor.shape)}', flush=True)
    return written


def xret_numel_from_env():
    """Return the ``PR180_XRET_NUMEL`` filter, or ``None`` when the capture is off."""
    value = os.environ.get('PR180_XRET_NUMEL')
    return int(value) if value else None
