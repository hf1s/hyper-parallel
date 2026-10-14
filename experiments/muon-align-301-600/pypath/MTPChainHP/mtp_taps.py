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
"""Operator-boundary backward taps for the HP multi-token-prediction branch.

Read-only instrumentation: the only repo-visible effect is a handful of monkey patches installed
on the running process (:func:`attach`); no file under ``/workspace/hyper-parallel`` or the
worktree is touched.  Every recorded tensor is written once as

* ``<key>.raw``  - the tensor in its **own** dtype, little-endian, headerless, C order;
* ``<key>.json`` - dtype/shape/device/layout/sha256-of-raw/fp64 moments/min/max/note;
* ``<key>.npy``  - source-dtype convenience copy, only below ``NPY_LIMIT`` bytes;

and the whole set is summarised in ``index.json`` next to them.

Backward chain covered (``mtp.layers.0`` unless stated otherwise)::

    mtp_loss                       foundation_loss/mtp 0-d loss          mtp_loss__{val,dy}
      -> projected_cross_entropy   chunked head+CE (loss_chunk_size=4096 => one chunk here)
           -> MTP CE scalar        per-chunk token-mean CE              mtp_layer_loss__{val,dy}
                -> logits          head output                        mtp_logits__act
                     -> dlogits    CE backward result                  mtp_logits__dy
                -> head input      final_layernorm output              mtp_head__act / __dy
      -> mtp.layers.0.final_layernorm   input grad -> transformer out  mtp_head__inputgrad
      -> mtp.layers.0.transformer_layer -> its input grad              mtp_eh_proj__outgrad0
      -> mtp.layers.0.eh_proj           cat output grad                mtp_concat__dy
      -> torch.cat split -> mtp.layers.0.hnorm   (shared tensor)       mtp_shared__*
                         -> mtp.layers.0.enorm   (own branch)          mtp_emb__*

``hnorm`` consumes the shared decoder output - the very tensor ``model.norm`` also consumes; the
framework adds the two ``RmsNormGrad`` input gradients on that tensor outside any kernel.
``enorm`` consumes the token-embedding branch and is the unrelated third normalisation.
"""

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

#: ``module path`` -> ``(published base, output role, output-gradient role, input base or None)``
TARGETS = (
    ('mtp.layers.0.hnorm', 'mtp_shared', 'out', 'dy', None),
    ('mtp.layers.0.enorm', 'mtp_emb', 'out', 'dy', None),
    ('mtp.layers.0.eh_proj', 'mtp_eh_proj', 'out', 'dy', 'mtp_concat'),
    ('mtp.layers.0.final_layernorm', 'mtp_head', 'act', 'dy', None),
    # the trunk side of the same backward pass, so an aligned variant can be judged against the
    # already-published production taps and against the shared-node accumulation
    ('model.norm', 'model_norm', 'out', 'dy', None),
    ('lm_head', 'lm_head', 'out', 'dy', None),
    ('model.layers.1.mlp', 'l1_mlp', 'out', 'dy', None),
)

#: canonical key -> extra published names holding exactly the same bytes (no extra file)
ALIASES = {
    'mtp_shared__x': ('mtp_l0_hnorm__x',),
    'mtp_shared__out': ('mtp_l0_hnorm__out',),
    'mtp_shared__dy': ('mtp_l0_hnorm__dy',),
    'mtp_shared__inputgrad': ('mtp_l0_hnorm__inputgrad',),
    'mtp_shared__gamma': ('mtp_l0_hnorm__gamma',),
    'mtp_shared__dw': ('mtp_l0_hnorm__dw',),
    'mtp_shared__rstd': ('mtp_l0_hnorm__rstd',),
    'mtp_emb__x': ('mtp_l0_enorm__x',),
    'mtp_emb__out': ('mtp_l0_enorm__out',),
    'mtp_emb__dy': ('mtp_l0_enorm__dy',),
    'mtp_emb__inputgrad': ('mtp_l0_enorm__inputgrad',),
    'mtp_emb__gamma': ('mtp_l0_enorm__gamma',),
    'mtp_emb__dw': ('mtp_l0_enorm__dw',),
    'mtp_emb__rstd': ('mtp_l0_enorm__rstd',),
    'mtp_concat__out': ('mtp_eh_proj__x',),
    'mtp_concat__dy': ('mtp_eh_proj__inputgrad',),
    'mtp_eh_proj__out': ('mtp_transformer_in__act',),
    'mtp_eh_proj__dy': ('mtp_transformer_in__dy',),
    'mtp_head__act': ('mtp_head_input__act', 'mtp_l0_final_layernorm__out'),
    'mtp_head__dy': ('mtp_head_input__dy', 'mtp_l0_final_layernorm__outgrad'),
    'mtp_head__x': ('mtp_transformer_out__x',),
    'mtp_head__inputgrad': ('mtp_transformer_out__dy',),
    'mtp_head__gamma': ('mtp_l0_final_layernorm__gamma',),
    'mtp_head__dw': ('mtp_l0_final_layernorm__dw',),
    'mtp_head__rstd': ('mtp_l0_final_layernorm__rstd',),
    'mtp_layer_loss__val': ('mtp_layer_loss_raw',),
    'mtp_loss__val': ('mtp_loss_total',),
    'mtp_logits__act': ('mtp_logits_act',),
}

#: tensors at or below this many bytes also get a source-dtype ``.npy`` copy
NPY_LIMIT = 64 << 20

#: hash/write block size
BLOCK = 32 << 20

_STRIP = ('_checkpoint_wrapped_module.',)

#: ``mtp_pending`` is raised by the depth output norm's forward (the MTP head input exists) and
#: consumed by the next projection call, so the MTP objective is identified without patching any
#: method of the model instance - a class whose module tree is wrapped by FSDP2 still routes here.
_STATE = {'mtp_pending': False}


def _local(value):
    """Return the local value of a possibly distributed tensor."""
    try:
        from hyper_parallel.core.optimizer.dtensor_compat import (  # pylint: disable=C0415
            to_local_if_dtensor)
        return to_local_if_dtensor(value)
    except Exception:  # pylint: disable=broad-except
        return value


def _clean(name):
    for prefix in _STRIP:
        name = name.replace(prefix, '')
    return name


def _raw_of(array, dtype):
    """Return the little-endian, headerless payload of ``array`` in its own dtype."""
    if 'bfloat16' in dtype:
        bits = array.view(torch.int16).numpy().view(np.uint16)
        return np.ascontiguousarray(bits).astype('<u2')
    if dtype == 'float32':
        return np.ascontiguousarray(array.numpy()).astype('<f4')
    if dtype == 'float16':
        return np.ascontiguousarray(array.numpy()).astype('<f2')
    if dtype == 'int64':
        return np.ascontiguousarray(array.numpy()).astype('<i8')
    if dtype == 'int32':
        return np.ascontiguousarray(array.numpy()).astype('<i4')
    raise ValueError(f'unsupported tap dtype {dtype}')


class Store:
    """Write every recorded tensor once, in its own dtype, with a hash and an fp64 moment."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index = {}
        self.fired = set()
        self.skip = set()

    def put(self, key, value, note='', first_only=False):
        """Record ``value`` under ``key``; ``first_only`` keeps the very first occurrence."""
        if value is None or not hasattr(value, 'shape'):
            return
        if first_only and key in self.fired:
            return
        if key in self.skip:
            return
        try:
            self._write(key, value, note)
            self.fired.add(key)
        except Exception as error:  # pylint: disable=broad-except
            self.skip.add(key)
            print(f'[MTPTAP] tap {key} failed: {type(error).__name__}: {error}', flush=True)

    def _write(self, key, value, note):
        tensor = _local(value).detach()
        device = str(tensor.device)
        dtype = str(tensor.dtype).replace('torch.', '')
        shape = list(tensor.shape)
        raw = _raw_of(tensor.to('cpu'), dtype)
        path = self.directory / (key + '.raw')
        digest = hashlib.sha256()
        flat = memoryview(raw.reshape(-1)).cast('B')
        with open(path, 'wb') as stream:
            for offset in range(0, flat.nbytes, BLOCK):
                block = flat[offset:offset + BLOCK]
                stream.write(block)
                digest.update(block)
        # fp64 moments computed in blocks, so a 2.5 GiB tensor never becomes an 8-byte copy
        values = raw.reshape(-1)
        count = int(values.size)
        total = 0.0
        smallest = math.inf
        largest = -math.inf
        absmax = 0.0
        for offset in range(0, count, 1 << 22):
            block = values[offset:offset + (1 << 22)].astype(np.float64)
            total += float((block * block).sum())
            smallest = min(smallest, float(block.min()))
            largest = max(largest, float(block.max()))
            absmax = max(absmax, float(np.abs(block).max()))
        record = {'key': key, 'dtype': dtype, 'shape': shape, 'numel': count,
                  'order': 'C row-major', 'device': device, 'note': note,
                  'sha256_of_raw_bytes': digest.hexdigest(),
                  'raw_file': key + '.raw', 'raw_bytes': int(flat.nbytes),
                  'fp64_ssum': total, 'fp64_rms': math.sqrt(total / count) if count else None,
                  'min': smallest, 'max': largest, 'absmax': absmax}
        if dtype == 'bfloat16':
            record['reconstruction'] = (
                f"(np.fromfile('{key}.raw', dtype='<u2').astype(np.uint32) << 16)"
                f".view('<f4').reshape({shape})")
        if raw.nbytes <= NPY_LIMIT:
            np.save(str(self.directory / (key + '.npy')), raw)
            record['npy'] = key + '.npy'
        else:
            np.save(str(self.directory / (key + '.head.npy')), raw.reshape(-1)[:4096].copy())
            record['head_npy'] = key + '.head.npy'
        (self.directory / (key + '.json')).write_text(
            json.dumps(record, indent=2, sort_keys=True) + '\n')
        self.index[key] = record
        print(f'[MTPTAP] {key:32s} {dtype:9s} {str(shape):20s} '
              f'{record["sha256_of_raw_bytes"][:16]} rms={record["fp64_rms"]:.6g}', flush=True)

    def write(self):
        """Persist the index of everything recorded, plus the alias map."""
        (self.directory / 'index.json').write_text(json.dumps(
            {'taps': sorted(self.index), 'skipped': sorted(self.skip),
             'aliases': {key: list(names) for key, names in ALIASES.items()},
             'tensors': self.index}, indent=2, sort_keys=True) + '\n')


def _module_hook(store, base, out_role, grad_role, input_base):
    """Return the forward hook of one registered MTP sub-module."""

    def hook(module, args, output):
        if base == 'mtp_head':
            # the MTP depth output is the head input; the next projection call is the MTP one
            _STATE['mtp_pending'] = True
        if args and hasattr(args[0], 'shape'):
            activation = args[0]
            store.put(f'{input_base}__out' if input_base else f'{base}__x', activation,
                      note=f'input operand of {module.__class__.__name__}', first_only=True)
            if activation.requires_grad:
                key = f'{input_base}__dy' if input_base else f'{base}__inputgrad'
                activation.register_hook(lambda gradient, k=key: store.put(
                    k, gradient, note=f'gradient wrt the input of {module.__class__.__name__}'))
        values = output if isinstance(output, tuple) else (output,)
        for position, value in enumerate(values):
            if value is None or not hasattr(value, 'shape'):
                continue
            if position == 0:
                store.put(f'{base}__{out_role}', value, note='module output', first_only=True)
            if value.requires_grad:
                key = f'{base}__{grad_role}' if position == 0 else f'{base}__outgrad{position}'
                value.register_hook(lambda gradient, k=key: store.put(
                    k, gradient, note='gradient arriving at the module output'))
        weight = getattr(module, 'weight', None)
        if isinstance(weight, torch.nn.Parameter):
            store.put(f'{base}__gamma', weight, note='module weight', first_only=True)
            weight.register_hook(lambda gradient, k=f'{base}__dw': store.put(
                k, gradient, note='gradient accumulated into the module weight'))
        epsilon = getattr(module, 'variance_epsilon', getattr(module, 'eps', None))
        if epsilon is not None and args and hasattr(args[0], 'shape'):
            try:
                import torch_npu  # pylint: disable=C0415
                with torch.no_grad():
                    _, rstd = torch_npu.npu_rms_norm(_local(args[0]).float(),
                                                     _local(weight).float(), epsilon=float(epsilon))
                store.put(f'{base}__rstd', rstd, first_only=True,
                          note=f'rstd recomputed in-process with the same fused npu_rms_norm '
                               f'(eps={epsilon}); forward-value check only')
            except Exception as error:  # pylint: disable=broad-except
                print(f'[MTPTAP] rstd for {base} unavailable: {type(error).__name__}: {error}',
                      flush=True)
        return None

    return hook


def install_projection_taps(model, store):
    """Tap the chunked MTP head/CE by wrapping ``projected_cross_entropy`` for the MTP call.

    ``_projection_loss`` delegates to the module-level ``projected_cross_entropy``, so patching
    that global catches the call wherever the model tree is wrapped.  The MTP call is recognised
    by the flag the depth output norm raises just before it (the LM objective is scored earlier,
    straight after ``model.norm``), which keeps the identification independent of how the module
    tree is composed.

    The head/loss hook pair is installed on the callables ``projected_cross_entropy`` itself
    invokes inside its checkpointed ``project_and_score``, so the recorded logits and dlogits are
    the very tensors the recomputation produces.
    """
    import hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 as modeling  # pylint: disable=C0415

    real_projected = modeling.projected_cross_entropy

    def head_with_taps(head):
        def call(states):
            logits = head(states)
            suffix = '' if states.storage_offset() == 0 else f'__chunk{states.storage_offset()}'
            store.put(f'mtp_logits__act{suffix}', logits, first_only=True,
                      note=f'MTP depth-0 logits = head(mtp final_layernorm out); head dtype '
                           f'{logits.dtype}')
            if logits.requires_grad:
                logits.register_hook(lambda gradient, s=suffix: store.put(
                    f'mtp_logits__dy{s}', gradient,
                    note='cross-entropy backward result, i.e. d(mtp loss)/d(mtp logits)'))
            return logits
        return call

    def loss_with_taps(loss_fn):
        def call(**loss_kwargs):
            value = loss_fn(**loss_kwargs)
            store.put('mtp_layer_loss__val', value.detach(), first_only=True,
                      note='per-chunk MTP token-mean CE before the mtp_loss_factor weighting')
            if value.requires_grad:
                value.register_hook(lambda gradient: store.put(
                    'mtp_layer_loss__dy', gradient,
                    note='gradient arriving at the per-chunk MTP CE scalar; HP multiplies the '
                         'summed loss by mtp_loss_factor/depths in the forward, so this equals '
                         'that factor'))
            return value
        return call

    def wrapped(hidden_states, targets, *, head, loss_fn, **kwargs):
        if not _STATE['mtp_pending']:
            return real_projected(hidden_states, targets, head=head, loss_fn=loss_fn, **kwargs)
        print(f'[MTPTAP] instrumenting the MTP projection call '
              f'(hidden {list(hidden_states.shape)} {hidden_states.dtype}, '
              f'targets {list(targets.shape)})', flush=True)
        return real_projected(hidden_states, targets, head=head_with_taps(head),
                              loss_fn=loss_with_taps(loss_fn), **kwargs)

    modeling.projected_cross_entropy = wrapped
    print('[MTPTAP] projected_cross_entropy wrapped for the MTP call', flush=True)


def _loss_owner(model):
    """Return the class that defines ``compute_jt_losses`` for this module tree."""
    for _, module in model.named_modules():
        for klass in type(module).__mro__:
            if 'compute_jt_losses' in klass.__dict__:
                return klass
    return None


def install_loss_hooks(model, store):
    """Record the named 0-d losses and the gradient the trainer delivers to each of them."""
    owner = _loss_owner(model)
    if owner is None:
        print('[MTPTAP] no compute_jt_losses owner found; named-loss taps skipped', flush=True)
        return
    original = owner.compute_jt_losses

    def wrapped(self, *args, **kwargs):
        losses = original(self, *args, **kwargs)
        for name, value in losses.items():
            short = {'mtp_loss': 'mtp_loss', 'lm_loss': 'lm_loss',
                     'aux_loss': 'aux_loss'}.get(name)
            if short is None or not hasattr(value, 'shape'):
                continue
            store.put(f'{short}__val', value.detach(), first_only=True,
                      note=f'the {name} 0-d loss handed to the trainer')
            if value.requires_grad:
                value.register_hook(lambda gradient, k=short: store.put(
                    k + '__dy', gradient, first_only=True,
                    note='gradient arriving at the named 0-d loss from the trainer association'))
        return losses

    owner.compute_jt_losses = wrapped
    print(f'[MTPTAP] {owner.__name__}.compute_jt_losses wrapped for the named-loss taps', flush=True)


def install_embedding_taps(model, store):
    """Record what the token embedding is asked to embed, call by call.

    ``model.embed_tokens`` is called once for the trunk ids and once for the shifted MTP ids, so
    the raw token ids of both are visible: the difference between the shifted MTP ids and the
    plain one-token shift of the trunk ids is exactly the document-tail masking HP applies to the
    MTP input.
    """
    calls = {'n': 0}

    def hook(module, args, output):  # pylint: disable=unused-argument
        if not args or not hasattr(args[0], 'dtype'):
            return None
        key = f'embed_ids__call{calls["n"]}'
        calls['n'] += 1
        store.put(key, args[0], note='token ids handed to model.embed_tokens; call 0 is the trunk, '
                                     'call 1 the shifted MTP input', first_only=True)
        return None

    hits = []
    for name, module in model.named_modules():
        if _clean(name) == 'model.embed_tokens':
            hits.append(name)
            module.register_forward_hook(hook)
    print(f'[MTPTAP] embedding id taps attached: {hits}', flush=True)


def install_alignment(model):
    """Optionally drop the document/segment-boundary zeroing of the MTP input ids.

    HP shifts the MTP input with the same ``shift_mtp_sequence`` MindFormers uses, but it then
    additionally zeroes the shifted ids at every document tail
    (``components/modules/mtp.py:131-133``, ``input_ids.masked_fill(sequence_end_mask, 0)``),
    so those three positions embed token 0 instead of the real next token.  MindFormers' roll
    leaves them alone.  ``MTPTAP_ALIGN_SHIFT=1`` removes exactly that, and nothing else: the mask
    still reaches ``iter_mtp_targets`` through ``compute_jt_losses``, so the target masking (and
    the loss denominator) stays the production one.
    """
    if os.environ.get('MTPTAP_ALIGN_SHIFT') != '1':
        return None
    module = model.mtp

    def drop_mask(mod, args, kwargs):
        if kwargs.get('sequence_end_mask') is not None:
            print('[MTPTAP] alignment: dropping sequence_end_mask on the MTP input', flush=True)
            kwargs = dict(kwargs, sequence_end_mask=None)
        return args, kwargs

    handle = module.register_forward_pre_hook(drop_mask, with_kwargs=True)
    print('[MTPTAP] alignment installed: MTP input is now the plain global roll', flush=True)
    return handle


def attach(model, store):
    """Attach every MTP tap to ``model``."""
    hits = []
    for name, module in model.named_modules():
        clean = _clean(name)
        for path, base, out_role, grad_role, input_base in TARGETS:
            if clean == path:
                hits.append(clean)
                module.register_forward_hook(
                    _module_hook(store, base, out_role, grad_role, input_base))
    print(f'[MTPTAP] module taps attached: {hits}', flush=True)
    missing = sorted({path for path, *_ in TARGETS} - set(hits))
    if missing:
        print(f'[MTPTAP] module taps without a module: {missing}', flush=True)
    install_embedding_taps(model, store)
    install_projection_taps(model, store)
    install_loss_hooks(model, store)
    return hits


def attach_trainer(trainer, directory):
    """Install the taps on ``trainer``'s model and return the collector."""
    store = Store(Path(directory))
    attach(trainer.base.model, store)
    return store
