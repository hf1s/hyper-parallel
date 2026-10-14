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
"""Attention sub-anchors, the per-token LM loss and the layer-1 router selection.

Everything here observes the existing computation: forward hooks read module inputs and
outputs, the loss wrapper calls the framework loss with the very same arguments, and the
router wrapper returns exactly what the model's own ``route`` returned. The step result is
therefore unchanged.

The two rotary anchors are the only RoPE boundary this model exposes as a module: the full
post-RoPE query and key are built by ``torch.cat`` inside ``_project_attention_inputs`` and
have no module to hook, so ``a5`` is the rotary output for the query rope part and ``a6``
the rotary output for the key rope part (first and second call of that module).
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from activation_anchors import _local, _statistics

SUB_ANCHORS = (
    ('a1', 'model.layers.0.self_attn.q_a_layernorm', 'out'),
    ('a2', 'model.layers.0.self_attn.q_b_proj', 'out'),
    ('a3', 'model.layers.0.self_attn.kv_a_layernorm', 'out'),
    ('a4', 'model.layers.0.self_attn.kv_b_proj', 'out'),
    ('a7', 'model.layers.0.self_attn.o_proj', 'in'),
    ('a8', 'model.layers.0.self_attn.o_proj', 'out'),
)
ROTARY = 'model.layers.0.self_attn.explicit_rotary'
ROTARY_ANCHORS = ('a5', 'a6')
ROUTER_MOE = 'model.layers.1.mlp'
BITS_POINTS = (
    ('p1', 'model.layers.0.self_attn.q_a_layernorm', 'in'),
    ('p2', 'model.layers.0.self_attn.kv_a_layernorm', 'in'),
    ('p3', 'model.layers.0.self_attn.kv_a_layernorm', 'out'),
    ('p4', 'model.layers.0.self_attn.q_a_layernorm', 'out'),
)
BITS_DIRECTORY_ENV = 'KV_BITS_DIR'
_LOSS_ROWS = 256


def _digest(array: np.ndarray) -> str:
    """Hash an array's own-dtype little-endian bytes."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _per_token_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple:
    """Return the per-token causal-LM loss and its valid mask.

    The loss is ``logsumexp(logits) - logits[target]``, reduced in float64 and stored in
    float32. Tokens whose target is negative are ignored, matching the model's own
    ``labels >= 0`` validity rule; their entry is 0.0 and marked in the returned mask.
    """
    value = _local(logits)
    flat = value.reshape(-1, value.shape[-1])
    targets = _local(labels).reshape(-1).to(torch.int64).cpu()
    valid = targets >= 0
    result = torch.zeros(flat.shape[0], dtype=torch.float32)
    for start in range(0, flat.shape[0], _LOSS_ROWS):
        stop = min(start + _LOSS_ROWS, flat.shape[0])
        rows = valid[start:stop]
        if not bool(rows.any()):
            continue
        block = flat[start:stop][rows].to(torch.float64).cpu()
        target = targets[start:stop][rows]
        picked = block.gather(1, target.unsqueeze(1)).squeeze(1)
        result[start:stop][rows] = (torch.logsumexp(block, dim=-1) - picked).to(torch.float32)
    return result, valid


def _install_loss_reader(model, state: dict) -> None:
    """Wrap the framework loss so the first call's per-token loss is recorded."""
    original = model.loss_function

    def loss_function(*args, **kwargs):
        """Call the framework loss unchanged and observe its inputs once."""
        result = original(*args, **kwargs)
        state['loss_calls'] = state.get('loss_calls', 0) + 1
        if 'token_loss' not in state and 'logits' in kwargs and 'shift_labels' in kwargs:
            values, valid = _per_token_loss(kwargs['logits'], kwargs['shift_labels'])
            state['token_loss'] = values
            state['token_loss_valid'] = valid
            state['token_loss_meta'] = {
                'call_index': state['loss_calls'],
                'logits_dtype': str(kwargs['logits'].dtype),
                'logits_shape': list(kwargs['logits'].shape),
                'logits_sha256': _digest(_local(kwargs['logits']).to(torch.float32).contiguous().cpu()
                                         .reshape(-1).view(torch.uint8).numpy()),
                'labels_sha256': _digest(_local(kwargs['shift_labels']).contiguous().cpu()
                                         .reshape(-1).view(torch.uint8).numpy()),
                'num_items_in_batch': float(kwargs.get('num_items_in_batch', -1)),
            }
        return result

    model.loss_function = loss_function


def _install_router_reader(model, state: dict) -> None:
    """Wrap the layer-1 MoE route so its own selection is recorded."""
    modules = dict(model.named_modules())
    moe = modules[ROUTER_MOE]
    original = moe.route

    def route(hidden):
        """Call the model's own routing and observe what it selected."""
        indices, weights = original(hidden)
        if 'router' not in state:
            state['router'] = {
                'indices': _local(indices).to(torch.int64).cpu().numpy(),
                'weights': _local(weights).to(torch.float32).cpu().numpy(),
                'padding': int(getattr(moe, 'padding', 0) or 0),
            }
        return indices, weights

    moe.route = route

    def gate_hook(module, inputs, output):
        """Record the FP32 router logits of the layer-1 gate."""
        del module, inputs
        if 'router_logits' not in state:
            state['router_logits'] = _local(output).to(torch.float32).cpu().numpy()

    modules[ROUTER_MOE + '.gate'].register_forward_hook(gate_hook)


def _write_sub_anchors(state: dict, output: Path) -> None:
    """Write the attention sub-anchor inventory."""
    payload = {'order': [anchor for _, anchor, _ in SUB_ANCHORS[:4]] + list(ROTARY_ANCHORS)
               + [anchor for _, anchor, _ in SUB_ANCHORS[4:]],
               'anchors': state['sub_anchors'],
               'notes': {'a5_a6': 'rotary module outputs (query rope part, then key rope part); the '
                                  'concatenated post-RoPE query/key have no module boundary',
                         'a7': 'o_proj input = attention context flattened to [tokens, heads * v_head_dim]',
                         'a8': 'o_proj output = the attention branch, identical to anchor c'}}
    (output / 'sub_anchors_step301.json').write_text(json.dumps(payload, indent=2) + '\n')


def _write_router(state: dict, output: Path) -> None:
    """Write the layer-1 routing selection and its summary."""
    if 'router' not in state:
        print('[ROUTER] the layer-1 MoE route was never called', flush=True)
        return
    indices = state['router']['indices']
    weights = state['router']['weights']
    arrays = {'indices': indices, 'weights': weights, 'padding': np.asarray(state['router']['padding'])}
    if 'router_logits' in state:
        arrays['router_logits'] = state['router_logits']
    np.savez(output / 'router_layer1_step301.npz', **arrays)
    top1 = indices[:, 0] if indices.ndim > 1 else indices
    counts = {str(expert): int((top1 == expert).sum()) for expert in range(int(top1.max()) + 1)}
    report = {
        'step': 301,
        'indices_shape': list(indices.shape), 'indices_dtype': 'int64',
        'indices_sha256': _digest(np.ascontiguousarray(indices)),
        'weights_shape': list(weights.shape), 'weights_dtype': 'float32',
        'weights_sha256': _digest(np.ascontiguousarray(weights)),
        'gate_logits_shape': (list(state['router_logits'].shape)
                              if 'router_logits' in state else None),
        'gate_logits_sha256': (_digest(np.ascontiguousarray(state['router_logits']))
                               if 'router_logits' in state else None),
        'padding_rows': int(state['router']['padding']),
        'top1_expert_counts': counts,
        'weights_sum': float(weights.astype(np.float64).sum()),
        'notes': 'rows are [padding + tokens]; indices/weights are the model route() return values',
    }
    (output / 'router_layer1_step301.json').write_text(json.dumps(report, indent=2) + '\n')


def _write_token_loss(state: dict, output: Path) -> None:
    """Write the per-token LM loss array and its summary."""
    if 'token_loss' not in state:
        print('[TOKEN-LOSS] the framework loss never received logits', flush=True)
        return
    values = state['token_loss'].numpy()
    valid = state['token_loss_valid'].numpy()
    np.save(output / 'token_loss_step301.npy', values)
    report = {
        'step': 301, 'dtype': 'float32', 'shape': list(values.shape),
        'sha256': _digest(values),
        'valid_count': int(valid.sum()),
        'mean_over_valid': float(values[valid].astype(np.float64).mean()) if valid.any() else None,
        'sum_over_valid': float(values[valid].astype(np.float64).sum()),
        'min': float(values[valid].min()) if valid.any() else None,
        'max': float(values[valid].max()) if valid.any() else None,
        'ignored_tokens_are_zero': True,
        'formula': 'logsumexp(logits) - logits[target], float64 reduction stored as float32; '
                   'targets < 0 ignored',
        'loss_calls': state.get('loss_calls'),
        **state['token_loss_meta'],
    }
    (output / 'token_loss_step301.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'[TOKEN-LOSS] step=301 valid={report["valid_count"]} '
          f'mean_over_valid={"%.17g" % report["mean_over_valid"]} sha256={report["sha256"]}',
          flush=True)


def _write_bits(state: dict, directory: Path) -> None:
    """Write the raw bf16 bits of the requested capture points."""
    directory.mkdir(parents=True, exist_ok=True)
    for point, name, point_kind in BITS_POINTS:
        tensor = state['bits'].get(point)
        if tensor is None:
            print(f'[KV-BITS] {point} {name} {point_kind} was not captured', flush=True)
            continue
        value = tensor.contiguous()
        bits = value.view(torch.int16).numpy().astype(np.uint16)
        np.save(directory / f'{point}.bits.npy', bits)
        squares = value.to(torch.float64)
        report = {'point': point, 'name': name, 'dtype': str(value.dtype), 'shape': list(value.shape),
                  'sha256_of_raw_bytes':
                      hashlib.sha256(np.ascontiguousarray(bits).tobytes()).hexdigest(),
                  'fp64_ssum': float((squares * squares).sum())}
        (directory / f'{point}.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'[KV-BITS] {point} {name} {point_kind} dtype={report["dtype"]} '
              f'shape={report["shape"]} sha256={report["sha256_of_raw_bytes"]}', flush=True)


def attach(trainer, output: str | Path) -> dict:
    """Register the sub-anchor hooks, the loss reader and the router reader.

    Args:
        trainer: Built HP trainer holding the model to observe.
        output: Directory receiving the JSON/NPY artifacts.

    Returns:
        The mutable capture state.
    """
    from hyper_parallel.trainer.callbacks.base import Callback  # pylint: disable=C0415

    output = Path(output)
    model = trainer.base.model
    modules = dict(model.named_modules())
    state = {'sub_anchors': [], 'done': set(), 'rotary_calls': 0, 'loss_calls': 0, 'bits': {}}

    def record(anchor: str, name: str, tensor) -> None:
        """Append one sub-anchor record."""
        if tensor is None or not torch.is_tensor(tensor):
            print(f'[SUB-ANCHOR-ERROR] {anchor} {name}: no tensor', flush=True)
            return
        values = _statistics(tensor)
        state['sub_anchors'].append({'anchor': anchor, 'name': name, **values})
        print(f'[SUB-ANCHOR] {anchor} {name} dtype={values["dtype"]} shape={values["shape"]} '
              f'sha256={values["sha256"]} fp64_ssum={"%.17g" % values["fp64_ssum"]}', flush=True)

    def output_hook(anchor: str, name: str):
        """Record a module's output the first time it runs."""

        def hook(module, inputs, result):
            del module, inputs
            if anchor in state['done']:
                return
            state['done'].add(anchor)
            record(anchor, name, result[0] if isinstance(result, (tuple, list)) else result)

        return hook

    def input_hook(anchor: str, name: str):
        """Record a module's first positional input."""

        def hook(module, inputs):
            del module
            if anchor in state['done']:
                return
            state['done'].add(anchor)
            record(anchor, name, inputs[0] if inputs else None)

        return hook

    def rotary_hook(module, inputs, result):
        """Record the rotary output for the query rope part, then for the key rope part."""
        del module, inputs
        if state['rotary_calls'] >= len(ROTARY_ANCHORS):
            return
        anchor = ROTARY_ANCHORS[state['rotary_calls']]
        state['rotary_calls'] += 1
        record(anchor, ROTARY, result[0] if isinstance(result, (tuple, list)) else result)

    def bits_hook(point: str, point_kind: str):
        """Capture one raw tensor for the bit dump."""

        def hook(module, inputs, result=None):
            del module
            if point in state['bits']:
                return
            if point_kind == 'in':
                tensor = inputs[0] if inputs else None
            else:
                tensor = result[0] if isinstance(result, (tuple, list)) else result
            if tensor is not None and torch.is_tensor(tensor):
                state['bits'][point] = _local(tensor).detach().cpu()

        return hook

    for point, name, point_kind in BITS_POINTS:
        if name not in modules:
            raise ValueError(f'Bit-dump module is missing: {name}')
        if point_kind == 'in':
            modules[name].register_forward_pre_hook(bits_hook(point, point_kind))
        else:
            modules[name].register_forward_hook(bits_hook(point, point_kind))
    for anchor, name, point in SUB_ANCHORS:
        if name not in modules:
            raise ValueError(f'Sub-anchor module is missing: {name}')
        if point == 'in':
            modules[name].register_forward_pre_hook(input_hook(anchor, name))
        else:
            modules[name].register_forward_hook(output_hook(anchor, name))
    if ROTARY not in modules:
        raise ValueError(f'Sub-anchor module is missing: {ROTARY}')
    modules[ROTARY].register_forward_hook(rotary_hook)
    _install_loss_reader(model, state)
    _install_router_reader(model, state)
    print(f'[SUB-ANCHOR] registered {len(SUB_ANCHORS) + len(ROTARY_ANCHORS)} sub-anchors, '
          f'the loss reader and the {ROUTER_MOE} router reader', flush=True)

    class Details(Callback):
        """Write the captured details once the step has finished."""

        def on_step_end(self, state_holder, **kwargs):
            """Persist the sub-anchors, the per-token loss and the routing selection."""
            del kwargs
            print(f'[SUB-ANCHOR] step {int(state_holder.global_step)} captured '
                  f'{len(state["sub_anchors"])} sub-anchors', flush=True)
            _write_sub_anchors(state, output)
            _write_token_loss(state, output)
            _write_router(state, output)
            directory = os.environ.get(BITS_DIRECTORY_ENV)
            if directory:
                _write_bits(state, Path(directory))

    trainer.base._callbacks.append(Details(trainer.base))
    return state
