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
"""Read-only forward hooks recording the alignment activation anchors.

The anchors describe one forward pass in order: the token embedding, the first attention
input, the first attention branch, the first MLP branch, the final norm and the logits.
Every hook only reads its module's output, so the forward pass and therefore the step
result stay bit-identical to a run without the hooks.

Where a module returns a tuple (the dual-output norms and the attention branch do), the
first element is the tensor the rest of the model consumes, which is the value the
anchors mean.
"""

import hashlib
import json
from pathlib import Path

import torch

from hyper_parallel.core.optimizer.dtensor_compat import to_local_if_dtensor

ANCHORS = (
    ('a', 'model.embed_tokens'),
    ('b', 'model.layers.0.input_layernorm'),
    ('c', 'model.layers.0.self_attn'),
    ('d', 'model.layers.0.mlp'),
    ('e', 'model.norm'),
    ('f', 'lm_head'),
)
ATTENTIONS = ('model.layers.0.self_attn', 'model.layers.1.self_attn',
              'mtp.layers.0.transformer_layer.self_attn')
FILE_NAME = 'anchors_step301.json'
_CHUNK = 1 << 22


def _local(tensor: torch.Tensor) -> torch.Tensor:
    """Detach a possibly distributed tensor and return its local value."""
    return to_local_if_dtensor(tensor).detach()


def _statistics(tensor: torch.Tensor) -> dict:
    """Hash a tensor's own dtype bytes and reduce it in float64, chunk by chunk."""
    value = _local(tensor).contiguous().cpu()
    digest = hashlib.sha256()
    for chunk in torch.split(value.view(torch.uint8).reshape(-1), 4 * _CHUNK):
        digest.update(chunk.numpy().tobytes())
    total, low, high = 0.0, None, None
    for chunk in torch.split(value.reshape(-1), _CHUNK):
        squares = chunk.to(torch.float64)
        total += float((squares * squares).sum())
        chunk_low, chunk_high = float(squares.min()), float(squares.max())
        low = chunk_low if low is None else min(low, chunk_low)
        high = chunk_high if high is None else max(high, chunk_high)
    return {'dtype': str(value.dtype), 'shape': list(value.shape), 'sha256': digest.hexdigest(),
            'fp64_ssum': total, 'min': low, 'max': high}


def _first(output):
    """Return the tensor downstream consumers see for a possibly tuple output."""
    if isinstance(output, (tuple, list)):
        return output[0] if output else None
    if isinstance(output, dict):
        return next(iter(output.values()), None)
    return output


def attach(trainer, output: str | Path) -> dict:
    """Register the anchor hooks and write their values after the first step.

    Args:
        trainer: Built HP trainer holding the model to observe.
        output: Directory receiving ``anchors_step301.json``.

    Returns:
        The mutable anchor state, for reporting.
    """
    from hyper_parallel.trainer.callbacks.base import Callback  # pylint: disable=C0415

    output = Path(output)
    model = trainer.base.model
    modules = dict(model.named_modules())
    state = {'records': [], 'logits': [], 'done': set(), 'attention_done': set()}
    missing = [name for _, name in ANCHORS if name not in modules]
    if missing:
        raise ValueError(f'Anchor modules are missing from the model: {missing}')

    def anchor_hook(anchor: str, name: str):
        """Record one module's output the first time the model calls it."""

        def record(module, inputs, result):
            del module, inputs
            if anchor in state['done']:
                return
            tensor = _first(result)
            if not torch.is_tensor(tensor):
                print(f'[ANCHOR-ERROR] {anchor} {name}: output is '
                      f'{type(tensor).__name__}', flush=True)
                return
            state['done'].add(anchor)
            values = _statistics(tensor)
            state['records'].append({'anchor': anchor, 'name': name, **values})
            print(f'[ANCHOR] {anchor} {name} dtype={values["dtype"]} shape={values["shape"]} '
                  f'sha256={values["sha256"]} fp64_ssum={"%.17g" % values["fp64_ssum"]}', flush=True)

        return record

    def attention_hook(name: str):
        """Record the pre-clipping per-head maximum logit of one attention module."""

        def record(module, inputs, result):
            del inputs, result
            if name in state['attention_done']:
                return
            statistic = getattr(module, 'max_logits_val', None)
            if statistic is None:
                return
            state['attention_done'].add(name)
            state['logits'].append({'name': name, **_statistics(statistic)})

        return record

    for anchor, name in ANCHORS:
        modules[name].register_forward_hook(anchor_hook(anchor, name))
    for name in ATTENTIONS:
        if name in modules:
            modules[name].register_forward_hook(attention_hook(name))
    print(f'[ANCHOR] registered {len(ANCHORS)} anchors: '
          f'{[name for _, name in ANCHORS]}', flush=True)

    class Anchors(Callback):
        """Write the recorded anchors once the step has finished."""

        def on_step_end(self, state_holder, **kwargs):
            """Persist the anchors captured during this step's forward pass."""
            del kwargs
            step = int(state_holder.global_step)
            payload = {'step': step, 'order': [anchor for anchor, _ in ANCHORS],
                       'anchors': sorted(state['records'], key=lambda row: row['anchor']),
                       'attention_max_logits': state['logits']}
            path = output / FILE_NAME
            path.write_text(json.dumps(payload, indent=2) + '\n')
            print(f'[ANCHOR] wrote {path} with {len(payload["anchors"])} anchors and '
                  f'{len(state["logits"])} attention max-logits at step {step}', flush=True)

    trainer.base._callbacks.append(Anchors(trainer.base))
    return state
