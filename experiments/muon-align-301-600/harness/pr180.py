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
"""Fix 3 — PR #180: form the token weight before scaling the local loss.

The upstream trainer builds the backward scale as ``(cur_loss * cur_token_len)
/ all_reduced_len * dp_size * cp_size``.  Autograd therefore evaluates the
gradient of the named loss as ``(1 / all_reduced_len) * cur_token_len`` — a
*division* first and a *multiplication* afterwards.  For the alignment data the
single-rank microbatch fills the whole step (``cur_token_len ==
all_reduced_len == 3872``), where ``fl(fl(1 / 3872) * 3872) = 0x3f7fffff``:
one float32 ulp **below** 1.0.  Every named loss therefore receives a gradient
of ``1 - 2**-24`` instead of ``1.0``, which scales the whole MTP backward pass
by that factor.

PR #180 reorders the expression so the token weight is formed *first*:
``backward_scale = cur_token_len / all_reduced_len * dp_size * cp_size`` and
``backward_loss = cur_loss * backward_scale``.  The division now happens on the
token counts (``3872 / 3872 == 1.0`` exactly), so the gradient of a full
single-rank microbatch is the exact identity.

This module applies the PR as a *runtime* monkey patch: the repository
(``/workspace/worktree/align-l2-f`` and ``/workspace/hyper-parallel``) is never
modified.  ``hyper_parallel.trainer.base`` imports ``mean_global_loss`` by name
at module import time, so the replacement has to happen in every module that
already holds a reference to the original function object.
"""
from __future__ import annotations

import sys
import threading

#: the one line the PR changes, kept verbatim for the diff in the report
UPSTREAM_LINES = (
    'local_weighted_loss = cur_loss * cur_token_len',
    'backward_loss = local_weighted_loss / all_reduced_len * device_mesh.dp_size * device_mesh.dp_size',
)


def mean_global_loss_pr180(losses, current_token_counts, step_token_counts, device_mesh):
    """``hyper_parallel.trainer.runtime.metrics.mean_global_loss`` with PR #180 applied.

    Identical to the upstream function except for the two lines that form the backward scale;
    the all-reduced (reported) loss is untouched.
    """
    import torch  # pylint: disable=C0415
    import torch.distributed as dist  # pylint: disable=C0415
    from hyper_parallel.trainer.runtime.distributed import all_reduce  # pylint: disable=C0415

    loss_dict = {}
    dp_cp_mesh = device_mesh.dp_cp_mesh
    dp_cp_group = dp_cp_mesh.get_group() if dp_cp_mesh is not None else None
    sequence_parallel = device_mesh.sequence_parallel
    sequence_parallel_group = None
    sequence_parallel_size = 1
    if sequence_parallel:
        if device_mesh.device_mesh is None or 'tp' not in device_mesh.device_mesh.mesh_dim_names:
            raise ValueError("Sequence parallelism requires a DeviceMesh with a 'tp' dimension.")
        sequence_parallel_group = device_mesh.device_mesh['tp'].get_group()
        sequence_parallel_size = dist.get_world_size(group=sequence_parallel_group)
    if isinstance(losses, torch.Tensor):  # text loss only
        losses = {'foundation_loss': losses}
    for key, cur_loss in losses.items():
        loss_name = key.split('_loss', maxsplit=1)[0]  # foundation/image_decoder/**
        cur_token_len = current_token_counts[f'{loss_name}_tokens']
        if sequence_parallel:
            cur_token_len = all_reduce(cur_token_len.item(), op='sum',
                                       group=sequence_parallel_group)
        all_reduced_len = all_reduce(
            step_token_counts[f'{loss_name}_tokens'].item(),
            op='sum',
            group=dp_cp_group,
        )
        if all_reduced_len != 0:
            local_weighted_loss = cur_loss * cur_token_len
            # PR #180: form the token weight first so a full single-rank microbatch has an exact
            # identity gradient.
            backward_scale = (cur_token_len / all_reduced_len
                              * device_mesh.dp_size * device_mesh.cp_size)
            backward_loss = cur_loss * backward_scale
            global_weighted_loss = all_reduce(
                local_weighted_loss.detach().item(),
                op='sum',
                group=dp_cp_group,
            )
            global_mean = cur_loss.new_tensor(global_weighted_loss / all_reduced_len)
            cur_loss = backward_loss + global_mean - backward_loss.detach()
            _note(loss_name, cur_token_len, all_reduced_len, device_mesh, backward_scale,
                  backward_loss)
        else:
            if not torch.allclose(cur_loss, torch.zeros_like(cur_loss)):
                raise ValueError(
                    f'The all_reduced_len for {loss_name}_tokens is 0, but the cur_loss is not 0: '
                    f'{cur_loss}'
                )
        if sequence_parallel:
            cur_loss = cur_loss / sequence_parallel_size
        loss_dict[key] = cur_loss
    return loss_dict


_LOCK = threading.Lock()
_NOTED = set()


def _bits(value):
    """Return the fp32 bit pattern of ``value`` as ``0x%08x``."""
    import struct  # pylint: disable=C0415

    if hasattr(value, 'item'):
        value = value.item()
    return '0x%08x' % struct.unpack('<I', struct.pack('<f', float(value)))[0]


def _note(loss_name, cur_token_len, all_reduced_len, device_mesh, backward_scale, backward_loss):
    """Print the token counts and the backward scale once per loss name (evidence for the report)."""
    with _LOCK:
        if loss_name in _NOTED:
            return
        _NOTED.add(loss_name)
    scale = backward_scale.item() if hasattr(backward_scale, 'item') else backward_scale
    grad = backward_loss.grad_fn
    del grad
    print(f'[PR180] {loss_name}: cur_token_len={cur_token_len} all_reduced_len={all_reduced_len} '
          f'dp={device_mesh.dp_size} cp={device_mesh.cp_size} backward_scale={scale!r} '
          f'bits={_bits(scale)} grad_wrt_named_loss={_bits(scale)}', flush=True)


def apply(verbose=True):
    """Replace ``mean_global_loss`` in every module that holds a reference to the original.

    Returns the list of module names that were patched.
    """
    import hyper_parallel.trainer.runtime.metrics as metrics  # pylint: disable=C0415

    original = metrics.mean_global_loss
    patched = mean_global_loss_pr180
    replaced = []
    # only hyper_parallel's own modules are inspected: a bare ``getattr`` on every loaded module
    # would wake transformers' lazy modules, whose ``__getattr__`` imports every model package
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith('hyper_parallel'):
            continue
        try:
            current = getattr(module, 'mean_global_loss', None)
        except Exception:  # pylint: disable=broad-except
            continue
        if current is original:
            setattr(module, 'mean_global_loss', patched)
            replaced.append(name)
    if metrics.mean_global_loss is not patched:
        metrics.mean_global_loss = patched
        replaced.append('hyper_parallel.trainer.runtime.metrics (forced)')
    if verbose:
        print(f'[PR180] mean_global_loss patched in {sorted(set(replaced))}', flush=True)
    if not replaced:
        print('[PR180] WARNING: no module held a reference to mean_global_loss', flush=True)
    return sorted(set(replaced))


def describe():
    """Return the three code lines that differ from the upstream implementation."""
    return {
        'upstream': [
            'local_weighted_loss = cur_loss * cur_token_len',
            'backward_loss = local_weighted_loss / all_reduced_len'
            ' * device_mesh.dp_size * device_mesh.cp_size',
        ],
        'with_pr180': [
            'local_weighted_loss = cur_loss * cur_token_len',
            'backward_scale = cur_token_len / all_reduced_len'
            ' * device_mesh.dp_size * device_mesh.cp_size',
            'backward_loss = cur_loss * backward_scale',
        ],
    }
