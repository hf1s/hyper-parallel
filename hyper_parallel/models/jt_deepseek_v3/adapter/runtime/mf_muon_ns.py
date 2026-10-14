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
"""MindFormers-exact Muon arithmetic, entirely on the model/adapter side.

Alignment-only: this module exists so that a ``mf_muon`` run reproduces
MindFormers' Muon numerically without touching ``core/optimizer/muon.py``.

It is injected through hooks the framework already exposes:

* ``muon_config["zeropower_fn"]``      -> :func:`make_zeropower_fn`
* ``muon_config["momentum_update_fn"]``-> :func:`make_momentum_update_fn`

Because the framework pre-casts Newton-Schulz inputs to bfloat16 whenever
``fp32_update`` is enabled, the float32 numerator a whole, non-transposed 2-D
block needs cannot be recovered from the tensor the NS hook receives.  The
momentum hook therefore keeps a float32 copy of every such block keyed by its
shape, and the NS hook substitutes it.  See ``_classes`` for how the small set
of alignment parameters is classified.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

_NS_LEGACY_COEFFS_MF = (3.4375, -4.78125, 2.03125)   # bf16 values of (3.4445, -4.7750, 2.0315)
_NS_MF_EPSILON = 1e-7


def _trace_on() -> bool:
    import os
    return os.environ.get("MF_NS_TRACE", "0") == "1"


def _trace(numerics: "MFMuonNumerics", ns_inputs: torch.Tensor) -> None:
    """One line per call: shape, dtype, whether the stashed float32 numerator was used."""
    key = tuple(ns_inputs.shape)
    wants_fp32 = numerics._shape_classes.get(key[-2:], False) if ns_inputs.dim() == 2 else None
    used_stash = wants_fp32 is True and key[-2:] in numerics._fp32_inputs
    print(f"[MFNS] call shape={key} dtype={ns_inputs.dtype} fp32_numerator={wants_fp32} "
          f"stash_hit={used_stash}", flush=True)


def _store_bfloat16(tensor: torch.Tensor) -> torch.Tensor:
    """Store a float32 tensor on the bfloat16 grid (round-to-nearest-even)."""
    return tensor.to(torch.bfloat16).to(torch.float32)


def zeropower_via_newtonschulz5_mf(
        ns_inputs: torch.Tensor,
        steps: int,
        ns_variant: str = "legacy",
        epsilon: float = _NS_MF_EPSILON,
        ns_coefficients: Optional[Sequence[Tuple[float, float, float]]] = None,
) -> torch.Tensor:
    """MindFormers' ``zeropower_via_newtonschulz5_2d``/``_3d`` arithmetic, verbatim."""
    del ns_variant, ns_coefficients
    mat_x = ns_inputs.to(torch.float32)
    transposed = mat_x.size(-2) > mat_x.size(-1)
    if transposed:
        mat_x = mat_x.mT
    if mat_x.dim() > 2:
        norm = _store_bfloat16(_store_bfloat16(mat_x).norm(dim=(-2, -1), keepdim=True) + epsilon)
        mat_x = _store_bfloat16(mat_x / norm)
    else:
        norm = mat_x.norm(dim=(-2, -1), keepdim=True)
        mat_x = _store_bfloat16(mat_x / (norm + epsilon))
    coeff_a, coeff_b, coeff_c = _NS_LEGACY_COEFFS_MF
    for _ in range(steps):
        a_mat = _store_bfloat16(torch.matmul(mat_x, mat_x.mT))
        if mat_x.dim() == 2:
            # MindFormers' 2-D branch binds ``c * a_mat @ a_mat`` left to right: (c * A) @ A.
            inner = _store_bfloat16(torch.matmul(_store_bfloat16(coeff_c * a_mat), a_mat))
        else:
            inner = coeff_c * _store_bfloat16(torch.matmul(a_mat, a_mat))
        b_mat = _store_bfloat16(coeff_b * a_mat + inner)
        mat_x = _store_bfloat16(mat_x * coeff_a + _store_bfloat16(torch.matmul(b_mat, mat_x)))
    if transposed:
        mat_x = mat_x.mT
    return mat_x.to(ns_inputs.dtype)


class MFMuonNumerics:
    """Alignment-only helper that reproduces MindFormers' Muon numerics.

    ``shape_classes`` maps a 2-D tensor shape to ``True`` when MindFormers
    normalises that whole, non-transposed block from its pre-cast float32 value.
    """

    def __init__(self, shape_classes: Dict[Tuple[int, int], bool]):
        self._shape_classes = dict(shape_classes)
        # FIFO per shape: the optimizer computes momentum for a whole parameter list first and
        # only then runs Newton-Schulz per parameter, so a shape-keyed single slot would let
        # same-shape parameters (e.g. the two/three router weights) share one numerator.
        self._fp32_inputs: Dict[Tuple[int, int], List[torch.Tensor]] = {}
        self._stash_mismatch = 0

    # -- hooks -----------------------------------------------------------------
    def momentum_update_fn(self, grads, bufs, momentum1, momentum2, nesterov):
        """MindFormers' ``next_m``/nesterov lane in float32 (``_update_muon_momentum``)."""
        step_buffers = list(bufs)
        state_dtype = bufs[0].dtype
        grads = [grad.to(torch.float32) for grad in grads]
        bufs = [buffer.to(torch.float32) for buffer in bufs]
        local_us = torch._foreach_add(grads, bufs, alpha=momentum1)
        torch._foreach_mul_(bufs, momentum2)
        torch._foreach_add_(bufs, grads)
        if nesterov:
            torch._foreach_mul_(local_us, momentum1)
            torch._foreach_add_(local_us, grads)
        for state_buffer, updated in zip(step_buffers, bufs):
            state_buffer.copy_(updated.to(state_dtype))
        for update in local_us:
            if update.dim() == 2 and self._shape_classes.get(tuple(update.shape), False):
                queue = self._fp32_inputs.setdefault(tuple(update.shape), [])
                queue.append(update.detach())
                del queue[:-4]  # bound leftovers from earlier steps
        return local_us

    def zeropower_fn(self, ns_inputs: torch.Tensor, steps: int, **_: Any) -> torch.Tensor:
        """MindFormers' Newton-Schulz, using the stashed float32 numerator when required."""
        if _trace_on():
            _trace(self, ns_inputs)
        cached = None
        if ns_inputs.dim() == 2 and self._shape_classes.get(tuple(ns_inputs.shape), False):
            queue = self._fp32_inputs.get(tuple(ns_inputs.shape))
            if queue:
                # take the entry that this very parameter produced (order-independent): compare a
                # strided sample in the NS input dtype, which is exact enough to distinguish
                # same-shape numerators without hashing gigabytes every step.
                probe = ns_inputs.reshape(-1)[::1024][:1_000_000]
                for index, candidate in enumerate(queue):
                    if candidate.reshape(-1)[::1024][:probe.numel()].to(ns_inputs.dtype).equal(probe):
                        cached = queue.pop(index)
                        break
                else:
                    self._stash_mismatch += 1
                    print(f'[MFNS] stash mismatch #{self._stash_mismatch} '
                          f'shape={tuple(ns_inputs.shape)} (no matching numerator; using bf16)',
                          flush=True)
        return zeropower_via_newtonschulz5_mf(ns_inputs if cached is None else cached, steps=steps)


def make_mf_muon_numerics(block_groups_fn: Callable[[str, Tuple[int, ...]], Optional[List[Any]],
                                                   ], model: torch.nn.Module) -> MFMuonNumerics:
    """Classify every Muon parameter of ``model`` by MindFormers' numerator rule.

    ``fp32`` numerator requires: no ``muon_split`` transform for the parameter, a 2-D
    tensor and no Newton-Schulz-internal transpose, i.e. ``size(-2) <= size(-1)``.
    """
    classes: Dict[Tuple[int, int], bool] = {}
    for name, param in model.named_parameters():
        shape = tuple(param.shape)
        if len(shape) != 2:
            continue
        transformed = block_groups_fn(name, shape) is not None
        value = (not transformed) and shape[-2] <= shape[-1]
        previous = classes.get(shape)
        if previous is not None and previous != value:
            raise ValueError(
                f"ambiguous Muon numerator class for shape {shape}: {name} wants fp32={value}, "
                f"a previous parameter wants fp32={previous}"
            )
        classes[shape] = value
    return MFMuonNumerics(classes)


__all__ = ["MFMuonNumerics", "make_mf_muon_numerics", "zeropower_via_newtonschulz5_mf"]
