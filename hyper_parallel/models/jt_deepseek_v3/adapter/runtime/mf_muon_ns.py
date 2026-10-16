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

``install_capture_observer`` optionally records parameter-associated boundaries.
It is disabled by default, leaves FIFO selection unchanged, and performs no file
I/O or content fingerprinting. Observers must copy tensors before returning.
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
        normalized_observer: Optional[Callable[[torch.Tensor], None]] = None,
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
    if normalized_observer is not None:
        normalized_observer(mat_x.mT if transposed else mat_x)
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
        self._capture_observer: Optional[Callable[..., None]] = None
        self._capture_buffer_names: Dict[Tuple[Any, ...], str] = {}
        self._capture_stash_sources: Dict[int, Dict[str, Any]] = {}
        self._capture_blocks: Dict[int, Dict[str, Any]] = {}
        self._capture_active_blocks: List[Dict[str, Any]] = []
        self._capture_momentum_call = 0
        self._capture_ns_call = 0

    def _capture_momentum(self, grads, bufs, updates=None) -> None:
        """Observe live buffers synchronously, without retaining tensor copies."""
        if updates is None:
            self._capture_momentum_call += 1
        for index, (grad, buffer) in enumerate(zip(grads, bufs)):
            name = self._capture_buffer_names.get(_capture_tensor_identity(buffer))
            if name is None:
                raise ValueError("Muon capture encountered an unbound momentum buffer; install after state restore")
            metadata = {"momentum_call": self._capture_momentum_call, "local_index": index}
            if updates is None:
                self._capture_observer("muon_momentum_pre", "muon_m_pre", name, buffer, **metadata)
                self._capture_observer("muon_momentum_pre", "muon_grad_fp32", name, grad, **metadata)
            else:
                self._capture_observer("muon_momentum_post", "muon_m_post", name, buffer, **metadata)
                self._capture_observer(
                    "muon_momentum_post", "muon_local_us_fp32", name, updates[index], **metadata
                )

    # -- hooks -----------------------------------------------------------------
    def momentum_update_fn(self, grads, bufs, momentum1, momentum2, nesterov):
        """MindFormers' ``next_m``/nesterov lane in float32 (``_update_muon_momentum``)."""
        step_buffers = list(bufs)
        state_dtype = bufs[0].dtype
        grads = [grad.to(torch.float32) for grad in grads]
        bufs = [buffer.to(torch.float32) for buffer in bufs]
        if self._capture_observer is not None:
            self._capture_momentum(grads, step_buffers)
        local_us = torch._foreach_add(grads, bufs, alpha=momentum1)
        torch._foreach_mul_(bufs, momentum2)
        torch._foreach_add_(bufs, grads)
        if nesterov:
            torch._foreach_mul_(local_us, momentum1)
            torch._foreach_add_(local_us, grads)
        for state_buffer, updated in zip(step_buffers, bufs):
            state_buffer.copy_(updated.to(state_dtype))
        if self._capture_observer is not None:
            self._capture_momentum(grads, step_buffers, local_us)
        for index, update in enumerate(local_us):
            if update.dim() == 2 and self._shape_classes.get(tuple(update.shape), False):
                queue = self._fp32_inputs.setdefault(tuple(update.shape), [])
                queue.append(update.detach())
                if self._capture_observer is not None:
                    name = self._capture_buffer_names[_capture_tensor_identity(step_buffers[index])]
                    self._capture_stash_sources[id(queue[-1])] = {
                        "source_name": name,
                        "momentum_call": self._capture_momentum_call,
                        "local_index": index,
                    }
                    for discarded in queue[:-4]:
                        self._capture_stash_sources.pop(id(discarded), None)
                del queue[:-4]  # bound leftovers from earlier steps
        return local_us

    def zeropower_fn(self, ns_inputs: torch.Tensor, steps: int, **_: Any) -> torch.Tensor:
        """MindFormers' Newton-Schulz, using the stashed float32 numerator when required."""
        if _trace_on():
            _trace(self, ns_inputs)
        cached = None
        capture = self._capture_observer
        capture_metadata = None
        capture_name = ""
        if capture is not None:
            self._capture_ns_call += 1
            if not self._capture_active_blocks:
                raise ValueError("Muon NS capture has no parameter context; use install_capture_observer")
            blocks = self._capture_active_blocks
            capture_name = blocks[0]["name"] if len(blocks) == 1 else f"muon_batch_{self._capture_ns_call}"
            capture_metadata = {
                "ns_call": self._capture_ns_call,
                "ns_steps": steps,
                "blocks": blocks,
                "fp32_numerator_class": (
                    ns_inputs.dim() == 2 and self._shape_classes.get(tuple(ns_inputs.shape), False)
                ),
                "queue_sources": [
                    dict(self._capture_stash_sources.get(id(candidate), {"source_name": None}))
                    for candidate in self._fp32_inputs.get(tuple(ns_inputs.shape), [])
                ],
                "selected_index": None,
                "selected_source_name": None,
            }
            capture("muon_ns_input", "muon_ns_input", capture_name, ns_inputs, **capture_metadata)
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
                        if capture is not None:
                            source = self._capture_stash_sources.pop(id(cached), {})
                            capture_metadata.update({
                                "selected_index": index,
                                "selected_source_name": source.get("source_name"),
                                "selected_momentum_call": source.get("momentum_call"),
                                "alias_stage": "muon_momentum_post",
                                "alias_kind": "muon_local_us_fp32",
                                "alias_name": source.get("source_name"),
                                "alias_momentum_call": source.get("momentum_call"),
                            })
                        break
                else:
                    self._stash_mismatch += 1
                    print(f'[MFNS] stash mismatch #{self._stash_mismatch} '
                          f'shape={tuple(ns_inputs.shape)} (no matching numerator; using bf16)',
                          flush=True)
        numerator = ns_inputs if cached is None else cached
        if capture is not None:
            if cached is None:
                capture_metadata.update({
                    "alias_stage": "muon_ns_input",
                    "alias_kind": "muon_ns_input",
                    "alias_name": capture_name,
                    "alias_ns_call": self._capture_ns_call,
                })
            capture("muon_ns_numerator", "muon_ns_numerator", capture_name, numerator, **capture_metadata)
        if capture is None:
            return zeropower_via_newtonschulz5_mf(numerator, steps=steps)
        output_metadata = {
            key: value for key, value in capture_metadata.items() if not key.startswith("alias_")
        }

        def observe_normalized(tensor):
            capture("muon_ns_normalized", "muon_ns_normalized", capture_name, tensor,
                    normalization_coordinates="ns_input", **output_metadata)

        output = zeropower_via_newtonschulz5_mf(
            numerator, steps=steps, normalized_observer=observe_normalized
        )
        capture("muon_ns_output", "muon_ns_output", capture_name, output, **output_metadata)
        return output


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


def _capture_tensor_identity(tensor: torch.Tensor) -> Tuple[Any, ...]:
    """Identify a live local tensor by address, never by its contents."""
    local = tensor.to_local() if hasattr(tensor, "to_local") else tensor
    return str(local.device), local.data_ptr(), local.storage_offset()


def _install_leaf_capture(
        model: torch.nn.Module,
        optimizer: Any,
        numerics: MFMuonNumerics,
        observer: Callable[..., None],
) -> Callable[[], None]:
    """Bind restored buffers and observe the optimizer's real block dispatch."""
    if numerics._capture_observer is not None:
        raise ValueError("Muon capture observer is already installed")
    model_names = {id(parameter): name for name, parameter in model.named_parameters()}
    buffer_names = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = getattr(parameter, "model_name", None) or model_names.get(id(parameter))
            buffer = optimizer.state.get(parameter, {}).get("momentum_buffer")
            if name is None or buffer is None:
                raise ValueError("Muon capture requires named parameters and restored momentum buffers")
            identity = _capture_tensor_identity(buffer)
            if identity in buffer_names:
                raise ValueError(f"Muon capture found aliased momentum buffers for {name}")
            buffer_names[identity] = name

    original_prepare = optimizer._prepare_ns_transform
    original_compute = optimizer._compute_batched_ns_outputs_for_tensors
    original_scaled = optimizer._compute_batched_ns_updates
    own_prepare = optimizer.__dict__.get("_prepare_ns_transform")
    own_compute = optimizer.__dict__.get("_compute_batched_ns_outputs_for_tensors")
    own_scaled = optimizer.__dict__.get("_compute_batched_ns_updates")
    scaled_calls = 0

    def observe_prepare(parameter, ns_input):
        working_input, transform = original_prepare(parameter, ns_input)
        name = getattr(parameter, "model_name", None) or model_names.get(id(parameter))
        if name is None:
            raise ValueError("Muon capture encountered an unnamed NS parameter")
        for block_index, tensor in enumerate(transform.tensors):
            numerics._capture_blocks[id(tensor)] = {
                "name": name,
                "block_index": block_index,
                "shape": list(tensor.shape),
            }
        return working_input, transform

    def observe_compute(tensors, *args, **kwargs):
        previous = numerics._capture_active_blocks
        try:
            numerics._capture_active_blocks = [numerics._capture_blocks[id(tensor)] for tensor in tensors]
            return original_compute(tensors, *args, **kwargs)
        finally:
            numerics._capture_active_blocks = previous
            for tensor in tensors:
                numerics._capture_blocks.pop(id(tensor), None)

    def observe_scaled(p_list, ns_inputs, group, *args, **kwargs):
        nonlocal scaled_calls
        updates = original_scaled(p_list, ns_inputs, group, *args, **kwargs)
        scaled_calls += 1
        applies_lr = group["apply_lr_in_update"]
        kind = "muon_scaled_update_with_negative_lr" if applies_lr else "muon_scaled_update_without_lr"
        for parameter, update in updates.items():
            name = getattr(parameter, "model_name", None) or model_names.get(id(parameter))
            if name is None:
                raise ValueError("Muon capture encountered an unnamed scaled-update parameter")
            observer(
                "muon_scaled_update", kind, name, update,
                scaled_call=scaled_calls,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                matched_adamw_rms=group["matched_adamw_rms"],
                group_step=group["step"],
                apply_lr_in_update=applies_lr,
                parameter_add_alpha=1.0 if applies_lr else -group["lr"],
                update_coordinates="parameter",
            )
        return updates

    def remove() -> None:
        if own_prepare is None:
            del optimizer._prepare_ns_transform
        else:
            optimizer._prepare_ns_transform = own_prepare
        if own_compute is None:
            del optimizer._compute_batched_ns_outputs_for_tensors
        else:
            optimizer._compute_batched_ns_outputs_for_tensors = own_compute
        if own_scaled is None:
            del optimizer._compute_batched_ns_updates
        else:
            optimizer._compute_batched_ns_updates = own_scaled
        numerics._capture_observer = None
        numerics._capture_buffer_names.clear()
        numerics._capture_stash_sources.clear()
        numerics._capture_blocks.clear()
        numerics._capture_active_blocks = []

    numerics._capture_buffer_names = buffer_names
    numerics._capture_observer = observer
    optimizer._prepare_ns_transform = observe_prepare
    optimizer._compute_batched_ns_outputs_for_tensors = observe_compute
    optimizer._compute_batched_ns_updates = observe_scaled
    return remove


def install_capture_observer(
        model: torch.nn.Module,
        optimizer: Any,
        observer: Callable[..., None],
) -> Callable[[], None]:
    """Install an opt-in, synchronous, parameter-associated Muon observer.

    Args:
        model: Model whose parameters carry the optimizer's original FQNs.
        optimizer: Restored leaf, chained, or mixed-precision optimizer.
        observer: ``observer(stage, kind, name, tensor, **metadata)``. Copy or
            serialize before returning: the optimizer later reuses these buffers.
            ``alias_*`` metadata identifies an already observed immutable value.
            Scaled-update records contain the original method's return value:
            ``muon_scaled_update_without_lr`` precedes the parameter add with
            ``alpha=-lr``; the alternate ``with_negative_lr`` kind already
            includes that sign and learning rate. Neither equals MF's positive
            ``adjusted_lr * x_ret`` by definition.

    Returns:
        A callable that removes only these diagnostic wrappers.

    Raises:
        ValueError: No MFMuonNumerics hook or restored parameter association exists.
    """
    leaves = getattr(optimizer, "chained_optimizers", [optimizer])
    removers = []
    try:
        for leaf in leaves:
            numerics = getattr(getattr(leaf, "momentum_update_fn", None), "__self__", None)
            if isinstance(numerics, MFMuonNumerics):
                if not leaf.ns_per_param or leaf.ns_transform_fn is None:
                    raise ValueError("Muon capture requires the mf_muon per-parameter transform path")
                removers.append(_install_leaf_capture(model, leaf, numerics, observer))
        if not removers:
            raise ValueError("Muon capture found no MFMuonNumerics momentum hook")
    except Exception:
        for remove in reversed(removers):
            remove()
        raise

    def remove_all() -> None:
        for remove in reversed(removers):
            remove()

    return remove_all


__all__ = [
    "MFMuonNumerics", "install_capture_observer", "make_mf_muon_numerics", "zeropower_via_newtonschulz5_mf"
]
