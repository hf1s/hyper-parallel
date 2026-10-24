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
"""Validate fixed-format offline state before restoring optimizer-owned tensors."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.core.optimizer.adamw import AdamW as HyperAdamW
from hyper_parallel.core.optimizer.dtensor_compat import is_dtensor, to_local_if_dtensor
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer


_ADAM_FIELDS = {"adam_m": "exp_avg", "adam_v": "exp_avg_sq"}
_MUON_FIELDS = {"muon_m": "momentum_buffer"}
_LEAF_TYPES = (torch.optim.AdamW, HyperAdamW, Muon)
_WRAPPER_TYPES = (ChainedOptimizer, Float16OptimizerWithFloat16Params)


@dataclass
class _Target:
    """One optimizer parameter and its registered model identity."""

    optimizer: Any
    group: dict[str, Any]
    parameter: torch.Tensor
    model_parameter: torch.Tensor
    name: str
    fields: dict[str, str]


def _local(tensor: torch.Tensor) -> torch.Tensor:
    """Reject layouts that do not hold an entire single-device parameter."""
    if is_dtensor(tensor) and tensor.device_mesh.size() != 1:
        raise ValueError("Offline state materialization supports only a single-device mesh")
    local = to_local_if_dtensor(tensor)
    if tuple(local.shape) != tuple(tensor.shape):
        raise ValueError("Offline state requires a complete, unsharded local parameter")
    return local


def _collect_targets(model: torch.nn.Module, optimizer: Any) -> tuple[list[Any], list[_Target]]:
    """Resolve recognized wrappers and verify their identity mappings."""
    names = {parameter: name for name, parameter in model.named_parameters()}
    main_to_model = {}
    leaves, visited = [], set()

    def _visit(current: Any) -> None:
        if id(current) in visited:
            raise ValueError("Optimizer composition contains a duplicate or cycle")
        visited.add(id(current))
        if type(current) in _LEAF_TYPES:
            leaves.append(current)
            return
        if type(current) not in _WRAPPER_TYPES:
            raise ValueError(f"Unsupported optimizer or wrapper: {type(current).__name__}")
        for main, registered in current.model_param_by_optimizer_param.items():
            if registered not in names:
                raise ValueError("Optimizer main parameter maps outside model.named_parameters()")
            if main in main_to_model and main_to_model[main] is not registered:
                raise ValueError("Conflicting main-to-model parameter identities")
            main_to_model[main] = registered
        if isinstance(current, Float16OptimizerWithFloat16Params):
            _visit(current.optimizer)
        else:
            for child in current.optimizers_dict.values():
                _visit(child)

    _visit(optimizer)
    targets, parameters, registered_parameters = [], set(), set()
    for leaf in leaves:
        fields = _MUON_FIELDS if isinstance(leaf, Muon) else _ADAM_FIELDS
        for group in leaf.param_groups:
            if group.get("amsgrad", False):
                raise ValueError("The MF offline format does not contain AMSGrad maximum second moments")
            for parameter in group["params"]:
                registered = main_to_model.get(parameter, parameter)
                if registered not in names:
                    raise ValueError("Optimizer parameter has no verified model parameter identity")
                if parameter in parameters or registered in registered_parameters:
                    raise ValueError("A model parameter is owned by more than one optimizer entry")
                if parameter.dtype != torch.float32:
                    raise ValueError("FP32 source state requires FP32 optimizer parameters or FP32 main parameters")
                _local(parameter)
                _local(registered)
                if tuple(parameter.shape) != tuple(registered.shape):
                    raise ValueError("Main and model parameter shapes differ")
                targets.append(_Target(leaf, group, parameter, registered, names[registered], fields))
                parameters.add(parameter)
                registered_parameters.add(registered)
    if not targets:
        raise ValueError("No supported optimizer parameters to restore")
    return leaves, targets


def _array(arrays: Mapping[str, np.ndarray], key: str, shape: tuple[int, ...]) -> np.ndarray:
    """Require one finite FP32 tensor with exactly the target shape."""
    if key not in arrays:
        raise ValueError(f"Missing required source tensor: {key}")
    value = np.asarray(arrays[key])
    if value.dtype != np.dtype("float32") or value.shape != shape:
        raise ValueError(
            f"Source tensor {key} must have dtype float32 and shape {shape}; got {value.dtype}, {value.shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite source tensor: {key}")
    return value


def _validate_arrays(targets, model_arrays, optimizer_arrays):
    """Use the destination schema, not discovered input keys, as the denominator."""
    required = {f"{prefix}.{target.name}" for target in targets for prefix in target.fields}
    by_name = {target.name: target for target in targets}
    for target in targets:
        shape = tuple(target.parameter.shape)
        _array(model_arrays, target.name, shape)
        for prefix in target.fields:
            value = _array(optimizer_arrays, f"{prefix}.{target.name}", shape)
            if prefix == "adam_v" and np.any(value < 0):
                raise ValueError(f"Negative Adam second moment: {target.name}")
    for key in set(optimizer_arrays) - required:
        if not isinstance(key, str):
            raise ValueError("Optimizer archive keys must be strings")
        prefix, separator, name = key.partition(".")
        target = by_name.get(name)
        if not separator or target is None:
            raise ValueError(f"Unconsumed optimizer tensor: {key}")
        inactive = _ADAM_FIELDS if target.fields is _MUON_FIELDS else _MUON_FIELDS
        value = np.asarray(optimizer_arrays[key])
        if prefix not in inactive or value.size != 1 or value.dtype != np.dtype("float32"):
            raise ValueError(f"Invalid inactive-family placeholder: {key}")
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite inactive-family placeholder: {key}")
    return len(required)


def _new_moment(parameter: torch.Tensor, value: np.ndarray) -> torch.Tensor:
    """Keep the parameter's DTensor layout while filling its complete local state."""
    moment = torch.empty_like(parameter, dtype=torch.float32)
    local = _local(moment)
    local.copy_(torch.as_tensor(value.copy(), device=local.device))
    return moment


def restore_optimizer_state(
        model: torch.nn.Module,
        optimizer: Any,
        model_arrays: Mapping[str, np.ndarray],
        optimizer_arrays: Mapping[str, np.ndarray],
        step: int,
) -> int:
    """Restore every required moment and optimizer-owned FP32 parameter.

    The caller loads the model's remaining parameters and buffers separately.
    Invalid input is rejected before changing parameters, moments, or counters.
    Runtime device failures during the final writes are not rolled back.

    Args:
        model: Model whose registered parameter identities define source names.
        optimizer: Supported AdamW/Muon optimizer or recognized HP composition.
        model_arrays: Lossless FP32 source model tensors, including main weights.
        optimizer_arrays: Converted moment tensors and inactive-family placeholders.
        step: Completed update count represented by the source checkpoint.

    Returns:
        Number of required moment tensors restored.

    Raises:
        ValueError: If an input, optimizer, parameter mapping, or layout is unsupported.
    """
    if (not isinstance(step, int) or isinstance(step, bool)) or step < 0:
        raise ValueError("Source step must be a nonnegative integer")
    leaves, targets = _collect_targets(model, optimizer)
    count = _validate_arrays(targets, model_arrays, optimizer_arrays)
    if any(isinstance(leaf, torch.optim.AdamW) for leaf in leaves) and step >= 2**24:
        raise ValueError("Source step cannot be incremented exactly by the Torch AdamW FP32 counter")
    prepared = []
    with torch.no_grad():
        for target in targets:
            parameter = target.parameter
            state = {field: _new_moment(parameter, np.asarray(optimizer_arrays[f"{prefix}.{target.name}"]))
                     for prefix, field in target.fields.items()}
            if isinstance(target.optimizer, torch.optim.AdamW):
                on_device = target.group.get("capturable", False) or target.group.get("fused", False)
                state["step"] = torch.tensor(step, dtype=torch.float32,
                                             device=_local(parameter).device if on_device else "cpu")
            source = torch.as_tensor(np.asarray(model_arrays[target.name]).copy(), device=_local(parameter).device)
            model_copy = source.to(device=_local(target.model_parameter).device, dtype=target.model_parameter.dtype)
            if not torch.isfinite(model_copy).all():
                raise ValueError(f"Source tensor overflows the model compute dtype: {target.name}")
            prepared.append((target, state, source, model_copy))
        for target, state, source, model_copy in prepared:
            _local(target.parameter).copy_(source)
            if target.parameter is not target.model_parameter:
                _local(target.model_parameter).copy_(model_copy)
            target.optimizer.state[target.parameter] = state
        for leaf in leaves:
            if type(leaf) in (HyperAdamW, Muon):
                for group in leaf.param_groups:
                    group["step"] = step
    return count
