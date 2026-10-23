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
"""Validate the fixed MF callback's non-model checkpoint fields."""

from __future__ import annotations

from typing import Any

import numpy as np


_DROPOUT_SCOPES = (
    "embedding.embedding_dropout",
    "decoder.layers.0.hidden_states_dropout",
    "decoder.layers.1.hidden_states_dropout",
    "mtp.embedding.embedding_dropout",
    "mtp.layers.0.transformer_layer.hidden_states_dropout",
)
_ROUTER_SCOPES = ("decoder.layers.1.mlp.router", "mtp.layers.0.transformer_layer.mlp.router")
_LOSS_MONITORS = ("lm_loss", "mtp_loss", "aux_loss")


def _single_value(src: dict[str, np.ndarray], name: str, *, integer: bool) -> int | float:
    value = src.get(name)
    if not isinstance(value, np.ndarray) or value.size != 1:
        raise ValueError(f"Native checkpoint {name} must be a single-element tensor")
    if integer:
        if value.dtype.kind not in "iu":
            raise ValueError(f"Native checkpoint {name} must be an integer tensor")
    elif value.dtype != np.dtype("float32"):
        raise ValueError(f"Native checkpoint {name} must use FP32")
    if not np.isfinite(value).all():
        raise ValueError(f"Native checkpoint {name} must be finite")
    return value.reshape(()).item()


def _consume(
    remaining: dict[str, np.ndarray], name: str, reason: str, records: list[dict[str, Any]],
) -> None:
    value = remaining.pop(name)
    row = {"name": name, "dtype": str(value.dtype), "shape": list(value.shape), "reason": reason}
    if value.size == 1:
        row["value"] = value.reshape(()).item()
    records.append(row)


def _consume_aliases(remaining: dict[str, np.ndarray], records: list[dict[str, Any]]) -> None:
    if "epoch_num" in remaining and _single_value(remaining, "epoch_num", integer=True) < 0:
        raise ValueError("Native epoch_num must be a nonnegative integer")
    if "global_step" in remaining:
        step = _single_value(remaining, "step_num", integer=True)
        global_step = _single_value(remaining, "global_step", integer=True)
        if step < 0 or global_step != step:
            raise ValueError("Native global_step and step_num must be equal nonnegative integers")
        _consume(remaining, "global_step", "verified optimizer-step alias of step_num", records)
    if "scale_sense" in remaining:
        loss_scale = _single_value(remaining, "loss_scale", integer=False)
        scale_sense = _single_value(remaining, "scale_sense", integer=False)
        if loss_scale != 1.0 or scale_sense != loss_scale:
            raise ValueError("Native loss_scale and scale_sense must agree and use unit loss scaling")
        _consume(remaining, "scale_sense", "verified wrapper alias of unit loss_scale", records)


def _consume_zero_runtime(remaining: dict[str, np.ndarray], records: list[dict[str, Any]]) -> None:
    for scope in _DROPOUT_SCOPES:
        for suffix in ("seed", "offset"):
            name = f"{scope}.{suffix}"
            if name not in remaining:
                continue
            if _single_value(remaining, name, integer=True) != 0:
                raise ValueError(f"Nonzero RNG state is unsupported by this deterministic adapter: {name}")
            _consume(remaining, name, "zero RNG state in the fixed dropout-free recipe", records)
    for scope in _ROUTER_SCOPES:
        for suffix in ("fi_accu", "expert_load"):
            name = f"{scope}.{suffix}"
            if name not in remaining:
                continue
            weight = remaining.get(f"{scope}.weight")
            value = remaining[name]
            if (not isinstance(weight, np.ndarray) or weight.ndim != 2 or weight.shape[0] == 0
                    or value.shape != (weight.shape[0],)
                    or value.dtype not in (np.dtype("float32"), np.dtype("int32"), np.dtype("int64"))):
                raise ValueError(f"Native router statistics must match its expert count: {name}")
            if not np.isfinite(value).all() or np.any(value):
                raise ValueError(f"Native router statistics must be zero after their callback is consumed: {name}")
            _consume(remaining, name, "zero already-consumed router statistics", records)


def normalize_native_checkpoint(src: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Normalize only known runtime fields of the fixed MF train-cell producer.

    Arrays and the caller's mapping remain unchanged. Counter/scaler aliases must
    agree; RNG and router statistics must be zero. Unknown keys remain present so
    the existing model/optimizer mapping rejects unsupported producer state.

    Args:
        src: Tensors read from a native MindSpore safetensors file.

    Returns:
        Shallow tensor mapping and a record of verified non-model fields.

    Raises:
        ValueError: Native runtime state is inconsistent or outside this recipe.
    """
    remaining = dict(src)
    records: list[dict[str, Any]] = []
    _consume_aliases(remaining, records)
    _consume_zero_runtime(remaining, records)
    for name in _LOSS_MONITORS:
        if name in remaining:
            _single_value(remaining, name, integer=False)
            _consume(remaining, name, "finite monitoring accumulator unused by model and optimizer updates", records)
    return remaining, {"source_format": "mf-native", "source_tensor_count": len(src),
                       "mapped_source_tensor_count": len(remaining), "non_model_fields": records}
