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
"""Read audited manifest-named NPZ artifacts without rewriting tensor data."""

import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np


_FIELDS = {"exp_avg": "adam_m", "exp_avg_sq": "adam_v"}
_ARCHIVES = ("model.npz", "optimizer.npz")


def _nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


class _NamedSlots(Mapping[str, np.ndarray]):
    """Translate names lazily so validation does not retain all moment arrays."""

    def __init__(self, arrays: Mapping[str, np.ndarray], names: dict[str, str]) -> None:
        """Retain the live archive and its validated destination-to-source map."""
        self._arrays = arrays
        self._names = names

    def __getitem__(self, key: str) -> np.ndarray:
        """Read the same source array under its validated destination name."""
        return self._arrays[self._names[key]]

    def __iter__(self) -> Iterator[str]:
        """Enumerate names without reading tensor payloads."""
        return iter(self._names)

    def __len__(self) -> int:
        """Return the number of mapped state tensors."""
        return len(self._names)


def map_manifest_slots(
        arrays: Mapping[str, np.ndarray], manifest: Mapping[str, Any],
) -> Mapping[str, np.ndarray]:
    """Resolve every named AdamW slot, rejecting missing or unconsumed entries.

    Args:
        arrays: Open archive using the manifest's source key names.
        manifest: Conversion metadata containing ``optimizer_manifest``.

    Returns:
        A lazy view with ``adam_m.<parameter>`` and ``adam_v.<parameter>`` keys.

    Raises:
        ValueError: If a mapping is incomplete, ambiguous, or not pure AdamW.
    """
    rows = manifest.get("optimizer_manifest")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Conversion manifest must declare its optimizer parameters")
    names, seen_parameters, seen_slots = {}, set(), set()
    for row in rows:
        if not isinstance(row, dict) or row.get("use_muon") is not False:
            raise ValueError("Manifest input currently supports pure AdamW state only")
        name, state = row.get("name"), row.get("state")
        if not isinstance(name, str) or not name or name in seen_parameters:
            raise ValueError("Manifest optimizer parameter names must be nonempty and unique")
        if not isinstance(state, dict) or set(state) != set(_FIELDS):
            raise ValueError(f"Manifest parameter {name} must declare both AdamW moments")
        seen_parameters.add(name)
        for field, prefix in _FIELDS.items():
            source = state[field]
            if not isinstance(source, str) or not source or source in seen_slots:
                raise ValueError("Each source optimizer tensor must have exactly one destination")
            seen_slots.add(source)
            names[f"{prefix}.{name}"] = source
    archive_names = list(arrays)
    if len(archive_names) != len(set(archive_names)) or set(archive_names) != seen_slots:
        raise ValueError("Optimizer archive keys differ from the complete manifest mapping")
    count = manifest.get("optimizer_state_tensors")
    if not _nonnegative_integer(count) or count != len(names):
        raise ValueError("Manifest optimizer tensor count differs from its slot mapping")
    return _NamedSlots(arrays, names)


def load_conversion_manifest(state_dir: Path, step: int) -> dict[str, Any]:
    """Verify persisted NPZ files and counters before constructing a Trainer.

    File hashes detect changed payloads, not the correctness of their earlier
    coordinate conversion. Data order, batch size, and loss scaling must come
    from separate, explicit resume metadata; they are not inferred here.

    Args:
        state_dir: Directory containing both NPZ files and ``CONVERSION.json``.
        step: Requested completed optimizer update count.

    Returns:
        The validated conversion manifest used for lazy slot-name mapping.

    Raises:
        ValueError: If the declared format, counters, hashes, or key coverage differ.
    """
    manifest = json.loads((state_dir / "CONVERSION.json").read_text(encoding="utf8"))
    if not isinstance(manifest, dict) or not _nonnegative_integer(step):
        raise ValueError("Expected a conversion manifest and a nonnegative integer step")
    for field, expected in (("format_version", 1), ("completed_steps", step), ("optimizer_global_step", step)):
        value = manifest.get(field)
        if not _nonnegative_integer(value) or value != expected:
            raise ValueError(f"Manifest {field} must equal integer {expected}")
    if manifest.get("optimizer_type") != "AdamW":
        raise ValueError("Manifest input requires a declared pure AdamW optimizer")
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(_ARCHIVES):
        raise ValueError("Manifest must provide SHA256 for model.npz and optimizer.npz")
    for name, expected in hashes.items():
        if not isinstance(expected, str) or len(expected) != 64 or _file_sha256(state_dir / name) != expected.lower():
            raise ValueError(f"Archive hash differs from conversion manifest: {name}")
    with np.load(state_dir / "model.npz", allow_pickle=False) as model:
        count = manifest.get("mapped_model_tensors")
        if not _nonnegative_integer(count) or count != len(model.files) or len(set(model.files)) != count:
            raise ValueError("Model archive key count differs from conversion manifest")
        with np.load(state_dir / "optimizer.npz", allow_pickle=False) as optimizer:
            map_manifest_slots(optimizer, manifest)
        if not {row["name"] for row in manifest["optimizer_manifest"]}.issubset(model.files):
            raise ValueError("Manifest optimizer parameter has no corresponding model tensor")
    return manifest
