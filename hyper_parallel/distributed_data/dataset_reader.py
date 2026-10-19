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
"""Dataset adapters and worker options for plan-aware sample loading."""
# This distributed-data package is intentionally PyTorch-only.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import torch  # pylint: disable=forbidden-backend-import
from torch.utils.data import Dataset  # pylint: disable=forbidden-backend-import

_INTERNAL_DATALOADER_OPTIONS = frozenset({
    "batch_sampler",
    "batch_size",
    "collate_fn",
    "dataset",
    "drop_last",
    "generator",
    "num_workers",
    "persistent_workers",
    "pin_memory",
    "prefetch_factor",
    "sampler",
    "shuffle",
})


class _IndexedPayload(NamedTuple):
    """Retain a Dataset index while allowing recursive pin-memory traversal."""

    dataset_index: int
    payload: Any


def _identity(value: Any) -> Any:
    return value


def _validate_worker_options(options: Mapping[str, Any]) -> None:
    """Validate worker settings shared by config, API overrides, and direct readers."""
    num_workers = options["num_workers"]
    if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
        raise ValueError(f"num_workers must be a non-negative integer, but got {num_workers!r}.")
    for name in ("pin_memory", "persistent_workers"):
        if not isinstance(options[name], bool):
            raise ValueError(f"{name} must be boolean, but got {options[name]!r}.")
    prefetch_factor = options["prefetch_factor"]
    if prefetch_factor is not None and (
            not isinstance(prefetch_factor, int) or isinstance(prefetch_factor, bool) or prefetch_factor < 1
    ):
        raise ValueError("prefetch_factor must be a positive integer or None.")
    if num_workers == 0 and prefetch_factor is not None:
        raise ValueError("prefetch_factor requires num_workers > 0.")
    if options["persistent_workers"] and num_workers == 0:
        raise ValueError("persistent_workers=True requires num_workers > 0.")


def _build_worker_options(
        *,
        num_workers: int,
        pin_memory: bool,
        prefetch_factor: int | None,
        persistent_workers: bool,
        generator: torch.Generator,
        dataloader_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge additional DataLoader options without exposing owned semantics."""
    try:
        options = dict({} if dataloader_kwargs is None else dataloader_kwargs)
    except Exception as exc:
        raise ValueError(f"dataloader_kwargs cannot be copied: {exc}") from exc
    conflicting_options = sorted(set(options) & _INTERNAL_DATALOADER_OPTIONS)
    if conflicting_options:
        raise ValueError(f"dataloader_kwargs contains internally managed options {conflicting_options}.")
    options.update({
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "prefetch_factor": prefetch_factor,
        "persistent_workers": persistent_workers,
        "generator": generator,
    })
    _validate_worker_options(options)
    if prefetch_factor is None:
        options.pop("prefetch_factor")
    return options


class _IndexedDataset(Dataset):
    """Attach the mapping-Dataset index used to materialize each payload."""

    def __init__(self, dataset: Any) -> None:
        """Store the user mapping Dataset."""
        self._dataset = dataset

    def __len__(self) -> int:
        """Return the wrapped Dataset length."""
        return len(self._dataset)

    def __getitem__(self, index: int) -> _IndexedPayload:
        """Materialize one sample while retaining its Dataset index."""
        return _IndexedPayload(index, self._dataset[index])
