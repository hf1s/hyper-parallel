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
"""JT-owned builder for pre-tokenized supervised indexed records."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyper_parallel.models.jt_deepseek_v3.adapter.data.dataset import IndexedSupervisedDataset


def build_jt_supervised_dataset(
        *,
        data_path: str | Path,
        sequence_length: int | None = None,
        data_config: Mapping[str, Any] | None = None,
        **_: Any,
) -> IndexedSupervisedDataset:
    """Build JT's aligned supervised records and retain DataLoader options.

    The generic indexed supervised reader intentionally only owns record I/O.
    JT keeps its ``data_config`` at the model boundary so the shared Trainer
    can select native batch-sampler balancing without teaching the framework
    about this model family.

    Args:
        data_path: Prefix before the ``.tokens``, ``.labels`` and
            ``.loss_mask`` indexed files.
        sequence_length: Required length for every complete record.
        data_config: Dataset options consumed by the shared DataLoader.

    Returns:
        An aligned supervised Dataset carrying the resolved JT data options.
    """
    dataset = IndexedSupervisedDataset(data_path, sequence_length)
    dataset.data_config = dict(data_config or {})
    return dataset
