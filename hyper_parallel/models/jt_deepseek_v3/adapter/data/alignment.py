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
"""Dataset selection for the JT/MindFormers Muon alignment runs."""

from typing import Any, Literal


from datasets import load_from_disk


AlignmentImplementation = Literal["baseline", "ref_muon"]


def load_alignment_dataset(
    *,
    implementation: AlignmentImplementation,
    unfiltered_path: str,
    filtered_path: str,
) -> Any:
    """Load the dataset paired with the selected Muon implementation.

    Args:
        implementation: ``baseline`` uses the unfiltered 4K dataset;
            ``ref_muon`` uses the row-filtered alignment dataset.
        unfiltered_path: Dataset path for the baseline run.
        filtered_path: Dataset path for the Ref-Muon run.

    Returns:
        The selected Hugging Face dataset loaded from disk.

    Raises:
        ValueError: If ``implementation`` is not a supported alignment mode.
    """
    paths = {
        "baseline": unfiltered_path,
        "ref_muon": filtered_path,
    }
    try:
        dataset_path = paths[implementation]
    except KeyError as exc:
        raise ValueError(
            "implementation must be 'baseline' or 'ref_muon', "
            f"got {implementation!r}"
        ) from exc
    return load_from_disk(dataset_path)
