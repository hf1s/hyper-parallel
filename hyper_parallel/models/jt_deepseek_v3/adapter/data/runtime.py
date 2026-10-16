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
"""Translate shared packed metadata to the JT forward contract."""

from collections.abc import Mapping
from typing import Any

from hyper_parallel.data.batching.runtime_input import RuntimeInputAdapter, RuntimeInputContext


class JTSequenceRuntime(RuntimeInputAdapter):
    """Pass the public batch's document boundaries to JT without constructing a dense mask."""

    def runtime_input_fields(self) -> tuple[str, ...]:
        """Declare the model-owned cumulative-length input."""
        return ("actual_seq_len",)

    def build_runtime_inputs(self, *, batch: Mapping[str, Any],
                             context: RuntimeInputContext) -> Mapping[str, Any]:
        """Convert global leading-zero boundaries once at the data/model boundary.

        Args:
            batch: Local tokens and global cumulative document boundaries.
            context: Local sequence shape and CP degree from the public data path.
        """
        boundaries = batch.get("cu_seq_lens")
        if boundaries is None:
            return {}
        values = tuple(int(value) for value in boundaries.tolist())
        global_length = context.local_input_shape[1] * context.parallel_sizes["cp"]
        if (context.local_input_shape[0] != 1 or len(values) < 2 or values[0] != 0
                or values[-1] != global_length
                or any(left >= right for left, right in zip(values, values[1:]))):
            raise ValueError("JT packed boundaries must cover one global batch-one sequence")
        return {"actual_seq_len": values[1:]}
