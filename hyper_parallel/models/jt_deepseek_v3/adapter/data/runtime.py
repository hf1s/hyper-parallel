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
"""Pass prepared document boundaries through the existing text batch interface."""

from collections.abc import Mapping
from typing import Any

from hyper_parallel.data.batching.runtime_input import RuntimeInputAdapter, RuntimeInputContext


class JTPackedRuntime(RuntimeInputAdapter):
    """Forward the global document boundaries after TP batch transport."""

    def runtime_input_fields(self) -> tuple[str, ...]:
        """Declare the additional model input."""
        return ("cu_seq_lens",)

    def build_runtime_inputs(self, *, batch: Mapping[str, Any], context: RuntimeInputContext) -> dict[str, Any]:
        """Convert cumulative lengths to the sequence metadata accepted by JT.

        Args:
            batch: Parallel batch containing validated global boundaries.
            context: Runtime topology; JT requires complete sequences on every CP rank.
        """
        if context.parallel_sizes["cp"] != 1:
            raise ValueError("JT packed input does not support context parallelism")
        if context.local_input_shape[0] != 1:
            raise ValueError("JT packed input requires one packed record per micro-batch")
        return {"cu_seq_lens": tuple(batch["cu_seq_lens"].tolist())}
