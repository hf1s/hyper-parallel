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
"""Packed-document runtime inputs for DeepSeek-V3.2 DSA attention."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.batching.runtime_input import (
    RuntimeInputAdapter,
    RuntimeInputContext,
)


class DeepseekV32PackedSequence(RuntimeInputAdapter):
    """Pass packed-document boundaries to the DSA attention.

    DSA derives its causal structure from cumulative sequence lengths and
    never reads a dense attention mask. When documents packed into one row
    must not attend to each other, this adapter forwards the global
    leading-zero ``cu_seq_lens`` of the ``[batch, sequence]`` token stream as
    ``cu_seq_lens_q``/``cu_seq_lens_k``. The non-CP attention consumes them
    directly and the CP wrapper projects them onto its local query shard and
    gathered KV prefix. Rows without this adapter are independent causal
    sequences.
    """

    def runtime_input_fields(self) -> tuple[str, ...]:
        """Declare the packed-boundary fields owned by this adapter."""
        return ("cu_seq_lens_q", "cu_seq_lens_k")

    def build_runtime_inputs(
            self,
            *,
            batch: Mapping[str, Any],
            context: RuntimeInputContext,
    ) -> Mapping[str, Any]:
        """Return the global packed boundaries for both query and key/value."""
        del context
        cu_seq_lens = batch.get("cu_seq_lens")
        if cu_seq_lens is None:
            raise ValueError("DeepSeek-V3.2 packed attention requires cu_seq_lens in the batch")
        boundaries = cu_seq_lens.to(dtype=torch.int32)
        return {"cu_seq_lens_q": boundaries, "cu_seq_lens_k": boundaries}


__all__ = ["DeepseekV32PackedSequence"]
