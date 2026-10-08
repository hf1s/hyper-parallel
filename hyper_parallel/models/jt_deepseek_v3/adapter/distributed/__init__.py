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
"""DeepSeek-V3 expert-parallel compute archetype."""

from hyper_parallel.models.jt_deepseek_v3.adapter.distributed.ep_compute import (
    jt_deepseek_v3_ep_compute_fn,
)

__all__ = ["jt_deepseek_v3_ep_compute_fn"]
