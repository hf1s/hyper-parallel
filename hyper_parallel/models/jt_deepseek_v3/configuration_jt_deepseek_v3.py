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
"""Hugging Face configuration for the independently registered JT model family."""

from transformers import AutoConfig, DeepseekV32Config


class JTDeepseekV3Config(DeepseekV32Config):
    """DeepSeek-V3.2-compatible configuration with an independent JT identity."""

    model_type = "jt_deepseek_v3"


AutoConfig.register(JTDeepseekV3Config.model_type, JTDeepseekV3Config, exist_ok=True)

__all__ = ["JTDeepseekV3Config"]
