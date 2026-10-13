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
"""Import JT's NPU entry points on CPU while rejecting unmocked kernel calls."""

import importlib.util
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import torch


if importlib.util.find_spec("torch_npu") is None:
    # Match the CPU fake-backend pattern in tests/ut/auto_models/conftest.py.
    _stub = ModuleType("torch_npu")
    _stub.__spec__ = ModuleSpec("torch_npu", loader=None)
    _stub.npu_fusion_attention = Mock(side_effect=RuntimeError("NPU attention must be mocked in CPU UT"))
    sys.modules["torch_npu"] = _stub
    torch.npu = SimpleNamespace(is_available=lambda: False)
