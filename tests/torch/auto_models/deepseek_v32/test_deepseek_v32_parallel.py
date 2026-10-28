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
"""Framework-free launchers for DeepSeek-V3.2 TP/SP/CP/EP parity on CPU/Gloo."""

from pathlib import Path
from unittest.mock import patch

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_WORKER = str(Path(__file__).with_name("_test_deepseek_v32_parallel.py"))


@pytest.mark.parametrize(
    ("case_name", "num_proc"),
    [
        ("test_tp2_sp_ep2_transformers_attention", 2),
        ("test_cp2_dsa", 2),
        ("test_tp2_ep2_dsa_loss_parallel", 2),
        ("test_tp2_cp2_ep4_dsa", 4),
    ],
)
@arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_deepseek_v32_parallel_parity_gloo(case_name: str, num_proc: int) -> None:
    """
    Feature: DeepSeek-V3.2 parallel integration.
    Description: Run TP/SP/loss-parallel, DSA CP and EP topologies on CPU ranks through the Trainer chain.
    Expectation: The loss and every locally exact gradient match the single-process model.
    """
    # CPU workers need no accelerator backend, including torch's entry-point loading.
    with patch.dict("os.environ", {"TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}):
        torchrun_case(_WORKER, case_name, num_proc=num_proc)
