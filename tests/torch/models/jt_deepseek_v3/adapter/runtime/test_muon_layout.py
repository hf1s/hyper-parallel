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
"""Launch the JT Muon callback workers without importing a framework."""

from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import TorchCase, parallel_run

WORKER = str(Path(__file__).with_name("_test_muon_layout.py"))


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_jt_muon_layout_single_card():
    """
    Feature: JT Muon logical matrix layout on one NPU.
    Description: Run the production builder and complete-versus-sharded worker.
    Expectation: The worker passes all parameter and state comparisons.
    """
    parallel_run([TorchCase(WORKER, "test_jt_muon_layout_single_card", None, 1)])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_jt_muon_layout_fsdp_four_cards():
    """
    Feature: JT Muon logical matrix layout across four NPU shards.
    Description: Gather and restore unequal matrix blocks over a four-rank FSDP mesh.
    Expectation: The worker passes all parameter and state comparisons.
    """
    parallel_run([TorchCase(WORKER, "test_jt_muon_layout_fsdp_four_cards", None, 4)])
