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

"""Native packed SwiGLU must reject unsafe TP plans before apply."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hyper_parallel.components.modules.swiglu_mlp import SwiGLUMLP
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.distributed import ShardingPlanner
from hyper_parallel.distributed.recipe_spec import CP, EP, TP, ModuleShardingSpec
from tests.common.mark_utils import arg_mark
from tests.ut.dual_mode_dtensor.conftest import TinyConfig, TinyLlamaMLP


class SharedModel(nn.Module):
    """Expose the same native module through a shared-expert boundary."""

    def __init__(self, source: nn.Module) -> None:
        """Pack the supplied unfused MLP as a shared expert."""
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.shared_experts = SwiGLUMLP(module=source)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the shared expert."""
        return self.mlp.shared_experts(hidden_states)


def replicated_plan(*, contracts: bool) -> dict[str, ModuleShardingSpec]:
    """Override both projections, optionally including their compute contracts."""
    params = {TP: Replicate(), CP: Replicate(), EP: Replicate()}
    activation = {TP: Replicate(), CP: Shard(1), EP: Replicate()}
    result = {}
    for name in ("linear_fc1", "linear_fc2"):
        fields = {}
        if contracts:
            fields = {"in_src": {"hidden_states": copy.deepcopy(activation)},
                      "in_dst": {"hidden_states": copy.deepcopy(activation)},
                      "out_src": {"output": copy.deepcopy(activation)},
                      "out_dst": {"output": copy.deepcopy(activation)}}
        result[f"mlp.shared_experts.{name}"] = ModuleShardingSpec(params={"weight": copy.deepcopy(params)}, **fields)
    return result


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestSwiGLUTPContract:
    """Plan-time behavior of native packed and fully replicated SwiGLU."""

    @staticmethod
    def mesh(size: int = 2) -> SimpleNamespace:
        """Supply planner-only metadata without a distributed process group."""
        return SimpleNamespace(mesh_dim_names=("tp",), mesh_shape=(size,))

    def test_default_shared_plan_rejects_unpaired_gate_up(self) -> None:
        """Default shared-expert colwise plans cannot split contiguous gate/up."""
        model = SharedModel(TinyLlamaMLP(TinyConfig()))
        with pytest.raises(ValueError, match="matched gate/up sections"):
            ShardingPlanner().plan(model, self.mesh(), tp_size=2, sequence_parallel=False)

    def test_parameter_only_override_rejects_duplicate_sum(self) -> None:
        """Replicated weights with a retained Partial output would duplicate work."""
        model = SharedModel(TinyLlamaMLP(TinyConfig()))
        with pytest.raises(ValueError, match="complete output.*Partial"):
            ShardingPlanner(plan_overrides=replicated_plan(contracts=False)).plan(model, self.mesh(), tp_size=2)

    def test_complete_replication_is_accepted(self) -> None:
        """A complete override remains usable with the original module semantics."""
        source = TinyLlamaMLP(TinyConfig())
        reference = copy.deepcopy(source)
        model = SharedModel(source)
        ShardingPlanner(plan_overrides=replicated_plan(contracts=True)).plan(model, self.mesh(), tp_size=2)
        hidden = torch.randn(2, 8, 16)
        torch.testing.assert_close(model(hidden), reference(hidden))

    def test_single_rank_keeps_existing_behavior(self) -> None:
        """An inactive TP axis does not impose a new distributed constraint."""
        model = SharedModel(TinyLlamaMLP(TinyConfig()))
        ShardingPlanner().plan(model, self.mesh(1), tp_size=1)

    def test_input_axis_is_not_a_packed_output_axis(self) -> None:
        """Planner validation distinguishes an input-axis shard from packed output."""
        model = SharedModel(TinyLlamaMLP(TinyConfig()))
        overrides = replicated_plan(contracts=True)
        first = overrides["mlp.shared_experts.linear_fc1"]
        first.params["weight"][TP] = Shard(1)
        first.in_dst["hidden_states"][TP] = Shard(-1)
        first.out_src["output"][TP] = Partial()
        ShardingPlanner(plan_overrides=overrides).plan(model, self.mesh(), tp_size=2)
