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
"""JT DeepSeek expert-parallel execution provider."""

from functools import partial
from typing import Any, Callable

from hyper_parallel.distributed.expert_parallel.recipes import build_ep_compute
from hyper_parallel.distributed.recipe_spec import local_compute


@local_compute
def build_jt_ep(*, module: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any, ep_mesh: Any) -> Callable:
    """Bind public EP execution using the JT model routing contract."""
    del mesh, tp_mesh, cp_mesh
    if ep_mesh is None:
        raise ValueError("JT requires an EP mesh")
    module.ep_group = ep_mesh.get_group("ep")
    module.ep_world = ep_mesh["ep"].size()
    executor = build_ep_compute(
        module,
        ep_mesh,
        router_fn=type(module).route,
        archetype_key="jt_deepseek_v3_hf",
        expected_attrs=["gate", "experts", "shared_experts", "config"],
        combine=module.combine_routed,
        use_grouped_gemm=True,
    )
    module.ep_compute = partial(executor, module)
    return type(module).forward


__all__ = ["build_jt_ep"]
