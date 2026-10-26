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
"""Register the Xing4.0 model and its adapter.

Expert parallelism and the optimizer are selected by the recipe through
``plan_overrides`` and the ``optimizer`` target.
"""

from typing import Any

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_custom_model, register_model_adapter


def build_parameter_sharding_rules() -> list[tuple[list[str], Any]]:
    """Return the attention parameter roles that generic naming rules do not infer.

    The mHC, stream-collapse head and MTP fusion parameters are declared replicated by the
    recipe's ``plan_overrides``.

    Returns:
        ``(name patterns, role)`` pairs checked before the generic naming rules.
    """
    # Lazy: discovery imports this module without the distributed runtime.
    from hyper_parallel.distributed.tensor_parallel.param_role import (  # pylint: disable=import-outside-toplevel
        ParamRole,
    )

    return [
        (["linear_qkv"], ParamRole.REPLICATED),
        (["q_b_proj", "kv_b_proj"], ParamRole.COLWISE),
    ]


register_custom_model(
    "Xing40ForCausalLM",
    "hyper_parallel.models.xing40.modeling_xing40",
    "Xing40ForCausalLM",
)

XING40_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Xing40ForCausalLM",
    model_type="xing40",
    sharding_rules=build_parameter_sharding_rules,
)
register_model_adapter(XING40_ADAPTER_SPEC)

__all__ = ["XING40_ADAPTER_SPEC", "build_parameter_sharding_rules"]
