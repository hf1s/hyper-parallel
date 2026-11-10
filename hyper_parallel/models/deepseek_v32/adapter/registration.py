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
"""Architecture and lazy provider registration for DeepSeek-V3.2."""

from typing import Any

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.deepseek_v32.adapter.materialization import register_rotary_materialization
from hyper_parallel.models.deepseek_v32.adapter.policies.sharding import (
    build_parameter_sharding_rules,
)
from hyper_parallel.models.registry import register_model_adapter


def _load_replacements():
    """Return the family's replacement factories without importing NPU ops."""
    from hyper_parallel.models.deepseek_v32.adapter.conversion import (  # pylint: disable=C0415
        module_replacement,
    )

    return module_replacement


def _load_context_parallel():
    """Return the DeepSeek-V3.2 DSA CP wrapper provider lazily."""
    from hyper_parallel.models.deepseek_v32.adapter.distributed import (  # pylint: disable=C0415
        context_parallel,
    )

    return context_parallel


def bind_training_parallel_groups(model: Any, context: Any) -> None:
    """Bind the router groups once the model is sharded; reject the indexer loss under TP.

    The framework calls the external-state provider after planning with the run's
    ``MeshContext``; DeepSeek-V3.2 owns no external state, so this returns None.
    The DSA indexer KL loss compares against the attention of all heads, which
    tensor parallelism splits over ranks, so that combination is rejected.

    Args:
        model: Sharded model before FSDP.
        context: ``ExternalBuildContext`` of the build.
    """
    from hyper_parallel.models.deepseek_v32.adapter.router import (  # pylint: disable=C0415
        bind_router_parallel_groups,
    )

    mesh = context.mesh_context
    if mesh is not None and mesh.tp_size > 1 and any(
            getattr(module, "dsa_loss_coeff", 0.0) for module in model.modules()):
        raise NotImplementedError(
            "The DSA indexer KL loss needs the attention of all heads; "
            "set dsa_loss_coeff to 0 or tp_size to 1"
        )
    bind_router_parallel_groups(model, mesh)


DEEPSEEK_V32_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="DeepseekV32ForCausalLM",
    model_type="deepseek_v32",
    replacements=_load_replacements,
    materialization=register_rotary_materialization,
    context_parallel=_load_context_parallel,
    external_state=bind_training_parallel_groups,
    sharding_rules=build_parameter_sharding_rules,
)
register_model_adapter(DEEPSEEK_V32_ADAPTER_SPEC)
