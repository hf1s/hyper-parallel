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
"""Architecture and lazy provider registration for DeepSeek-V3.2.

``DeepseekV32ForCausalLM`` checkpoints build the training model
:class:`~hyper_parallel.models.deepseek_v32.modeling_deepseek_v32.DeepseekV32ForCausalLM`
unless a recipe forces the Transformers class.
"""

from typing import Any

from hyper_parallel.models.adapter_spec import ModelAdapterSpec, RecomputePolicy
from hyper_parallel.models.deepseek_v32.adapter.materialization import register_rotary_materialization
from hyper_parallel.models.deepseek_v32.adapter.policies.sharding import (
    build_parameter_sharding_rules,
)
from hyper_parallel.models.registry import register_custom_model, register_model_adapter


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


def build_recompute_policy() -> RecomputePolicy:
    """Expose trunk decoder layers and MTP depths as recompute regions."""
    return RecomputePolicy(region_patterns=("model.layers.*", "mtp.layers.*"))


def get_fsdp_wrap_modules(model: Any) -> tuple[str, ...]:
    """Make each MTP depth its own FSDP unit."""
    mtp = getattr(model, "mtp", None)
    return () if mtp is None else tuple(f"mtp.layers.{name}" for name, _ in mtp.layers.named_children())


def get_fsdp_excluded_subtrees(model: Any) -> tuple[str, ...]:
    """Keep decoder discovery out of the MTP branch, which declares its own units."""
    return () if getattr(model, "mtp", None) is None else ("mtp",)


register_custom_model(
    "DeepseekV32ForCausalLM",
    "hyper_parallel.models.deepseek_v32.modeling_deepseek_v32",
    "DeepseekV32ForCausalLM",
)

DEEPSEEK_V32_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="DeepseekV32ForCausalLM",
    model_type="deepseek_v32",
    replacements=_load_replacements,
    materialization=register_rotary_materialization,
    context_parallel=_load_context_parallel,
    sharding_rules=build_parameter_sharding_rules,
    fsdp_wrap_modules=get_fsdp_wrap_modules,
    fsdp_excluded_subtrees=get_fsdp_excluded_subtrees,
    recompute=build_recompute_policy,
)
register_model_adapter(DEEPSEEK_V32_ADAPTER_SPEC)
