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
"""DeepSeek-V3.2 replacements backed by shared high-performance components."""

from collections.abc import Mapping
from typing import Any, Optional

from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.models.replacement import module_replacement


@module_replacement
def replace_deepseek_v32_rms_norm(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
) -> nn.Module:
    """Replace DeepSeek RMSNorm with the shared fused RMSNorm module."""
    from hyper_parallel.components.modules import RMSNorm  # pylint: disable=C0415

    return RMSNorm(module=module, module_fqn=module_fqn, context=context)


@module_replacement
def replace_deepseek_v32_dsa_attention(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
    dsa_loss_coeff: Optional[float] = None,
) -> nn.Module:
    """Replace DeepSeek-V3.2 MLA/indexer attention with the fused DSA module.

    ``dsa_loss_coeff`` sets the weight of the indexer KL loss; ``None`` keeps
    the configuration's value, which the released configuration does not set.
    """
    from hyper_parallel.components.modules import DeepseekV32DSAAttention  # pylint: disable=C0415

    attention = DeepseekV32DSAAttention(module=module, module_fqn=module_fqn, context=context)
    if dsa_loss_coeff is not None:
        attention.dsa_loss_coeff = float(dsa_loss_coeff)
    return attention


@module_replacement
def replace_deepseek_v32_router(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
    aux_loss_coeff: float = 0.0,
    enable_expert_bias: bool = False,
    expert_bias_update_rate: float = 1e-3,
) -> nn.Module:
    """Replace the MoE router with the training router, sharing its weight and bias.

    Args:
        module: Transformers ``DeepseekV32TopkRouter``.
        module_fqn: Unused.
        context: Unused.
        aux_loss_coeff: Weight of the sequence-wise balance loss; 0 disables it.
        enable_expert_bias: Whether the selection bias follows the routed load.
        expert_bias_update_rate: Bias step per optimizer step.
    """
    del module_fqn, context
    from hyper_parallel.models.deepseek_v32.adapter.router import DeepseekV32Router  # pylint: disable=C0415

    return DeepseekV32Router.from_router(
        module, aux_loss_coeff=aux_loss_coeff, enable_expert_bias=enable_expert_bias,
        expert_bias_update_rate=expert_bias_update_rate,
    )


@module_replacement
def replace_deepseek_v32_grouped_experts(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
) -> nn.Module:
    """Replace batched DeepSeek experts with the shared grouped-GEMM module."""
    required = ("gate_up_proj", "down_proj")
    missing = [name for name in required if not isinstance(getattr(module, name, None), nn.Parameter)]
    if missing:
        raise TypeError(
            f"{module_fqn}: DeepSeek-V3.2 experts require batched parameters {missing}"
        )
    from hyper_parallel.components.modules import GroupedExperts  # pylint: disable=C0415

    return GroupedExperts(module=module, module_fqn=module_fqn, context=context)


__all__ = [
    "replace_deepseek_v32_dsa_attention",
    "replace_deepseek_v32_grouped_experts",
    "replace_deepseek_v32_rms_norm",
    "replace_deepseek_v32_router",
]
