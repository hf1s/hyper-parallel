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
from typing import Any

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
) -> nn.Module:
    """Replace DeepSeek-V3.2 MLA/indexer attention with the fused DSA module.

    The indexer KL loss distils the attention distribution summed over every
    head. Tensor parallelism leaves each rank a subset of the heads and the
    fused KL operator has no cross-rank reduction, so the trainable-indexer
    loss is rejected together with ``tp_size > 1``.
    """
    config = module.config
    indexer_loss = getattr(config, "dsa_loss_coeff", 0.0) and not getattr(
        config, "freeze_DSA", getattr(config, "freeze_dsa", False)
    )
    if indexer_loss and context.get("tp", False):
        raise ValueError(
            f"{module_fqn}: the DeepSeek-V3.2 indexer KL loss (dsa_loss_coeff="
            f"{config.dsa_loss_coeff}) needs every attention head on one rank; "
            "train it with tp_size=1 or set dsa_loss_coeff=0 / freeze_dsa=true"
        )
    from hyper_parallel.components.modules import DeepseekV32DSAAttention  # pylint: disable=C0415

    return DeepseekV32DSAAttention(module=module, module_fqn=module_fqn, context=context)


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
]
