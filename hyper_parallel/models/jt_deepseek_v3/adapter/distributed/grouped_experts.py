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
"""Canonical-parameter routed expert replacement with grouped forward."""

from typing import Any

import torch
from torch import nn
from hyper_parallel.components.checkpoint.conversion_ops import Split
from hyper_parallel.components.checkpoint.weight_conversion import WeightConverter
from hyper_parallel.components.functional.npu_grouped_swiglu import npu_grouped_swiglu
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3Experts
from hyper_parallel.models.replacement import module_replacement


@module_replacement
class JTDeepseekV3FusedExperts(nn.Module):
    """Pack canonical gate/up parameters only for grouped expert execution."""

    def __init__(self, *, module: JTDeepseekV3Experts, module_fqn: str = "", context: Any = None) -> None:
        """Transfer canonical expert parameters and expose grouped execution."""
        super().__init__()
        del module_fqn, context
        for name, parameter in module._parameters.items():  # pylint: disable=protected-access
            self.register_parameter(name, parameter)
        for name, buffer in module._buffers.items():  # pylint: disable=protected-access
            self.register_buffer(
                name,
                buffer,
                persistent=name not in module._non_persistent_buffers_set,  # pylint: disable=protected-access
            )
        self.num_experts = module.num_experts
        self.hidden_dim = module.hidden_dim
        self.intermediate_dim = module.intermediate_dim
        self.act_fn = module.act_fn
        self.train(module.training)

    def forward_expert_major(self, inputs: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Run grouped SwiGLU after a forward-only gate/up packing step."""
        gate_up_proj = torch.cat((self.gate_proj, self.up_proj), dim=1)
        return npu_grouped_swiglu(
            inputs.to(torch.bfloat16),
            gate_up_proj.to(torch.bfloat16),
            self.down_proj.to(torch.bfloat16),
            counts,
        )

    def make_transforms(self) -> list[WeightConverter]:
        """Split legacy fused expert checkpoints into canonical parameters."""
        return [
            WeightConverter(
                source_patterns="gate_up_proj",
                target_patterns=["gate_proj", "up_proj"],
                operations=[
                    Split((self.intermediate_dim, self.intermediate_dim), dim=1)
                ],
            )
        ]


__all__ = ["JTDeepseekV3FusedExperts"]
