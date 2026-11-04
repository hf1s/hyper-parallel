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
"""Replacement: build the shared NPU GroupedExperts from Xing4.0's per-expert nn.ModuleList."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.checkpoint.weight_conversion import Transpose, WeightConverter
from hyper_parallel.components.modules.grouped_experts import GroupedExperts
from hyper_parallel.models.replacement import module_replacement as register_module_replacement

__all__ = ["Xing40GroupedExperts"]


class _PackedExpertSource(nn.Module):
    """Synthetic already-packed-shape stand-in for a ModuleList of per-expert MLPs.

    ``GroupedExperts.__init__`` requires its source module to already expose
    ``gate_up_proj``/``down_proj`` as packed ``nn.Parameter`` attributes (``hasattr``-checked by
    ``_source_parameters``). The real ``Xing4_0MoE.experts`` is an ``nn.ModuleList`` of separate
    ``Xing4_0MLP`` modules instead -- one checkpoint key per expert per projection, confirmed
    against the real checkpoint's ``model.safetensors.index.json``
    (``experts.{i}.{gate,up,down}_proj.weight``), not one packed key.

    This synthesizes the packed *shape* ``GroupedExperts.__init__`` expects, so its own
    transpose/dimension-inference logic (``_source_layout``, ``_source_dimensions``) runs
    unmodified. Real values under meta-device construction come later, from
    :meth:`Xing40GroupedExperts.make_transforms`'s declared per-expert-key mapping -- never from
    this wrapper. Real values under eager (non-meta) construction are filled in here too, so
    CPU/eager testing also produces a correctly-packed result.
    """

    def __init__(self, experts: nn.ModuleList, config: Any) -> None:
        """Build the packed-shape stand-in from the real per-expert ModuleList.

        Args:
            experts: ``nn.ModuleList`` of ``Xing4_0MLP`` instances (or any module exposing the
                same ``gate_proj``/``up_proj``/``down_proj`` ``nn.Linear`` attributes).
            config: Model configuration, forwarded to ``GroupedExperts.__init__`` unchanged.
        """
        super().__init__()
        self.config = config
        num_experts = len(experts)
        self.num_experts = num_experts
        self.has_gate = True
        self.has_bias = False
        first = experts[0]
        hidden_size = first.gate_proj.in_features
        intermediate_size = first.gate_proj.out_features
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        weight = first.gate_proj.weight
        gate_up = torch.empty(num_experts, 2 * intermediate_size, hidden_size,
                              device=weight.device, dtype=weight.dtype)
        down = torch.empty(num_experts, hidden_size, intermediate_size,
                           device=weight.device, dtype=weight.dtype)
        if not weight.is_meta:
            with torch.no_grad():
                for index, expert in enumerate(experts):
                    gate_up[index].copy_(
                        torch.cat((expert.gate_proj.weight, expert.up_proj.weight), dim=0))
                    down[index].copy_(expert.down_proj.weight)
        self.gate_up_proj = nn.Parameter(gate_up, requires_grad=weight.requires_grad)
        self.down_proj = nn.Parameter(down, requires_grad=first.down_proj.weight.requires_grad)


@register_module_replacement
class Xing40GroupedExperts(GroupedExperts):
    """Replacement of ``Xing4_0MoE.experts``: the shared NPU ``GroupedExperts``.

    Every other model this repo trains (DeepSeek-V3, Qwen3-MoE) already has its experts packed
    into one ``gate_up_proj``/``down_proj`` tensor at the HF/checkpoint level, so ``GroupedExperts``
    has never needed to bridge a per-expert, separately-keyed source before -- confirmed by its
    own ``make_transforms`` only handling transpose and bias renaming, nothing that introduces a
    new dimension. ``_PackedExpertSource`` does the construction-time shape bridging;
    :meth:`make_transforms` declares the actual checkpoint-loading-time per-expert-key mapping,
    with the same Transformers ``MergeModulelist``/``Concatenate`` operations the DeepSeek-V4.1
    conversion in ``deepseek_v41.adapter.conversion`` uses for the identical job.
    """

    def __init__(self, *, module: nn.ModuleList, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None) -> None:
        """Build the replacement from Xing4.0's per-expert ``nn.ModuleList``.

        Args:
            module: ``nn.ModuleList`` of ``Xing4_0MLP`` instances being replaced.
            module_fqn: Fully qualified name of ``module``.
            context: Replacement context of the build pipeline.

        Raises:
            ValueError: If ``module`` is empty.
        """
        if len(module) == 0:
            raise ValueError("Xing40GroupedExperts requires a nonempty experts ModuleList")
        config = getattr(module[0], "config", None)
        source = _PackedExpertSource(module, config)
        super().__init__(module=source, module_fqn=module_fqn, context=context)

    def make_transforms(self) -> list[WeightConverter]:
        """Describe the real per-expert-key to packed-NPU-layout checkpoint conversion.

        ``MergeModulelist(dim=0)`` stacks the tensors that *one* source pattern collected, so the
        patterns are one glob per projection, not one pattern per expert index. The patterns are
        relative to the replaced module's FQN (``*.mlp.experts``), which the framework assigns as
        each converter's ``scope_prefix``, so the glob matches the bare expert index that is left
        of a released key once that prefix is stripped.

        Returns:
            One converter per packed parameter, reading the released per-expert checkpoint keys.

        Raises:
            ImportError: If Transformers does not expose the conversion operations.
        """
        try:
            # Lazy, so that importing this module does not itself require Transformers.
            from transformers.core_model_loading import (  # pylint: disable=C0415
                Concatenate,
                MergeModulelist,
            )
        except ImportError as error:
            # Not an empty list: ``_install_replacement_target`` reads one as "this replacement
            # needs no checkpoint transform" and then validates the replacement against the
            # source ModuleList's own tensor names, failing on that instead of on the real cause.
            # Raise the real cause here; make_transforms() only runs when a replacement is
            # installed.
            raise ImportError(
                "Xing40GroupedExperts needs transformers.core_model_loading.MergeModulelist to "
                "pack the released per-expert Xing4.0 expert weights"
            ) from error

        gate_up_ops: list[Any] = [MergeModulelist(dim=0), Concatenate(dim=1)]
        down_ops: list[Any] = [MergeModulelist(dim=0)]
        if self._transpose_source_weights:
            gate_up_ops.append(Transpose(dim0=-2, dim1=-1))
            down_ops.append(Transpose(dim0=-2, dim1=-1))
        return [
            WeightConverter(
                source_patterns=["*.gate_proj.weight", "*.up_proj.weight"],
                target_patterns=self._source_fc1_name,
                operations=gate_up_ops,
            ),
            WeightConverter(
                source_patterns="*.down_proj.weight",
                target_patterns="down_proj",
                operations=down_ops,
            ),
        ]
