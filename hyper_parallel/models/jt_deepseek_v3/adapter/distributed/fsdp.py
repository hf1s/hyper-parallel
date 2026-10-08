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
"""JT-owned FP32 router/norm units over the shared FSDP implementation."""

from dataclasses import replace
from typing import Any

import torch  # pylint: disable=forbidden-backend-import  # JT uses the Torch-only model builder.

from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager, _WrapModuleInfo


class JTFSDP2Manager(FSDP2Manager):
    """Keep JT's sensitive parameters live in FP32 without a global adapter policy.

    Only unit selection and mixed precision differ. Parameter ownership,
    sharding, replication, gradient scaling and prefetching use the base manager.
    """

    def _find_wrap_modules(self, model: torch.nn.Module,
                           metadata_by_parameter: Any = None) -> list[_WrapModuleInfo]:
        units = super()._find_wrap_modules(model, metadata_by_parameter)
        selected = {id(unit.module) for unit in units}
        # Transparent checkpoint wrappers can expose duplicate module names.
        for name, module in dict(model.named_modules()).items():
            if getattr(module, "keep_compute_in_fp32", False) and id(module) not in selected:
                units.append(_WrapModuleInfo(name, module))
                selected.add(id(module))
        return units

    def _parallelize_child_units(self, wrap_modules, owner_by_parameter, metadata_by_parameter,
                                 replicate_params, dense_fsdp_kwargs) -> int:
        fp32_kwargs = dict(dense_fsdp_kwargs)
        fp32_kwargs["mp_policy"] = replace(
            dense_fsdp_kwargs["mp_policy"], param_dtype=torch.float32,
            cast_forward_inputs=False, output_dtype=None)
        scaled = 0
        for unit in sorted(wrap_modules, key=lambda item: item.fqn.count("."), reverse=True):
            kwargs = fp32_kwargs if getattr(unit.module, "keep_compute_in_fp32", False) else dense_fsdp_kwargs
            scaled += super()._parallelize_child_units(
                [unit], owner_by_parameter, metadata_by_parameter, replicate_params, kwargs)
        return scaled
