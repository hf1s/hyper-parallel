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
"""NPU kernels, shared weights, checkpoint replay and actual FSDP communication."""

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint
import torch_npu

from hyper_parallel import SkipDTensorDispatch, init_device_mesh
from hyper_parallel.components.functional.npu_grouped_swiglu import _get_npu_grouped_matmul
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from tests.torch.utils import init_dist


class PackedProjection(nn.Module):
    """Reuse one packed expert parameter in fused and ordinary autograd paths."""

    def __init__(self, recompute: bool) -> None:
        """Create a small packed weight whose first dimension can be FSDP sharded."""
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 32, 64, dtype=torch.float32) * 0.02)
        self.recompute = recompute

    def _project(self, inputs: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        operation = _get_npu_grouped_matmul()
        weight = self.weight
        first = operation.apply(inputs, weight.transpose(-1, -2), groups, weight)
        second = operation.apply(inputs * 0.5, weight.transpose(-1, -2), groups, weight)
        return first + second

    def forward(self, inputs: torch.Tensor, groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the packed projection and a separate ordinary weight contribution."""
        output = (checkpoint(self._project, inputs, groups, use_reentrant=False)
                  if self.recompute else self._project(inputs, groups))
        return output, self.weight.sum() * 0.125


def _run_accumulation(rank: int, recompute: bool) -> list[torch.Tensor]:
    torch.manual_seed(42)
    model = PackedProjection(recompute).npu()
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                  cast_forward_inputs=False, apply_grad_on_fp32_main_grad=True)
    fully_shard(model, mesh=init_device_mesh("npu", (dist.get_world_size(),), mesh_dim_names=("dp",)),
                mp_policy=policy)
    model.set_reduce_op_type("avg")
    gradients = []
    for iteration in range(2):
        generator = torch.Generator().manual_seed(200 + rank + iteration * 10)
        expected = torch.full((2, 32, 64), 0.375, dtype=torch.float64)
        for microstep in range(3):
            bounds = ([3, 5] if rank == 0 else [2, 2]) if microstep != 2 else [0, 0]
            inputs = torch.randn(bounds[-1], 64, generator=generator, dtype=torch.bfloat16) * 0.125
            derivative = torch.randn(bounds[-1], 32, generator=generator, dtype=torch.bfloat16) * 0.125
            for expert, (left, right) in enumerate(zip([0, *bounds[:-1]], bounds)):
                expected[expert].add_(derivative[left:right].double().T @ inputs[left:right].double() * 1.5)
            model.set_requires_gradient_sync(microstep == 2)
            model.set_is_last_backward(microstep == 2)
            with SkipDTensorDispatch():
                output, ordinary = model(inputs.npu().requires_grad_(), torch.tensor(bounds, device="npu"))
                torch.autograd.backward((output, ordinary), (derivative.npu(), torch.ones_like(ordinary)))
            for parameter in model.hsdp_scheduler.hsdp_state.hsdp_params:
                assert parameter.unsharded_param.main_grad is None, "Consumed fused storage remains on the owner"
                assert not parameter.unsharded_param.grad_added_to_main_grad, "Consumed fused flag was not cleared"
        expected = expected.float().npu()
        dist.all_reduce(expected, op=dist.ReduceOp.SUM)
        expected.div_(dist.get_world_size())
        actual = model.weight.main_grad.full_tensor()
        assert actual.dtype == torch.float32, f"Expected FP32 reduced gradient, got {actual.dtype}"
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=2e-6)
        gradients.append(actual.cpu())
        for parameter in model.hsdp_scheduler.hsdp_state.hsdp_params:
            assert parameter.unsharded_accumulated_grad is None, "Reduced accumulation storage was not released"
        model.weight.main_grad = None
        model.zero_grad()
    return gradients


def test_fused_expert_gradients() -> None:
    """Check CPU FP64 oracle parity across replay, empty experts and delayed sync."""
    rank, _ = init_dist()
    torch_npu.npu.matmul.allow_hf32 = False
    ordinary = _run_accumulation(rank, recompute=False)
    replayed = _run_accumulation(rank, recompute=True)
    for reference, actual in zip(ordinary, replayed):
        torch.testing.assert_close(reference, actual, rtol=0, atol=0)
