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
"""Exercise the real grouped-GEMM autograd path using CPU kernel oracles."""
# pylint: disable=protected-access

import importlib
import sys
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import torch
from torch.distributed.tensor import DTensor as TorchDTensor
from torch.utils.checkpoint import DefaultDeviceType, checkpoint

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.fully_shard.hsdp_param import AllGatherCommCtx, HSDPParamV2, ReduceScatterCommCtx
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerContext
from hyper_parallel.core.fully_shard.hsdp_state import HSDPStateV2
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy


grouped_swiglu = importlib.import_module("hyper_parallel.components.functional.npu_grouped_swiglu")


def grouped_mm(*, x: list[torch.Tensor], weight: list[torch.Tensor], group_list: torch.Tensor,
               group_type: int, **_kwargs: Any) -> list[torch.Tensor]:
    """Model BF16 kernel output rounding separately from FP32 accumulation."""
    bounds = [0, *group_list.tolist()]
    if group_type == 2:
        return [torch.stack([x[0][:, left:right].float() @ weight[0][left:right].float()
                             for left, right in zip(bounds, bounds[1:])]).to(x[0].dtype)]
    return [torch.cat([(x[0][left:right].float() @ weight[0][index].float()).to(x[0].dtype)
                       for index, (left, right) in enumerate(zip(bounds, bounds[1:]))])]


def grouped_add(destination: torch.Tensor, inputs: torch.Tensor, weights: torch.Tensor,
                groups: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
    """The fused kernel adds the FP32 product before any BF16 output rounding."""
    bounds = [0, *groups.tolist()]
    destination = destination.view(len(bounds) - 1, inputs.shape[-1], weights.shape[-1])
    for index, (left, right) in enumerate(zip(bounds, bounds[1:])):
        destination[index].add_(inputs[left:right].float().T @ weights[left:right].float())
    return destination


def make_expert_state(weights: torch.Tensor) -> HSDPParamV2:
    """Use the actual stable FSDP parameter and its accumulation methods."""
    state = object.__new__(HSDPParamV2)
    state.sharded_param = torch.nn.Parameter(weights.clone())
    state._orig_size = weights.shape
    state._contiguous_orig_stride = weights.stride()
    state.unsharded_param_buffers = [weights.flatten().clone()]
    state.allgather_comm_ctx = AllGatherCommCtx()
    state.source_shard_info = None
    state.unsharded_accumulated_grad = None
    state.reduce_dtype = torch.float32
    state.mp_policy = MixedPrecisionPolicy(apply_grad_on_fp32_main_grad=True)
    state.init_unsharded_param()
    return state


def make_backward_state(parameter: HSDPParamV2) -> HSDPStateV2:
    """Use the real no-sync and world-size-one reduction scheduling path."""
    state = object.__new__(HSDPStateV2)
    state.modules = ()
    state.hsdp_params = [parameter]
    state.comm_fusion_policy = SimpleNamespace(enable_comm_fusion=False)
    state.reshard_after_backward = False
    state.scheduler_ctx = HSDPSchedulerContext()
    state.requires_all_reduce = False
    state.reduce_op_type = torch.distributed.ReduceOp.AVG
    parameter.hsdp_placement = Shard(0)
    parameter.shard_world_size = 1
    parameter.gradient_scaling_factor = None
    parameter.mesh_info = None
    parameter.reduce_scatter_comm_ctx = ReduceScatterCommCtx()
    return state


class TestGroupedSwiGLUGradients(unittest.TestCase):
    """Preserve forward/dgrad while retaining FP32 expert weight gradients."""

    def setUp(self) -> None:
        """Replace only NPU kernels, keeping the production autograd wrapper."""
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")
        backend = SimpleNamespace(npu_grouped_matmul=grouped_mm, npu_grouped_matmul_add_=grouped_add)
        self.enterContext(patch.dict(sys.modules, {"torch_npu": backend}))
        self.enterContext(patch.object(grouped_swiglu, "_NPU_GROUPED_MATMUL", None))
        torch.manual_seed(12)

    def test_accumulation_matches_fp32_oracle_and_releases_buffers(self) -> None:
        """Multiple microbatches, an empty expert and an empty batch retain each contribution once."""
        state = make_expert_state(torch.randn(2, 4, 3, dtype=torch.bfloat16))
        expected = torch.zeros(2, 4, 3, dtype=torch.float32)
        operation = grouped_swiglu._get_npu_grouped_matmul()
        for boundaries in ([3, 5], [2, 2], [0, 0]):
            with self.subTest(boundaries=boundaries):
                tokens = boundaries[-1]
                inputs = torch.randn(tokens, 3, dtype=torch.bfloat16, requires_grad=True)
                grad_output = torch.randn(tokens, 4, dtype=torch.bfloat16)
                groups = torch.tensor(boundaries, dtype=torch.int64)
                weight = state.unsharded_param
                result = operation.apply(inputs, weight.transpose(-1, -2), groups, weight)
                result.backward(grad_output)
                torch.testing.assert_close(weight.grad, torch.zeros_like(weight), rtol=0, atol=0)
                state.accumulate_unsharded_grad_if_needed()
                start = 0
                for expert, end in enumerate(boundaries):
                    expected[expert].add_(grad_output[start:end].float().T @ inputs.detach()[start:end].float())
                    start = end
                torch.testing.assert_close(state.unsharded_accumulated_grad, expected, rtol=0, atol=0)
                self.assertIsNone(weight.main_grad)
                self.assertIsNone(weight.grad)
                self.assertIsNone(state.unsharded_accumulated_grad.grad_fn)
        state.clear_unsharded_source_grad()
        self.assertIsNone(state.unsharded_accumulated_grad)

    def test_forward_and_input_gradient_match_unfused_path(self) -> None:
        """Only weight-gradient storage changes; the forward and dgrad kernels are identical."""
        state = make_expert_state(torch.randn(2, 4, 3, dtype=torch.bfloat16))
        inputs = torch.randn(5, 3, dtype=torch.bfloat16)
        groups = torch.tensor([2, 5], dtype=torch.int64)
        dy = torch.randn(5, 4, dtype=torch.bfloat16)
        outputs, input_grads = [], []
        for fused in (False, True):
            operand = inputs.clone().requires_grad_()
            weight = state.unsharded_param
            result = grouped_swiglu._get_npu_grouped_matmul().apply(
                operand, weight.transpose(-1, -2), groups, weight if fused else None)
            result.backward(dy)
            outputs.append(result.detach())
            input_grads.append(operand.grad)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        torch.testing.assert_close(input_grads[0], input_grads[1], rtol=0, atol=0)

    def test_fused_mode_rejects_missing_owner(self) -> None:
        """Do not silently cast the requested FP32 gradient back to BF16."""
        weight = torch.nn.Parameter(torch.zeros(2, 4, 3, dtype=torch.bfloat16))
        inputs = torch.zeros(2, 3, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "main_grad owner"):
            grouped_swiglu._get_npu_grouped_matmul().apply(
                inputs, weight.transpose(-1, -2), torch.tensor([1, 2]), weight)

    def test_frozen_weight_keeps_input_gradient_without_allocating_main_grad(self) -> None:
        """A frozen expert must not retain a full gradient skipped by FSDP reduction."""
        state = make_expert_state(torch.randn(2, 4, 3, dtype=torch.bfloat16))
        weight = state.unsharded_param.requires_grad_(False)
        inputs = torch.randn(5, 3, dtype=torch.bfloat16, requires_grad=True)
        groups = torch.tensor([2, 5])
        gradient = torch.randn(5, 4, dtype=torch.bfloat16)
        result = grouped_swiglu._get_npu_grouped_matmul().apply(
            inputs, weight.transpose(-1, -2), groups, weight)
        result.backward(gradient)
        expected = grouped_mm(x=[gradient], weight=[weight], group_list=groups, group_type=0)[0]
        torch.testing.assert_close(inputs.grad, expected, rtol=0, atol=0)
        self.assertIsNone(weight.grad)
        self.assertIsNone(weight.main_grad)
        self.assertFalse(weight.grad_added_to_main_grad)
        state.accumulate_unsharded_grad_if_needed()
        self.assertIsNone(state.unsharded_accumulated_grad)

    def test_shared_weight_checkpoint_and_no_sync_retain_each_contribution_once(self) -> None:
        """Real autograd replay and shared ordinary uses survive no-sync then sync."""
        initial = torch.randn(2, 4, 3, dtype=torch.bfloat16)
        inputs = [torch.randn(5, 3, dtype=torch.bfloat16) for _ in range(2)]
        gradients = [torch.randn(5, 4, dtype=torch.bfloat16) for _ in range(2)]
        groups = torch.tensor([2, 5])
        reduced = []
        for recompute in (False, True):
            with self.subTest(recompute=recompute):
                parameter = make_expert_state(initial)
                state = make_backward_state(parameter)
                weight = parameter.unsharded_param
                operation = grouped_swiglu._get_npu_grouped_matmul()

                def shared_forward(operand: torch.Tensor) -> torch.Tensor:
                    """Reuse one parameter twice inside a replayable region."""
                    first = operation.apply(operand, weight.transpose(-1, -2), groups, weight)
                    second = operation.apply(operand * 0.5, weight.transpose(-1, -2), groups, weight)
                    return first + second

                for step, (operand, gradient) in enumerate(zip(inputs, gradients)):
                    operand = operand.clone().requires_grad_()
                    output = checkpoint(shared_forward, operand, use_reentrant=False) if recompute else shared_forward(
                        operand)
                    torch.autograd.backward((output, weight.sum()), (gradient, torch.tensor(0.125)))
                    state.reduce_grads = step == 1
                    state.post_backward()
                    self.assertIsNone(weight.main_grad)
                    self.assertIsNone(weight.grad)
                    self.assertFalse(weight.grad_added_to_main_grad)
                    self.assertEqual(len(state.scheduler_ctx.pre_reduce_scatter_params), int(step == 1))
                    if step == 0:
                        self.assertIsNone(parameter.reduce_scatter_comm_ctx.reduce_scatter_output)
                output = parameter.reduce_scatter_comm_ctx.reduce_scatter_output.view(initial.shape)
                expected = torch.full(initial.shape, 0.25, dtype=torch.float64)
                for operand, gradient in zip(inputs, gradients):
                    for expert, (left, right) in enumerate(((0, 2), (2, 5))):
                        product = gradient[left:right].double().T @ operand[left:right].double()
                        expected[expert].add_(product * 1.5)
                torch.testing.assert_close(output.double(), expected, rtol=0, atol=1e-6)
                reduced.append(output.clone())
                parameter.clear_unsharded_source_grad()
                self.assertIsNone(parameter.unsharded_accumulated_grad)
        torch.testing.assert_close(reduced[0], reduced[1], rtol=0, atol=0)

    def test_existing_fused_producer_returns_neutral_autograd_placeholder(self) -> None:
        """The older public producer also composes with ordinary autograd gradients."""
        module = importlib.import_module("hyper_parallel.components.functional.grouped_matmul")
        self.enterContext(patch.object(module, "torch_npu", sys.modules["torch_npu"]))
        parameter = make_expert_state(torch.randn(2, 3, 4, dtype=torch.bfloat16))
        weight = parameter.unsharded_param
        weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
        inputs = torch.randn(5, 3, dtype=torch.bfloat16, requires_grad=True)
        gradient = torch.randn(5, 4, dtype=torch.bfloat16)
        output = module.grouped_matmul(inputs, weight, group_list=torch.tensor([2, 5]), gemm_fusion=True)
        torch.autograd.backward((output, weight.sum()), (gradient, torch.tensor(0.125)))
        parameter.accumulate_unsharded_grad_if_needed()
        expected = torch.full(weight.shape, 0.125, dtype=torch.float32)
        for expert, (left, right) in enumerate(((0, 2), (2, 5))):
            expected[expert].add_(inputs.detach()[left:right].float().T @ gradient[left:right].float())
        torch.testing.assert_close(parameter.unsharded_accumulated_grad, expected, rtol=0, atol=1e-6)
        self.assertIsNone(weight.main_grad)
        self.assertIsNone(weight.grad)

    def test_dtensor_is_rejected_before_any_local_kernel(self) -> None:
        """Reject distributed tensor types before inspecting their global metadata."""
        local = torch.zeros(2, 4, 3, dtype=torch.bfloat16)
        for tensor_type in (DTensor, TorchDTensor):
            distributed = torch.Tensor._make_subclass(tensor_type, local, require_grad=False)
            with self.subTest(tensor_type=tensor_type):
                owner = make_expert_state(local).unsharded_param
                with self.assertRaisesRegex(ValueError, "local tensors"):
                    grouped_swiglu._validate_fused_weight(distributed, local)
                with self.assertRaisesRegex(ValueError, "local tensors"):
                    grouped_swiglu._validate_fused_weight(owner, distributed)
                owner.main_grad = distributed
                with self.assertRaisesRegex(ValueError, "gradient buffer"):
                    grouped_swiglu._validate_fused_weight(owner, local)


if __name__ == "__main__":
    unittest.main()
