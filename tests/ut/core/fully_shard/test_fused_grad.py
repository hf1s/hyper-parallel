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
"""Fused weight gradients must reach the existing FSDP accumulation path intact."""
# pylint: disable=protected-access

import unittest
from types import SimpleNamespace

import torch

from hyper_parallel.core.fully_shard.hsdp_param import AllGatherCommCtx, HSDPParamV2
from hyper_parallel.core.fully_shard.hsdp_state import HSDPStateV2
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerContext
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy


def make_state() -> HSDPParamV2:
    """Construct real parameter storage without a distributed process group."""
    state = object.__new__(HSDPParamV2)
    state._orig_size = torch.Size((2, 3))
    state._contiguous_orig_stride = (3, 1)
    state.unsharded_param_buffers = [torch.zeros(6, dtype=torch.bfloat16)]
    state.source_shard_info = None
    state.allgather_comm_ctx = AllGatherCommCtx()
    state.sharded_param = SimpleNamespace(requires_grad=True)
    state.reduce_dtype = torch.float32
    state.mp_policy = MixedPrecisionPolicy(apply_grad_on_fp32_main_grad=True)
    state.unsharded_accumulated_grad = None
    state.init_unsharded_param()
    return state


class TestFusedGradientAccumulation(unittest.TestCase):
    """Exercise mixed ordinary/fused gradients and delayed synchronization."""

    def test_unsharded_parameter_exposes_lazy_main_grad(self) -> None:
        """The stable parameter advertises a lazy buffer only for FP32 reduction."""
        state = make_state()
        self.assertTrue(hasattr(state.unsharded_param, "main_grad"))
        self.assertIsNone(state.unsharded_param.main_grad)
        self.assertFalse(state.unsharded_param.grad_added_to_main_grad)

    def test_unused_unit_post_backward_does_not_require_unsharded_storage(self) -> None:
        """Root finalization also visits units whose forward did not execute."""
        parameter = object.__new__(HSDPParamV2)
        parameter.unsharded_accumulated_grad = None
        parameter.unsharded_param_buffers = []
        state = object.__new__(HSDPStateV2)
        state.modules = ()
        state.hsdp_params = [parameter]
        state.comm_fusion_policy = SimpleNamespace(enable_comm_fusion=False)
        state.reshard_after_backward = False
        state.scheduler_ctx = HSDPSchedulerContext()
        for reduce_grads in (False, True):
            with self.subTest(reduce_grads=reduce_grads):
                state.reduce_grads = reduce_grads
                state.post_backward()
                self.assertIsNone(parameter.unsharded_accumulated_grad)
                self.assertFalse(hasattr(parameter, "_unsharded_param"))

    def test_fused_and_ordinary_contributions_are_consumed_once(self) -> None:
        """A fused gradient, ordinary gradient and prior microbatch all contribute."""
        for method_name in ("accumulate_unsharded_grad_if_needed", "to_accumulated_grad_if_needed"):
            with self.subTest(method=method_name):
                state = make_state()
                parameter = state.unsharded_param
                prior = torch.full((2, 3), 0.125, dtype=torch.float32)
                fused = torch.full((2, 3), 0.001234567, dtype=torch.float32)
                ordinary = torch.full((2, 3), 0.0625, dtype=torch.bfloat16)
                state.unsharded_accumulated_grad = prior.clone()
                parameter.main_grad = fused.clone()
                parameter.grad_added_to_main_grad = True
                parameter.grad = ordinary.clone()
                consume = getattr(state, method_name)
                consume()
                torch.testing.assert_close(state.unsharded_accumulated_grad, prior + fused + ordinary.float(),
                                           rtol=0, atol=0)
                self.assertIsNone(parameter.main_grad)
                self.assertIsNone(parameter.grad)
                self.assertFalse(parameter.grad_added_to_main_grad)
                before = state.unsharded_accumulated_grad.clone()
                consume()
                torch.testing.assert_close(state.unsharded_accumulated_grad, before, rtol=0, atol=0)

    def test_fused_gradient_without_autograd_gradient_is_preserved(self) -> None:
        """A delayed-sync transition retains a fused buffer even when grad is None."""
        state = make_state()
        gradient = torch.full((2, 3), 0.001234567, dtype=torch.float32)
        state.unsharded_param.main_grad = gradient
        state.unsharded_param.grad_added_to_main_grad = True
        state.to_accumulated_grad_if_needed()
        self.assertIs(state.unsharded_accumulated_grad, gradient)
        self.assertIsNone(state.unsharded_param.main_grad)
        state.clear_unsharded_source_grad()
        self.assertIsNone(state.unsharded_accumulated_grad)

    def test_unused_buffer_does_not_create_a_gradient(self) -> None:
        """An unused preallocated buffer does not make a frozen path look used."""
        state = make_state()
        state.unsharded_param.main_grad = torch.ones((2, 3))
        state.unsharded_param.grad_added_to_main_grad = False
        state.to_accumulated_grad_if_needed()
        self.assertIsNone(state.unsharded_accumulated_grad)

    def test_malformed_fused_buffer_fails_before_consumption(self) -> None:
        """Reject buffers that cannot represent the local FP32 parameter gradient."""
        for gradient in (torch.zeros(2, 3, dtype=torch.bfloat16), torch.zeros(3, 2),
                         torch.zeros(2, 3, requires_grad=True)):
            with self.subTest(shape=gradient.shape, dtype=gradient.dtype):
                state = make_state()
                state.unsharded_param.main_grad = gradient
                state.unsharded_param.grad_added_to_main_grad = True
                with self.assertRaisesRegex(ValueError, "fused.*gradient"):
                    state.accumulate_unsharded_grad_if_needed()
                self.assertIs(state.unsharded_param.main_grad, gradient)


if __name__ == "__main__":
    unittest.main()
