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
"""Real NPU communication checks for the opt-in FP32 Newton-Schulz pipeline."""

import torch
import torch.distributed as dist

from hyper_parallel import DTensor, Shard, SkipDTensorDispatch, distribute_tensor, init_device_mesh
from hyper_parallel.core.optimizer.muon import Muon
from tests.torch.optimizer._test_muon_e2e import (
    _assign_gradients,
    _build_model,
    _full_grads,
    _logical_optimizer_state,
    _logical_value,
    _make_optimizer,
    _optimizer_step,
    _shard_full_8p,
    _shard_hybrid_2x4,
)
from tests.torch.utils import init_dist


def _compare_fp32_trajectory(shard_model) -> None:
    """Compare actual FSDP/relay updates with an identically configured plain optimizer."""
    reference = _build_model()
    model = shard_model(_build_model())
    expected_optimizer = _make_optimizer(reference, ns_dtype="float32")
    optimizer = _make_optimizer(model, ns_dtype="float32")
    muon = optimizer.optimizers_dict["muon"]
    expected_muon = expected_optimizer.optimizers_dict["muon"]
    reference_parameters = dict(reference.named_parameters())
    for step in range(10):
        gradients = _full_grads(step, reference)
        _assign_gradients(reference, gradients)
        _assign_gradients(model, gradients)
        _optimizer_step(expected_optimizer)
        _optimizer_step(optimizer)
        muon._broadcast_state_fused_for_ckpt()  # pylint: disable=protected-access
        try:
            for name, parameter in model.named_parameters():
                expected = reference_parameters[name]
                torch.testing.assert_close(_logical_value(parameter), expected, rtol=1e-5, atol=1e-6)
                if parameter.ndim >= 2:
                    actual_momentum = _logical_optimizer_state(muon.state[parameter]["momentum_buffer"], parameter)
                    torch.testing.assert_close(actual_momentum, expected_muon.state[expected]["momentum_buffer"],
                                               rtol=1e-6, atol=1e-7)
        finally:
            muon.cleanup_synced_state()


def test_muon_fp32_ns_sharding() -> None:
    """Keep FP32 updates through gather, owner computation and relay in two layouts."""
    init_dist()
    for shard_model in (_shard_full_8p, _shard_hybrid_2x4):
        _compare_fp32_trajectory(shard_model)


def test_muon_fp32_ns_missing_gradients() -> None:
    """BF16 parameter storage must not select the no-gradient rank's collective dtype."""
    init_dist()
    mesh = init_device_mesh("npu", (2,), mesh_dim_names=("dp",))
    rank = dist.get_rank()
    for shape in ((64, 32), (1, 64)):
        torch.manual_seed(53)
        initial = torch.randn(shape, dtype=torch.bfloat16, device="npu") * 0.02
        reference = torch.nn.Parameter(initial.clone())
        parameter = torch.nn.Parameter(distribute_tensor(initial, mesh, [Shard(0)]))
        parameter.model_name = "weight"
        expected_optimizer = Muon([reference], ns_dtype="float32", weight_decay=0.0)
        optimizer = Muon([parameter], ns_dtype="float32", weight_decay=0.0)
        for _ in range(3):
            gradient = torch.randn_like(initial)
            first_shard_rows = (shape[0] + 1) // 2
            gradient[first_shard_rows:].zero_()
            reference.grad = gradient
            local = distribute_tensor(gradient, mesh, [Shard(0)])
            parameter.grad = local if rank == 0 else None
            with SkipDTensorDispatch():
                expected_optimizer.step()
                optimizer.step()
            expected_shard = reference.tensor_split(2, dim=0)[rank]
            torch.testing.assert_close(parameter.to_local(), expected_shard, rtol=0, atol=0)
        if shape[0] == 1 and rank == 1:
            assert isinstance(parameter, DTensor) and parameter.to_local().numel() == 0
