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
"""JT matrix-layout optimizer parity between complete and FSDP-sharded parameters."""

from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel import DTensor, SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.models.jt_deepseek_v3.adapter.runtime.jt_optimizer import build_optimizer
from tests.torch.optimizer._test_muon_e2e import (
    _assert_close, _assign_gradients, _full_grads, _logical_optimizer_state,
)
from tests.torch.utils import init_dist


class LayoutModel(nn.Module):
    """Use real JT parameter suffixes with unequal periodic and uneven fused rows."""

    def __init__(self) -> None:
        """Build deterministic small MLA, SwiGLU, expert and normalization parameters."""
        super().__init__()
        torch.manual_seed(17)
        self.config = SimpleNamespace(q_lora_rank=8, kv_lora_rank=4, qk_nope_head_dim=4,
                                      qk_rope_head_dim=2, v_head_dim=4, num_attention_heads=4,
                                      moe_router_enable_expert_bias=False)
        self.qk_clip_group = None
        self.block = nn.Module()
        attention = self.block.self_attn = nn.Module()
        attention.linear_qkv = nn.Linear(16, 14, bias=False)
        attention.q_b_proj = nn.Linear(8, 24, bias=False)
        attention.kv_b_proj = nn.Linear(4, 32, bias=False)
        mlp = self.block.mlp = nn.Module()
        mlp.linear_fc1 = nn.Linear(16, 24, bias=False)
        mlp.experts = nn.Module()
        mlp.experts.gate_up_proj = nn.Parameter(torch.randn(4, 12, 16) * 0.02)
        self.norm = nn.LayerNorm(16)


def make_optimizer(model: nn.Module) -> object:
    """Construct the actual model adapter and preserve both Muon and AdamW groups.

    Args:
        model: Complete or FSDP-sharded test projection model.

    Returns:
        The production chained optimizer with JT layout callbacks.
    """
    return build_optimizer(
        model=model, qk_clip_threshold=100, muon_layout="mindformers",
        muon_config={"lr": 0.01, "ns_variant": "legacy", "ns_steps": 4, "ns_epsilon": 1e-7},
        adamw_config={"adamw_lr": 0.001}).get_optimizer()


def logical_value(value: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
    """Gather the fixture's uneven dimension-zero shards with explicit padding.

    Args:
        value: Parameter or corresponding optimizer state.
        parameter: Parameter carrying the complete logical shape and shard layout.

    Returns:
        Complete logical value on each rank, including uneven final shards.
    """
    if not isinstance(parameter, DTensor) or not parameter.layout.has_uneven_shard:
        return _logical_optimizer_state(value, parameter)
    local = value.to_local() if isinstance(value, DTensor) else value
    rows = (parameter.shape[0] + dist.get_world_size() - 1) // dist.get_world_size()
    padded = local.new_zeros((rows, *local.shape[1:]))
    padded[:local.shape[0]].copy_(local)
    outputs = [torch.empty_like(padded) for _ in range(dist.get_world_size())]
    dist.all_gather(outputs, padded)
    return torch.cat(outputs, dim=0)[:parameter.shape[0]]


def run_parity(world_size: int) -> None:
    """Compare each full parameter and momentum buffer after identical logical steps.

    The builder binds JTMuonLayout.from_config, reshape_fn and ns_transform_fn;
    these tests exercise their parameter layouts through the actual Muon step.

    Args:
        world_size: Size of the flat FSDP shard mesh.
    """
    init_dist()
    reference = LayoutModel().npu()
    reference_optimizer = make_optimizer(reference)
    mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("dp",))
    candidate = fully_shard(LayoutModel().npu(), mesh=mesh, reshard_after_forward=True,
                            mp_policy=MixedPrecisionPolicy())
    candidate_optimizer = make_optimizer(candidate)
    candidate_muon = candidate_optimizer.optimizers_dict["muon"]
    for step in range(3):
        gradients = _full_grads(step, reference)
        _assign_gradients(reference, gradients)
        _assign_gradients(candidate, gradients)
        with SkipDTensorDispatch():
            reference_optimizer.step()
            candidate_optimizer.step()
        candidate_muon._broadcast_state_fused_for_ckpt()  # pylint: disable=protected-access
        try:
            expected = dict(reference.named_parameters())
            for name, parameter in candidate.named_parameters():
                _assert_close(expected[name], logical_value(parameter, parameter), f"step{step} {name}", 1e-3, 1e-5)
                if parameter.ndim >= 2:
                    ref_muon = reference_optimizer.optimizers_dict["muon"]
                    actual_momentum = logical_value(
                        candidate_muon.state[parameter]["momentum_buffer"], parameter)
                    _assert_close(ref_muon.state[expected[name]]["momentum_buffer"],
                                  actual_momentum,
                                  f"step{step} momentum {name}", 1e-5, 1e-7)
        finally:
            candidate_muon.cleanup_synced_state()


def test_jt_muon_layout_single_card() -> None:
    """
    Feature: JT layout callbacks through the production optimizer builder.
    Description: Compare three complete and FSDP-mesh-one steps on identical gradients.
    Expectation: All logical parameters and Muon momentum buffers match.
    """
    run_parity(1)


def test_jt_muon_layout_fsdp_four_cards() -> None:
    """
    Feature: JT logical matrix transforms after FSDP gathering.
    Description: Compare four sharded ranks with the complete single-card view, including uneven QKV rows.
    Expectation: Full parameter updates and momentum buffers match after every step.
    """
    run_parity(4)
