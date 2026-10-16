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

"""Unit tests for the Muon optimizer."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer.optimizer import BaseDistributedOptimizer


def _plain_ns_group(**overrides):
    """Group dict covering every key consumed by the batched NS paths."""
    group = {
        "lr": 0.01,
        "matched_adamw_rms": 0.2,
        "ns_steps": 1,
        "ns_variant": "asym5",
        "ns_coefficients": None,
        "ns_epsilon": 1e-10,
        "zero_rms_scale_mode": "zero",
        "apply_lr_in_update": False,
    }
    group.update(overrides)
    return group


@pytest.mark.parametrize("update_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("apply_lr_in_update", [False, True])
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_single_shard_writeback_preserves_computed_updates(update_dtype, apply_lr_in_update, weight_decay):
    """A singleton shard group must apply the supplied update without an extra cast."""
    optimizer = object.__new__(Muon)
    optimizer._param_shard_metadata = {}  # pylint: disable=protected-access
    parameters = [torch.nn.Parameter(torch.tensor([[0.01, 0.02], [0.03, 0.04]])),
                  torch.nn.Parameter(torch.tensor([[0.1, -0.2, 0.3]]))]
    updates = [torch.tensor([[0.1234567, -0.2345678], [0.3456789, -0.456789]], dtype=update_dtype),
               torch.tensor([[0.1234567, 0.3456789, -0.5678912]], dtype=update_dtype)]
    originals = [update.clone() for update in updates]
    group = _plain_ns_group(lr=1e-4, weight_decay=weight_decay, apply_lr_in_update=apply_lr_in_update)
    alpha = 1.0 if apply_lr_in_update else -group["lr"]
    expected = [parameter.detach().clone().mul_(1 - group["lr"] * weight_decay).add_(update, alpha=alpha)
                for parameter, update in zip(parameters, updates)]

    with patch.object(optimizer, "_get_shard_info", return_value=((1,), (0,), (None,), 1)), \
            patch.object(optimizer, "_batched_relay_broadcast") as broadcast, torch.no_grad():
        optimizer._fused_broadcast_and_apply(  # pylint: disable=protected-access
            parameters, dict(zip(parameters, updates)), {parameter: (0,) for parameter in parameters},
            group, SimpleNamespace(layout_spec=None))

    broadcast.assert_not_called()
    for actual, reference in zip(parameters, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for actual, original in zip(updates, originals):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


def test_multi_shard_writeback_keeps_broadcast_packing():
    """The singleton shortcut must not bypass the existing multi-shard broadcast."""
    optimizer = object.__new__(Muon)
    optimizer._param_shard_metadata = {}  # pylint: disable=protected-access
    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    update = torch.tensor([[0.1234567, -0.2345678], [0.3456789, -0.456789]])
    group = _plain_ns_group(lr=1e-4, weight_decay=0.0, apply_lr_in_update=True)
    process_group = MagicMock()

    with patch.object(optimizer, "_get_shard_info", return_value=((2,), (0,), (process_group,), 2)), \
            patch.object(optimizer, "_batched_relay_broadcast") as broadcast, torch.no_grad():
        optimizer._fused_broadcast_and_apply(  # pylint: disable=protected-access
            [parameter], {parameter: update}, {parameter: (0,)}, group, SimpleNamespace(layout_spec=None))

    broadcast.assert_called_once()
    packed = broadcast.call_args.args[0][(0,)]
    assert packed.dtype == torch.bfloat16, f"expected={torch.bfloat16}, got={packed.dtype}"
    torch.testing.assert_close(parameter, update.bfloat16().float(), rtol=0, atol=0)


def test_compute_batched_ns_updates_accepts_plain_parameter_without_sharding():
    """The no-shard path should support ordinary torch parameters."""
    optimizer = object.__new__(Muon)
    optimizer.reshape_fn = None
    optimizer.ns_transform_fn = None
    optimizer.zeropower_fn = None
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    ns_inputs = {parameter: torch.ones(2, 2)}
    group = _plain_ns_group()

    updates = optimizer._compute_batched_ns_updates(  # pylint: disable=protected-access
        [parameter], ns_inputs, group, no_shard=True)

    assert updates[parameter].shape == parameter.shape


def test_compute_batched_ns_updates_with_transform_accepts_plain_parameter():
    """The ns_transform_fn routing path should also support plain parameters."""
    optimizer = object.__new__(Muon)
    optimizer.reshape_fn = None
    optimizer.ns_transform_fn = lambda param_fqn, ns_input: None
    optimizer.zeropower_fn = None
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    ns_inputs = {parameter: torch.ones(2, 2)}
    group = _plain_ns_group()

    updates = optimizer._compute_batched_ns_updates(  # pylint: disable=protected-access
        [parameter], ns_inputs, group, no_shard=True)

    assert updates[parameter].shape == parameter.shape


def test_muon_rejects_invalid_ns_variant():
    """ns_variant outside legacy/asym5/custom must raise."""
    with pytest.raises(ValueError, match="ns_variant"):
        Muon([torch.nn.Parameter(torch.ones(2, 2))], ns_variant="unknown")


def test_muon_rejects_ns_coefficients_without_custom_variant():
    """ns_coefficients is only valid together with ns_variant='custom'."""
    with pytest.raises(ValueError, match="ns_coefficients"):
        Muon(
            [torch.nn.Parameter(torch.ones(2, 2))],
            ns_variant="asym5",
            ns_coefficients=[(3.4445, -4.7750, 2.0315)],
        )


def test_muon_rejects_missing_or_short_ns_coefficients():
    """Custom NS requires at least ns_steps coefficient groups."""
    with pytest.raises(ValueError, match="ns_coefficients"):
        Muon([torch.nn.Parameter(torch.ones(2, 2))], ns_variant="custom")
    with pytest.raises(ValueError, match="exceeds"):
        Muon(
            [torch.nn.Parameter(torch.ones(2, 2))],
            ns_variant="custom",
            ns_steps=2,
            ns_coefficients=[(3.4445, -4.7750, 2.0315)],
        )


def test_muon_rejects_malformed_ns_coefficients():
    """Each custom coefficient group must hold three finite values."""
    with pytest.raises(ValueError, match="exactly three"):
        Muon(
            [torch.nn.Parameter(torch.ones(2, 2))],
            ns_variant="custom",
            ns_steps=1,
            ns_coefficients=[(3.4445, -4.7750)],
        )
    with pytest.raises(ValueError, match="finite"):
        Muon(
            [torch.nn.Parameter(torch.ones(2, 2))],
            ns_variant="custom",
            ns_steps=1,
            ns_coefficients=[(3.4445, float("inf"), 2.0315)],
        )


def test_muon_rejects_invalid_ns_epsilon_and_zero_rms_scale_mode():
    """ns_epsilon must be finite non-negative; zero_rms_scale_mode must be known."""
    with pytest.raises(ValueError, match="ns_epsilon"):
        Muon([torch.nn.Parameter(torch.ones(2, 2))], ns_epsilon=-1e-10)
    with pytest.raises(ValueError, match="zero_rms_scale_mode"):
        Muon([torch.nn.Parameter(torch.ones(2, 2))], zero_rms_scale_mode="keep")


def test_async_hierarchical_broadcast_waits_between_mesh_dimensions():
    """A relay dimension must wait before forwarding the shared buffer."""
    first_work = MagicMock()
    final_work = MagicMock()
    process_groups = (MagicMock(), MagicMock())

    with patch(
            "hyper_parallel.core.optimizer.optimizer.dist.get_global_rank",
            side_effect=[0, 1],
    ), patch(
            "hyper_parallel.core.optimizer.optimizer.dist.broadcast",
            side_effect=[first_work, final_work],
    ) as broadcast:
        handles = BaseDistributedOptimizer._hierarchical_broadcast_buffer_async(  # pylint: disable=protected-access
            torch.ones(4),
            src_coord=(0, 0),
            replicate_pgs=process_groups,
            local_coord=(0, 0),
        )

    assert broadcast.call_count == 2
    first_work.wait.assert_called_once_with()
    final_work.wait.assert_not_called()
    assert handles == [final_work]
