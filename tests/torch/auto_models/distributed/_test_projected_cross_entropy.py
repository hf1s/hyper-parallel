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
"""Check rank-order chunk gathering against a full-vocabulary reference."""

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.nn.functional import all_gather
from torch.utils.checkpoint import DefaultDeviceType
from transformers.loss.loss_utils import ForCausalLMLoss

from hyper_parallel.components.losses._vocab_parallel_cross_entropy import vocab_parallel_cross_entropy_local
from hyper_parallel.components.losses.mtp import iter_mtp_targets
from hyper_parallel.components.losses.chunked_cross_entropy import projected_cross_entropy
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.models._transformers.loss_parallel import causal_lm_loss_parallel
from hyper_parallel.models.jt_deepseek_v3.adapter.jt_builder import _bind_statistics_groups


@pytest.fixture(autouse=True, scope="module")
def cpu_checkpoint_device() -> Iterator[None]:
    """CPU-only ranks must not initialize an installed accelerator during recomputation."""
    previous = DefaultDeviceType.get_device_type()
    DefaultDeviceType.set_device_type("cpu")
    try:
        yield
    finally:
        DefaultDeviceType.set_device_type(previous)


def setup_module() -> None:
    """Initialize only a CPU group; this worker never acquires NPU devices."""
    dist.init_process_group("gloo")


def teardown_module() -> None:
    """Release the process group after comparisons."""
    dist.destroy_process_group()


class SequenceGatherHead(nn.Module):
    """Gather SP states before applying one vocabulary shard of the output head."""

    def __init__(self, weight: torch.Tensor) -> None:
        """Own one vocabulary shard of the reference projection weight."""
        super().__init__()
        self.weight = nn.Parameter(weight.clone())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Use a differentiable gather, including its matching backward reduction."""
        gathered = torch.cat(all_gather(hidden.contiguous()), dim=1)
        return nn.functional.linear(gathered, self.weight)


class TestParallelCrossEntropy:
    """Run the essential CE and projection contracts in one two-rank launch."""

    @pytest.mark.parametrize("reduction,target_values", (
        ("none", (0, -100, 4, 2)), ("sum", (0, -100, 4, 2)), ("mean", (0, -100, 4, 2)),
        ("sum", (0, -100, 2, 1)), ("mean", (-100, -100, -100, -100)),
    ))
    def test_vocab_parallel_reductions(self, reduction: str, target_values: tuple[int, ...]) -> None:
        """Every rank must return full-vocabulary losses and the matching local gradient."""
        rank = dist.get_rank()
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        target = torch.tensor(target_values)
        logits = torch.randn(4, 5, generator=torch.Generator().manual_seed(23), dtype=torch.float64)
        local_logits = logits.chunk(2, dim=-1)[rank].clone().requires_grad_()
        reference_logits = logits.clone().requires_grad_()
        actual = vocab_parallel_cross_entropy_local(
            local_logits, target, vocab_size=5, mesh=mesh, reduction=reduction,
        )
        expected = nn.functional.cross_entropy(reference_logits, target, reduction=reduction)
        torch.testing.assert_close(actual.reshape(expected.shape), expected, equal_nan=True)
        scale = torch.linspace(0.3, 1.2, expected.numel(), dtype=expected.dtype)
        actual.backward(scale.reshape(actual.shape))
        expected.backward(scale.reshape(expected.shape))
        torch.testing.assert_close(local_logits.grad, reference_logits.grad.chunk(2, dim=-1)[rank])

    @pytest.mark.parametrize("tp_size,sequence_parts", ((1, 1), (2, 1), (2, 2)))
    def test_sequence_and_vocab_shards(self, tp_size: int, sequence_parts: int) -> None:
        """Tail chunks, empty chunks and two MTP target depths keep reference gradients."""
        rank = dist.get_rank() % tp_size
        device_mesh = init_device_mesh("cpu", (2 // tp_size, tp_size), mesh_dim_names=("dp", "tp"))
        torch.manual_seed(18)
        hidden = torch.randn(1, 10, 5)
        weight = torch.randn(13, 5)
        labels = torch.tensor([[-100, -100, 2, 4, -100, -100, -100, 8, 11, 12]])
        tails = torch.zeros_like(labels, dtype=torch.bool)
        tails[:, [0, 3, 9]] = True
        for targets in [labels, *iter_mtp_targets(labels, 2, sequence_end_mask=tails)]:
            start, stop = (rank * 5, (rank + 1) * 5) if sequence_parts == 2 else (0, 10)
            local = hidden[:, start:stop].clone().requires_grad_()
            local_weight = weight.chunk(tp_size, dim=0)[rank]
            if sequence_parts == 2:
                head = SequenceGatherHead(local_weight)
            else:
                head = nn.Linear(5, local_weight.shape[0], bias=False)
                head.load_state_dict({"weight": local_weight})
            context = SimpleNamespace(dp_cp_mesh=None, tp_size=tp_size, sequence_parallel=sequence_parts > 1,
                                      loss_parallel=True, device_mesh=device_mesh)
            _bind_statistics_groups(head, context)
            actual = projected_cross_entropy(local, targets, head=head, loss_fn=causal_lm_loss_parallel,
                                              vocab_size=13, chunk_size=4,
                                              sequence_parallel_size=sequence_parts, tp_mesh=head.loss_tp_mesh)
            reference_hidden = hidden.clone().requires_grad_()
            reference_weight = weight.clone().requires_grad_()
            expected = ForCausalLMLoss(logits=nn.functional.linear(reference_hidden, reference_weight), labels=None,
                                       vocab_size=13, shift_labels=targets,
                                       num_items_in_batch=(targets != -100).sum().clamp_min(1))
            torch.testing.assert_close(actual, expected)
            (actual * 0.37).backward()
            (expected * 0.37).backward()
            if tp_size > 1 and sequence_parts == 1:
                # Without SP's gather backward, the caller's replicated-input boundary sums vocabulary contributions.
                dist.all_reduce(local.grad, group=head.loss_tp_mesh.get_group())
            torch.testing.assert_close(local.grad, reference_hidden.grad[:, start:stop],
                                       atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(head.weight.grad, reference_weight.grad.chunk(tp_size, dim=0)[rank],
                                       atol=1e-6, rtol=1e-5)
