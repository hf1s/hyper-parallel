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

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.nn.functional import all_gather
from transformers.loss.loss_utils import ForCausalLMLoss

from hyper_parallel.components.losses.mtp import iter_mtp_targets
from hyper_parallel.components.losses.projected_cross_entropy import projected_cross_entropy
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.models._transformers.loss_parallel import causal_lm_loss_parallel


def setup_module() -> None:
    """Initialize only a CPU group; this worker never acquires NPU devices."""
    dist.init_process_group("gloo")


def teardown_module() -> None:
    """Release the process group after comparisons."""
    dist.destroy_process_group()


class SequenceGatherHead(nn.Module):
    """Gather SP states before applying one vocabulary shard of the output head."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.clone())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Use a differentiable gather, including its matching backward reduction."""
        gathered = torch.cat(all_gather(hidden.contiguous()), dim=1)
        return nn.functional.linear(gathered, self.weight)


def test_sequence_and_vocab_shards() -> None:
    """Tail chunks, empty chunks and two MTP target depths keep reference gradients."""
    rank = dist.get_rank()
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
    torch.manual_seed(18)
    hidden = torch.randn(1, 10, 5)
    weight = torch.randn(13, 5)
    labels = torch.tensor([[-100, -100, 2, 4, -100, -100, -100, 8, 11, 12]])
    for targets in [labels, *iter_mtp_targets(labels, 2, sequence_ends=(1, 4, 10))]:
        local = hidden[:, rank * 5:(rank + 1) * 5].clone().requires_grad_()
        head = SequenceGatherHead(weight[rank * 7:min((rank + 1) * 7, 13)])
        actual = projected_cross_entropy(local, targets, head=head, loss_fn=causal_lm_loss_parallel,
                                          vocab_size=13, chunk_size=4, sequence_parallel_size=2, tp_mesh=mesh)
        reference_hidden = hidden.clone().requires_grad_()
        reference_weight = weight.clone().requires_grad_()
        expected = ForCausalLMLoss(logits=nn.functional.linear(reference_hidden, reference_weight), labels=None,
                                   vocab_size=13, shift_labels=targets,
                                   num_items_in_batch=(targets != -100).sum().clamp_min(1))
        torch.testing.assert_close(actual, expected)
        (actual * 0.37).backward()
        (expected * 0.37).backward()
        torch.testing.assert_close(local.grad, reference_hidden.grad[:, rank * 5:(rank + 1) * 5],
                                   atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(head.weight.grad, reference_weight.grad[rank * 7:min((rank + 1) * 7, 13)],
                                   atol=1e-6, rtol=1e-5)
