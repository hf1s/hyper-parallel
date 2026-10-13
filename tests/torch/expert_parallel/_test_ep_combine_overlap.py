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
"""Two-rank numerical acceptance for independent compute during EP combine.

Run directly for CPU/Gloo acceptance, or launch the pytest worker with
torchrun and HYPER_PARALLEL_EP_TEST_BACKEND=hccl/nccl on an accelerator host.
"""

from copy import deepcopy
import os
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from hyper_parallel.distributed.expert_parallel.collectives import ep_all_to_all


def _compare(device: torch.device) -> None:
    """Check uneven/zero splits, shared gradients and inference on two ranks."""
    rank = dist.get_rank()
    for empty in (False, True):
        counts = ([0, 3] if rank == 0 else [2, 1]) if not empty else [0, 0]
        receive = ([0, 2] if rank == 0 else [3, 1]) if not empty else [0, 0]
        torch.manual_seed(23 + rank)
        serial = torch.nn.Linear(4, 4, bias=False).to(device)
        candidate = deepcopy(serial)
        states = torch.randn(sum(counts), 4, device=device, requires_grad=True)
        copied_states = states.detach().clone().requires_grad_()
        reference_dispatch = ep_all_to_all(states, counts, receive, dist.group.WORLD)
        reference = ep_all_to_all(reference_dispatch, receive, counts, dist.group.WORLD) + serial(states)
        shared = []

        def compute_shared() -> None:
            """Retain the model-owned shared branch outside the communication Function."""
            shared.append(candidate(copied_states))

        dispatched = ep_all_to_all(copied_states, counts, receive, dist.group.WORLD)
        output = ep_all_to_all(dispatched, receive, counts, dist.group.WORLD,
                               overlap_fn=compute_shared) + shared[0]
        torch.testing.assert_close(output, reference)
        reference.square().sum().backward()
        output.square().sum().backward()
        torch.testing.assert_close(copied_states.grad, states.grad)
        torch.testing.assert_close(candidate.weight.grad, serial.weight.grad)
        shared.clear()
        with torch.no_grad():
            output = ep_all_to_all(dispatched, receive, counts, dist.group.WORLD,
                                   overlap_fn=compute_shared) + shared[0]
        torch.testing.assert_close(output, reference)
        if shared[0].requires_grad:
            raise AssertionError("Inference shared output unexpectedly requires gradients")


def test_ep_combine_overlap() -> None:
    """Exercise the real backend under torchrun, including accelerator streams."""
    backend = os.environ.get("HYPER_PARALLEL_EP_TEST_BACKEND", "gloo")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cpu")
    if backend == "hccl":
        import torch_npu  # pylint: disable=import-outside-toplevel
        torch_npu.npu.set_device(local_rank)
        device = torch.device("npu", local_rank)
    elif backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    dist.init_process_group(backend=backend)
    try:
        _compare(device)
    finally:
        dist.destroy_process_group()


def _spawn_worker(rank: int, rendezvous: str) -> None:
    """Use FileStore so Windows CPU acceptance does not require TCPStore/libuv."""
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        _compare(torch.device("cpu"))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_spawn_worker, args=(Path(directory, "rendezvous").as_uri(),), nprocs=2, join=True)
    print("Two-rank Gloo EP combine overlap: forward, backward, empty splits and inference passed")
