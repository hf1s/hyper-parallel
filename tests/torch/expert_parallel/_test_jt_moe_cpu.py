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
"""Two-rank CPU/Gloo acceptance of JT learned/fixed routing and EP gradients.

Run with python -m tests.torch.expert_parallel._test_jt_moe_cpu.
NPU imports use the CPU UT fixture; local expert math uses the Torch reference.
"""

from copy import deepcopy
from functools import partial
from pathlib import Path
import runpy
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from hyper_parallel.distributed.expert_parallel.experts import bind_local_expert_forward, ep_routed_forward


def _worker(rank: int, rendezvous: str) -> None:
    """Compare complete expert weights against two sharded ranks using real communication."""
    repo = Path(__file__).resolve().parents[3]
    runpy.run_path(str(repo / "tests/ut/models/jt_deepseek_v3/conftest.py"))
    # JT's optional NPU imports require the CPU fixture before model import.
    from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (  # pylint: disable=C0415
        JTDeepseekV3Config, JTDeepseekV3MoE,
    )

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        for fixed in (False, True):
            for padding in (False, True):
                torch.manual_seed(31)
                config = JTDeepseekV3Config.from_dict({
                    "hidden_size": 16, "moe_intermediate_size": 16, "n_routed_experts": 4, "n_shared_experts": 1,
                    "num_experts_per_tok": 2, "n_group": 1, "topk_group": 1, "routed_scaling_factor": 1.0,
                    "use_pad_tokens": padding, "moe_aux_loss_coeff": 0.01, "fix_router": fixed})
                serial = JTDeepseekV3MoE(config)
                for parameter in serial.parameters():
                    torch.nn.init.normal_(parameter, std=0.1)
                serial.gate.e_score_correction_bias.zero_()
                candidate = deepcopy(serial)
                # Also exercise combine overlap when both independent features are installed.
                if hasattr(candidate.config, "shared_expert_overlap"):
                    candidate.config.shared_expert_overlap = True
                for name in ("gate_up_proj", "down_proj"):
                    weight = getattr(candidate.experts, name)
                    setattr(candidate.experts, name, torch.nn.Parameter(weight[rank * 2:(rank + 1) * 2].clone()))
                bind_local_expert_forward(candidate, 2, use_grouped_gemm=False)
                candidate.ep_compute = partial(ep_routed_forward, candidate, router_fn=type(candidate).route,
                                               ep_group=dist.group.WORLD)
                states = torch.randn(1, 8, 16, requires_grad=True)
                local = states[:, rank * 4:(rank + 1) * 4].detach().clone().requires_grad_()
                reference = serial(states)
                output = candidate(local)
                torch.testing.assert_close(output, reference[:, rank * 4:(rank + 1) * 4])
                reference.square().sum().backward()
                output.square().sum().backward()
                torch.testing.assert_close(local.grad, states.grad[:, rank * 4:(rank + 1) * 4])
                reference_parameters = dict(serial.named_parameters())
                for name, parameter in candidate.named_parameters():
                    expected = reference_parameters[name].grad
                    if name.startswith("experts."):
                        expected = expected[rank * 2:(rank + 1) * 2]
                    else:
                        dist.all_reduce(parameter.grad)
                    torch.testing.assert_close(parameter.grad, expected, rtol=1e-4, atol=1e-6)
                if fixed:
                    counts = candidate.tokens_per_expert
                    torch.testing.assert_close(counts, torch.full_like(counts, 2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_worker, args=(Path(directory, "rendezvous").as_uri(),), nprocs=2, join=True)
    print("Two-rank JT EP: learned/fixed routes, padding, output and input/router/expert/shared gradients passed")
