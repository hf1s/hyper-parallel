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
"""JT shared-expert scheduling and numerical contracts on CPU."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch

import torch
from torch.nn import functional as F

from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3Config, JTDeepseekV3MoE
from hyper_parallel.models.jt_deepseek_v3.adapter.distributed.ep_compute import jt_deepseek_v3_ep_compute_fn
from tests.ut.models.jt_deepseek_v3.test_modeling_jt_deepseek_v3 import small_config


class TestSharedExpertOverlap(unittest.TestCase):
    """Compare identical states and verify shared compute precedes the combine wait."""

    def test_configuration(self):
        """The opt-in flag is validated and survives serialization."""
        self.assertFalse(JTDeepseekV3Config().shared_expert_overlap)
        config = JTDeepseekV3Config(shared_expert_overlap=True)
        self.assertTrue(JTDeepseekV3Config.from_dict(config.to_dict()).shared_expert_overlap)
        with self.assertRaisesRegex(ValueError, "boolean"):
            JTDeepseekV3Config(shared_expert_overlap="true")

    def test_local_and_ep_forward_backward(self):
        """Shared outputs, all parameter gradients and counts match, with and without pad tokens."""
        for padding in (False, True):
            for expert_parallel in (False, True):
                with self.subTest(padding=padding, ep=expert_parallel):
                    self._compare(padding, expert_parallel)

    def _compare(self, padding, expert_parallel):
        """Run two steps so retained callback outputs cannot hide a stale-output bug."""
        torch.manual_seed(19)
        config = small_config()
        config.use_pad_tokens = padding
        serial = JTDeepseekV3MoE(config)
        for parameter in serial.parameters():
            torch.nn.init.normal_(parameter, std=0.1)
        serial.gate.e_score_correction_bias.zero_()
        candidate = deepcopy(serial)
        candidate.config.shared_expert_overlap = True
        group = Mock()
        group.size.return_value = 1
        mesh = Mock()
        mesh.get_group.return_value = group
        mesh.__getitem__ = Mock(return_value=SimpleNamespace(size=lambda: 1))
        if expert_parallel:
            for module in (serial, candidate):
                jt_deepseek_v3_ep_compute_fn(module=module, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=mesh)

        events = []

        def exchange(outputs: list[torch.Tensor], inputs: list[torch.Tensor], **kwargs: Any) -> Any:
            """A deferred CPU copy models data becoming valid only after work.wait."""
            if kwargs.get("async_op", False):
                events.append("launch")

                @torch.no_grad()
                def wait() -> None:
                    """Model untracked receive writes at collective completion."""
                    events.append("wait")
                    for output, source in zip(outputs, inputs):
                        output.copy_(source)

                return SimpleNamespace(wait=wait)
            for output, source in zip(outputs, inputs):
                output.copy_(source)
            return None

        def grouped(inputs: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor,
                    counts: torch.Tensor) -> torch.Tensor:
            """Use actual Torch operations as the oracle for the NPU grouped kernel."""
            pieces = []
            for expert, tokens in enumerate(inputs.split(counts.tolist())):
                gate, up = F.linear(tokens, gate_up[expert]).chunk(2, -1)  # pylint: disable=not-callable
                pieces.append(F.linear(F.silu(gate) * up, down[expert]))  # pylint: disable=not-callable
            return torch.cat(pieces)

        hook = candidate.shared_experts.register_forward_hook(
            lambda _module, _inputs, _output: events.append("shared"))
        try:
            with patch("hyper_parallel.distributed.expert_parallel.collectives.dist.get_backend",
                       return_value="hccl"), \
                 patch("hyper_parallel.distributed.expert_parallel.experts.dist.get_rank", return_value=0), \
                 patch("hyper_parallel.distributed.expert_parallel.experts.dist.all_to_all_single",
                       side_effect=lambda output, source, **_kwargs: output.copy_(source)), \
                 patch("hyper_parallel.distributed.expert_parallel.collectives.dist.all_to_all",
                       side_effect=exchange), \
                 patch("hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3.npu_grouped_swiglu",
                       side_effect=grouped):
                for _step in range(2):
                    events.clear()
                    states = torch.randn(1, 5, 16, requires_grad=True)
                    copied_states = states.detach().clone().requires_grad_()
                    expected, actual = serial(states), candidate(copied_states)
                    torch.testing.assert_close(actual, expected)
                    if expert_parallel:
                        self.assertEqual(events, ["launch", "shared", "wait"])
                    else:
                        self.assertEqual(events, ["shared"])
                    for output, module in ((expected, serial), (actual, candidate)):
                        (output.square().sum() + module.auxiliary_loss).backward()
                    torch.testing.assert_close(copied_states.grad, states.grad)
                    for expected_parameter, actual_parameter in zip(serial.parameters(), candidate.parameters()):
                        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)
                    torch.testing.assert_close(candidate.tokens_per_expert, serial.tokens_per_expert)
                    serial.zero_grad()
                    candidate.zero_grad()
        finally:
            hook.remove()
