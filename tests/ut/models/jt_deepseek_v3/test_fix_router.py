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
"""Fixed routing determinism, load balance, learned weights and EP contracts."""

from copy import deepcopy
from functools import partial
from typing import Any
import unittest
from unittest.mock import Mock, patch

import torch
from torch.nn import functional as F

from hyper_parallel.distributed.expert_parallel.experts import bind_local_expert_forward, ep_routed_forward
from hyper_parallel.distributed.expert_parallel.routing import MOE_ROUTER_ADAPTERS
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (
    JTDeepseekV3Config, JTDeepseekV3ForCausalLM, JTDeepseekV3MoE,
)
from tests.ut.models.jt_deepseek_v3.test_modeling_jt_deepseek_v3 import small_config


def fixed_config(**overrides: Any) -> JTDeepseekV3Config:
    """Enable deterministic routing without changing the standard fixture."""
    return JTDeepseekV3Config.from_dict(small_config().to_dict() | {"fix_router": True} | overrides)


def initialize_moe(config: JTDeepseekV3Config) -> JTDeepseekV3MoE:
    """Initialize packed experts and router buffers independently of a full LM."""
    torch.manual_seed(29)
    module = JTDeepseekV3MoE(config)
    for parameter in module.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    module.gate.e_score_correction_bias.zero_()
    return module


class TestFixedRouter(unittest.TestCase):
    """Verify the benchmark option while retaining normal routing as the default."""

    def test_configuration(self):
        """The flag defaults off, roundtrips and rejects non-boolean or impossible top-k."""
        self.assertFalse(JTDeepseekV3Config().fix_router)
        self.assertTrue(JTDeepseekV3Config.from_dict(fixed_config().to_dict()).fix_router)
        for value in ("true", 1, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "boolean"):
                fixed_config(fix_router=value)
        for top_k in (0, 5):
            with self.subTest(top_k=top_k), self.assertRaisesRegex(ValueError, "num_experts_per_tok"):
                fixed_config(num_experts_per_tok=top_k)

    def test_default_matches_learned_router(self):
        """Disabled mode keeps correction-bias selection, probabilities and auxiliary statistics."""
        module = initialize_moe(small_config())
        hidden = torch.randn(1, 5, 16)
        expected_indices, expected_weights = MOE_ROUTER_ADAPTERS["deepseekv3"](module, hidden)
        prefixed = torch.cat((hidden.new_zeros(1, module.padding, 16), hidden), dim=1)
        indices, weights = module.route(prefixed)
        torch.testing.assert_close(indices[module.padding:], expected_indices)
        torch.testing.assert_close(weights[module.padding:], expected_weights)
        self.assertIsNone(module.gate.router_logits)

    def test_balanced_unique_and_deterministic(self):
        """Arbitrary token/expert/top-k counts are balanced, unique and independent of RNG."""
        for experts, top_k, tokens in ((4, 2, 8), (6, 4, 5), (4, 1, 3), (4, 4, 2), (4, 2, 0)):
            with self.subTest(experts=experts, top_k=top_k, tokens=tokens):
                module = initialize_moe(fixed_config(n_routed_experts=experts, num_experts_per_tok=top_k))
                scores = torch.rand(tokens, experts)
                rng_state = torch.random.get_rng_state()
                indices, _weights = module.fixed_routing(scores)
                self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_state))
                expected = torch.arange(tokens * top_k).reshape(tokens, top_k) % experts
                torch.testing.assert_close(indices, expected)
                counts = torch.bincount(indices.flatten(), minlength=experts)
                self.assertLessEqual(int(counts.max() - counts.min()), 1)
                self.assertTrue(all(row.unique().numel() == top_k for row in indices))
                other_indices, _weights = module.fixed_routing(scores.flip(-1))
                torch.testing.assert_close(other_indices, indices)

    def test_learned_weights_padding_bias_and_gradients(self):
        """Fix only assignments; keep FP32 scores, differentiable weights, scaling and real-token counts."""
        for normalized in (False, True):
            for padding in (False, True):
                with self.subTest(normalized=normalized, padding=padding):
                    module = initialize_moe(fixed_config(norm_topk_prob=normalized, use_pad_tokens=padding,
                                                         routed_scaling_factor=2.827))
                    module.to(dtype=torch.bfloat16)
                    hidden = torch.randn(1, 5 + module.padding, 16, dtype=torch.bfloat16, requires_grad=True)
                    with patch.dict(MOE_ROUTER_ADAPTERS, {"deepseekv3": Mock(side_effect=AssertionError)}):
                        indices, weights = module.route(hidden)
                    scores = F.linear(  # pylint: disable=not-callable
                        hidden[:, module.padding:].reshape(-1, 16).float(), module.gate.weight.float()).sigmoid()
                    expected = scores.gather(1, indices[module.padding:])
                    if normalized:
                        expected = expected / (expected.sum(-1, keepdim=True) + 1e-20)
                    torch.testing.assert_close(weights[module.padding:], expected * 2.827)
                    self.assertEqual(weights.dtype, torch.float32)
                    self.assertEqual(weights[:module.padding].count_nonzero().item(), 0)
                    counts = torch.bincount(indices[module.padding:].flatten(), minlength=4).float()
                    torch.testing.assert_close(module.tokens_per_expert, counts)
                    (weights.square().sum() + module.auxiliary_loss).backward()
                    self.assertGreater(module.gate.weight.grad.abs().sum().item(), 0)
                    self.assertTrue(torch.isfinite(hidden.grad).all())
                    module.gate.e_score_correction_bias.fill_(10000)
                    changed_indices, _weights = module.route(hidden.detach())
                    torch.testing.assert_close(changed_indices, indices)

    def test_ep_forward_backward(self):
        """The common EP path consumes fixed routes and matches local output and all gradients."""
        serial = initialize_moe(fixed_config())
        candidate = deepcopy(serial)
        bind_local_expert_forward(candidate, ep_size=1)
        group = Mock()
        group.size.return_value = 1
        candidate.ep_compute = partial(ep_routed_forward, candidate, router_fn=type(candidate).route, ep_group=group)

        def exchange(outputs: list[torch.Tensor], inputs: list[torch.Tensor], **_kwargs: Any) -> None:
            """Mirror a rank-local collective with untracked receive writes."""
            for output, source in zip(outputs, inputs):
                output.copy_(source)

        with patch("hyper_parallel.distributed.expert_parallel.collectives.dist.get_backend",
                   return_value="hccl"), \
             patch("hyper_parallel.distributed.expert_parallel.experts.dist.get_rank", return_value=0), \
             patch("hyper_parallel.distributed.expert_parallel.experts.dist.all_to_all_single",
                   side_effect=lambda output, source, **_kwargs: output.copy_(source)), \
             patch("hyper_parallel.distributed.expert_parallel.collectives.dist.all_to_all", side_effect=exchange):
            hidden = torch.randn(1, 7, 16, requires_grad=True)
            copied_hidden = hidden.detach().clone().requires_grad_()
            reference, result = serial(hidden), candidate(copied_hidden)
            torch.testing.assert_close(result, reference)
            (reference.square().sum() + serial.auxiliary_loss).backward()
            (result.square().sum() + candidate.auxiliary_loss).backward()
            torch.testing.assert_close(copied_hidden.grad, hidden.grad)
            for expected, actual in zip(serial.parameters(), candidate.parameters()):
                torch.testing.assert_close(actual.grad, expected.grad)
            torch.testing.assert_close(candidate.tokens_per_expert, serial.tokens_per_expert)

    def test_trunk_and_mtp_training(self):
        """All sparse layers inherit fixed routing and retain finite losses and router gradients."""
        model = JTDeepseekV3ForCausalLM(fixed_config())
        tokens = torch.arange(8).unsqueeze(0)
        losses = model(tokens, (tokens + 1) % 32).loss
        self.assertTrue(all(torch.isfinite(loss) for loss in losses.values()))
        sum(losses.values()).backward()
        modules = [module for module in model.modules() if isinstance(module, JTDeepseekV3MoE)]
        self.assertEqual(len(modules), 2)
        for module in modules:
            self.assertTrue(module.config.fix_router)
            self.assertGreater(module.gate.weight.grad.abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(module.gate.weight.grad).all())
            self.assertLessEqual(int(module.tokens_per_expert.max() - module.tokens_per_expert.min()), 1)
