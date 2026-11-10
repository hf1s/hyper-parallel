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
"""DeepSeek-V3.2 training router: sequence-wise balance loss and expert-bias update."""

import unittest
from unittest.mock import patch

import torch
from transformers import DeepseekV32Config
from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32TopkRouter

from hyper_parallel.models.deepseek_v32.adapter.router import DeepseekV32Router
from tests.common.mark_utils import arg_mark


def _tiny_config() -> DeepseekV32Config:
    """Return a configuration with two expert groups of four experts."""
    return DeepseekV32Config.from_dict({
        "vocab_size": 96, "hidden_size": 48, "intermediate_size": 64, "moe_intermediate_size": 16,
        "num_hidden_layers": 2, "first_k_dense_replace": 1, "num_attention_heads": 4,
        "num_key_value_heads": 4, "n_shared_experts": 1, "n_routed_experts": 8, "n_group": 2,
        "topk_group": 1, "num_experts_per_tok": 2, "kv_lora_rank": 16, "q_lora_rank": 24,
        "qk_rope_head_dim": 8, "v_head_dim": 16, "qk_nope_head_dim": 16, "index_topk": 4,
        "index_head_dim": 16, "index_n_heads": 4, "max_position_embeddings": 64,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
    })


def _router(config, **options):
    """Return a training router over a randomly initialized Transformers router."""
    torch.manual_seed(1)
    source = DeepseekV32TopkRouter(config)
    with torch.no_grad():
        source.weight.normal_(std=0.2)
        source.e_score_correction_bias.normal_(std=0.05)
    return source, DeepseekV32Router.from_router(source, **options)


def _reference_balance_loss(scores, indices, batch_size, top_k):
    """Sequence-wise balance loss, one sample at a time."""
    num_experts = scores.shape[-1]
    sequence_length = scores.shape[0] // batch_size
    losses = []
    for sample in range(batch_size):
        rows = slice(sample * sequence_length, (sample + 1) * sequence_length)
        load = torch.zeros(num_experts)
        for expert in indices[rows].reshape(-1).tolist():
            load[expert] += 1
        load = load * num_experts / (sequence_length * top_k)
        probability = scores[rows] / scores[rows].sum(dim=-1, keepdim=True)
        losses.append((load * probability.mean(dim=0)).sum())
    return torch.stack(losses).mean()


class TestDeepseekV32Router(unittest.TestCase):
    """The training router keeps Transformers routing and adds the training objectives."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_selection_and_balance_loss(self):
        """Feature: DeepSeek-V3.2 training router.

        Description: Route two samples through the Transformers and the training router.
        Expectation: Indices and gate values match; the balance loss and its router gradient
        match the per-sample definition on the biased, group-limited selection.
        """
        config = _tiny_config()
        source, router = _router(config, aux_loss_coeff=0.5)
        hidden = torch.randn(2, 6, config.hidden_size)
        _, expected_weights, expected_indices = source(hidden)
        router.train()
        _, weights, indices = router(hidden)
        torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
        torch.testing.assert_close(weights, expected_weights)

        router.weight.grad = None
        (weights * 0).sum().backward()
        reference_weight = router.weight.detach().clone().requires_grad_()
        flat = hidden.reshape(-1, config.hidden_size)
        scores = torch.nn.functional.linear(flat, reference_weight).sigmoid()  # pylint: disable=not-callable
        reference = _reference_balance_loss(scores, indices, batch_size=2, top_k=config.num_experts_per_tok)
        torch.testing.assert_close(router.last_aux_loss, reference.detach())
        (0.5 * reference).backward()
        torch.testing.assert_close(router.weight.grad, reference_weight.grad)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_expert_bias_update(self):
        """Feature: Loss-free expert-bias update.

        Description: Count routed tokens in training, skip a recomputed forward, then update.
        Expectation: Each bias moves by the rate toward the mean load and the counts reset.
        """
        config = _tiny_config()
        _, router = _router(config, enable_expert_bias=True)
        router.train()
        hidden = torch.randn(1, 5, config.hidden_size)
        _, _, indices = router(hidden)
        with patch("hyper_parallel.models.deepseek_v32.adapter.router.is_recomputing", return_value=True):
            router(hidden)
        counts = torch.bincount(indices.reshape(-1), minlength=config.n_routed_experts)
        torch.testing.assert_close(router.tokens_per_expert, counts)
        before = router.e_score_correction_bias.clone()
        router.update_expert_bias(lr=1e-3)
        expected = before + 1e-3 * torch.sign(counts.float().mean() - counts.float())
        torch.testing.assert_close(router.e_score_correction_bias, expected)
        self.assertEqual(int(router.tokens_per_expert.sum()), 0)


if __name__ == "__main__":
    unittest.main()
