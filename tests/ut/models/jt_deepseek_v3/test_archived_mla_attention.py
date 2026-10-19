# Copyright 2026 Huawei Technologies Co., Ltd.
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
"""Archived MLA attention: product forward parity, archived latent norms and the FP32 half-tree shared key."""
# The Transformers config signature is generated dynamically; keyword checks are false positives here.
# pylint: disable=unexpected-keyword-arg
import copy
from typing import Any
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from hyper_parallel.models.jt_deepseek_v3 import archived_layers
from hyper_parallel.models.jt_deepseek_v3.archived_mla_attention import ArchivedMLAAttention, SharedKeyHalfTree
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (
    JTDeepseekV3Attention, JTDeepseekV3Config, JTDeepseekV3MLAAttention,
)
from tests.common.mark_utils import arg_mark


def rms_norm(values: torch.Tensor, weight: torch.Tensor, *, epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU stand-in for ``torch_npu.npu_rms_norm`` with the product RMSNorm's FP32 formula."""
    rstd = torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + epsilon)
    return values * rstd * weight, rstd


def eager_attention(module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                    attention_mask: Any, *, dropout: float, scaling: float, **kwargs: Any) -> tuple[torch.Tensor, None]:
    """CPU causal attention returning the fused kernel's ``[batch, sequence, heads, dim]`` layout."""
    del module, attention_mask, kwargs
    output = F.scaled_dot_product_attention(  # pylint: disable=not-callable
        query, key, value, dropout_p=dropout, is_causal=True, scale=scaling)
    return output.transpose(1, 2), None


def source_attention(seed: int) -> JTDeepseekV3Attention:
    """Build a materialized four-head source MLA whose latent norm weights differ from their initial ones."""
    torch.manual_seed(seed)
    config = JTDeepseekV3Config(
        vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        q_lora_rank=8, kv_lora_rank=8, qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=4,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0})
    config.rope_interleave = True
    source = JTDeepseekV3Attention(config, 0)
    with torch.no_grad():
        for name in ("q_a_layernorm.weight", "kv_a_layernorm.weight"):
            source.get_parameter(name).uniform_(0.5, 1.5)
    return source


class TestSharedKeyHalfTree(unittest.TestCase):
    """The shared rope key expands over heads and reduces their gradients by an FP32 half tree."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_shared_key_expands_heads_not_tokens(self):
        """The current MLA projection contract is token-major, including its backward."""
        values = torch.arange(30, dtype=torch.float32).reshape(2, 5, 1, 3).requires_grad_()
        actual = SharedKeyHalfTree.apply(values, 8)
        expected = values.expand(2, 5, 8, 3)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        weights = torch.arange(actual.numel(), dtype=torch.float32).reshape_as(actual)
        (actual * weights).sum().backward()
        torch.testing.assert_close(values.grad, weights.sum(2, keepdim=True), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'singleton shared head'):
            SharedKeyHalfTree.apply(torch.zeros(1, 1, 5, 3), 8)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_shared_key_sums_head_gradients_by_fp32_half_tree(self):
        """Head j meets head j + n / 2 first, in FP32, and the sum is rounded once to the key dtype."""
        # (1e8 - 1e8) + (1 + 1) = 2, whereas a left-to-right FP32 sum loses each 1 against 1e8 (spacing 8) and
        # gives 1, and pairing neighbours gives 0.
        values = torch.zeros(1, 1, 1, 1, requires_grad=True)
        SharedKeyHalfTree.apply(values, 4).backward(torch.tensor([1e8, 1.0, -1e8, 1.0]).reshape(1, 1, 4, 1))
        self.assertEqual(values.grad.item(), 2.0)
        # BF16 heads: (1 + 2**-8) + (2**-8 + 0) = 1 + 2**-7 in FP32, whereas BF16 partial sums tie 1 + 2**-8
        # back to 1 and then 1 + 2**-8 back to 1 again.
        values = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16, requires_grad=True)
        gradient = torch.tensor([1.0, 2.0 ** -8, 2.0 ** -8, 0.0], dtype=torch.bfloat16).reshape(1, 1, 4, 1)
        SharedKeyHalfTree.apply(values, 4).backward(gradient)
        self.assertEqual(values.grad.item(), 1.0 + 2.0 ** -7)


class TestArchivedMLAAttention(unittest.TestCase):
    """The archived replacement keeps the product forward values and changes only the archived arithmetic."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_forward_matches_product_mla_bitwise_in_fp32(self):
        """From the same materialized source, query, key, value and output equal the product MLA bitwise."""
        source = source_attention(5)
        product = JTDeepseekV3MLAAttention(module=copy.deepcopy(source))
        archived = ArchivedMLAAttention(module=copy.deepcopy(source))
        hidden = torch.randn(2, 3, 16)
        angle = torch.randn(2, 3, 4)
        position_embeddings = (angle.cos(), angle.sin())
        results = []
        for attention in (product, archived):
            attention.attention_interface = eager_attention
            with patch.object(archived_layers.torch_npu, "npu_rms_norm", rms_norm):
                projected = attention._project_attention_inputs(hidden, position_embeddings, None)
                output, _ = attention(hidden, position_embeddings=position_embeddings)
            results.append((*projected, output))
        for name, expected, actual in zip(("query", "key", "value", "output"), *results):
            with self.subTest(name):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_latent_norms_round_fp32_rms_with_source_weights_once(self):
        """BF16 latents are the FP32 RMSNorm with the materialized source weights, rounded once to BF16."""
        source = source_attention(7)
        archived = ArchivedMLAAttention(module=copy.deepcopy(source)).to(torch.bfloat16)
        product = JTDeepseekV3MLAAttention(module=copy.deepcopy(source)).to(torch.bfloat16)
        hidden = torch.randn(2, 8, 16, dtype=torch.bfloat16)
        with patch.object(archived_layers.torch_npu, "npu_rms_norm", rms_norm):
            query_latent, kv_latent, _ = archived.project_latent_inputs(hidden)
        query_local, kv_local, _ = archived.linear_qkv(hidden).split((8, 8, 4), dim=-1)
        weights = source.state_dict()
        eps = source.config.rms_norm_eps
        cases = (("q_a_layernorm", query_latent, query_local), ("kv_a_layernorm", kv_latent, kv_local))
        for name, latent, local in cases:
            with self.subTest(name):
                weight = weights[f"{name}.weight"].bfloat16().double()
                expected = rms_norm(local.double(), weight, epsilon=eps)[0].to(torch.bfloat16)
                # The product norm rounds the normalized value to BF16 before the weight multiply.
                self.assertFalse(torch.equal(product.get_submodule(name)(local), expected))
                torch.testing.assert_close(latent, expected, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_shared_rope_key_gradient_reaches_projection_as_half_tree_sum(self):
        """Per-head rope-key gradients [1e8, 1, -1e8, 1] reach the latent projection as their half-tree sum 2."""
        attention = ArchivedMLAAttention(module=source_attention(3))
        hidden = torch.ones(1, 1, 16)
        no_rotation = (torch.ones(1, 1, 4), torch.zeros(1, 1, 4))
        with patch.object(archived_layers.torch_npu, "npu_rms_norm", rms_norm):
            _, key, _ = attention._project_attention_inputs(hidden, no_rotation, None)
        gradient = torch.zeros_like(key)
        gradient[..., 4:] = torch.tensor([1e8, 1.0, -1e8, 1.0]).reshape(1, 4, 1, 1)
        key.backward(gradient)
        # The rope rows are the last four of the fused latent projection; a plain head sum would give 1.
        torch.testing.assert_close(attention.linear_qkv.weight.grad[-4:], torch.full((4, 16), 2.0), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
