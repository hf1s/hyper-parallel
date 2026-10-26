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
"""CPU tests of the Xing4.0 optimizer: Muon schema, AdamW decay groups and QK clipping."""

import unittest

import torch

from hyper_parallel.core.optimizer.muon import Muon as MuonOptimizer
from hyper_parallel.models.xing40.adapter.optim import _ns_transform, build_optimizer, clip_qk
from hyper_parallel.models.xing40.modeling_xing40 import (
    Xing40Attention,
    Xing40Config,
    Xing40ForCausalLM,
)
from tests.common.mark_utils import arg_mark

CONFIG = Xing40Config(  # pylint: disable=unexpected-keyword-arg
    vocab_size=64, hidden_size=32, intermediate_size=48, moe_intermediate_size=16, num_hidden_layers=2,
    first_k_dense_replace=1, num_attention_heads=4, num_key_value_heads=4, q_lora_rank=16, kv_lora_rank=8,
    qk_rope_head_dim=4, qk_nope_head_dim=6, v_head_dim=5, n_routed_experts=4, num_experts_per_tok=2,
    n_shared_experts=1, n_group=1, topk_group=1, moe_aux_loss_coeff=1e-4, moe_router_enable_expert_bias=True,
    moe_router_bias_update_rate=1e-3, use_pad_tokens=True, num_residual_streams=4, mhc_sinkhorn_iterations=20,
    mhc_init_gating_factor=0.01, mhc_layernorm_epsilon=1e-6, num_nextn_predict_layers=1, mtp_loss_factor=0.3,
    chunk_loss_size=8)


def _row_ratios(after: torch.Tensor, before: torch.Tensor) -> torch.Tensor:
    """Return the per-row scale applied to a matrix."""
    return (after / before)[:, 0]


class TestQkClip(unittest.TestCase):
    """MuonClip rescaling of the MLA up projections."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_clip_scales_only_heads_above_threshold(self):
        """
        Feature: QK clipping after the optimizer step.
        Description: Head 1 reached 4x the threshold; the other heads stayed below it.
        Expectation: Head 1's query/key no-RoPE rows scale by 1/2 and its query RoPE rows by 1/4; value rows
            and other heads are unchanged; the maxima are cleared.
        """
        torch.manual_seed(0)
        attention = Xing40Attention(CONFIG)
        query, key_value = attention.q_b_proj.weight.detach().clone(), attention.kv_b_proj.weight.detach().clone()
        attention.max_logits = torch.tensor([10.0, 400.0, 99.0, 100.0])
        clip_qk([attention], 100.0)
        nope, rope, value = CONFIG.qk_nope_head_dim, CONFIG.qk_rope_head_dim, CONFIG.v_head_dim
        query_scale = torch.ones(4, nope + rope)
        query_scale[1] = torch.tensor([0.5] * nope + [0.25] * rope)
        key_value_scale = torch.ones(4, nope + value)
        key_value_scale[1, :nope] = 0.5
        torch.testing.assert_close(_row_ratios(attention.q_b_proj.weight, query), query_scale.flatten())
        torch.testing.assert_close(_row_ratios(attention.kv_b_proj.weight, key_value), key_value_scale.flatten())
        self.assertIsNone(attention.max_logits, f"max_logits={attention.max_logits}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_optimizer_step_runs_the_clip(self):
        """
        Feature: Optimizer integration of QK clipping.
        Description: The same training step with and without clipping; every attention module recorded
            maxima where only head 1 exceeds the threshold.
        Expectation: The clipped step equals the unclipped one with head 1's query/key no-RoPE rows scaled
            by 1/2 and its query RoPE rows by 1/4, and it clears the maxima.
        """
        input_ids = torch.randint(0, CONFIG.vocab_size, (1, 8), generator=torch.Generator().manual_seed(1))
        stepped = {}
        for threshold in (0.0, 100.0):
            torch.manual_seed(0)
            model = Xing40ForCausalLM(CONFIG)
            optimizer = build_optimizer(
                model=model, muon_config={"muon_lr": 1e-3}, adamw_config={"adamw_lr": 1e-3},
                qk_clip_threshold=threshold).get_optimizer()
            torch.stack(list(model(input_ids, shift_labels=input_ids.roll(-1, 1)).loss.values())).sum().backward()
            attention = [module for module in model.modules() if isinstance(module, Xing40Attention)]
            for module in attention:
                module.max_logits = torch.tensor([10.0, 400.0, 99.0, 100.0])
            optimizer.step()
            stepped[threshold] = attention
        nope, rope, value = CONFIG.qk_nope_head_dim, CONFIG.qk_rope_head_dim, CONFIG.v_head_dim
        query_scale = torch.ones(4, nope + rope)
        query_scale[1] = torch.tensor([0.5] * nope + [0.25] * rope)
        key_value_scale = torch.ones(4, nope + value)
        key_value_scale[1, :nope] = 0.5
        for clipped, unclipped in zip(stepped[100.0], stepped[0.0]):
            torch.testing.assert_close(clipped.q_b_proj.weight, unclipped.q_b_proj.weight * query_scale.reshape(-1, 1),
                                       msg="case: query rows of the clipped step")
            torch.testing.assert_close(clipped.kv_b_proj.weight,
                                       unclipped.kv_b_proj.weight * key_value_scale.reshape(-1, 1),
                                       msg="case: key/value rows of the clipped step")
            self.assertIsNone(clipped.max_logits, f"uncleared maxima: {clipped.max_logits}")


class TestAdamwDecayGroups(unittest.TestCase):
    """MindFormers weight-decay groups of the AdamW fallback."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_one_dimensional_parameters_are_not_decayed(self):
        """
        Feature: AdamW weight-decay groups.
        Description: Build the optimizer with an AdamW weight decay of 0.1.
        Expectation: Every 1-D parameter (norm weights, mHC scales and shifts) has no decay; the 2-D AdamW
            parameters (embedding, LM head) decay by 0.1.
        """
        model = Xing40ForCausalLM(CONFIG)
        optimizer = build_optimizer(model=model, muon_config={"muon_lr": 1e-3},
                                    adamw_config={"adamw_lr": 1e-3, "adamw_weight_decay": 0.1}).get_optimizer()
        decay = {parameter.model_name: group["weight_decay"] for leaf in optimizer.chained_optimizers
                 if not isinstance(leaf, MuonOptimizer) for group in leaf.param_groups for parameter in group["params"]}
        expected = {name: 0.0 if parameter.ndim == 1 else 0.1
                    for name, parameter in model.named_parameters() if name in decay}
        self.assertEqual(decay, expected)
        for name in ("model.layers.0.attn_hc_pre.branch_alpha", "model.hc_head.hc_base", "model.hc_head.hc_scale"):
            self.assertEqual(decay.get(name), 0.0, f"{name} decay={decay.get(name)}")
        self.assertEqual(decay.get("lm_head.weight"), 0.1, f"lm_head decay={decay.get('lm_head.weight')}")


class TestMuonSchema(unittest.TestCase):
    """MindFormers matrix splits for Newton-Schulz."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_splits_restore_the_original_layout(self):
        """
        Feature: Muon NS input transform.
        Description: Split the fused latent projection into its query/KV/RoPE-key blocks, the MLA up
            projections per head and the fused expert gate/up matrix in halves, in the Transformers
            [E, 2I, H] layout and in the transposed GroupedExperts [E, H, 2I] layout.
        Expectation: Each part has the expected shape and restoring the unmodified parts reproduces the input.
        """
        transform = _ns_transform(CONFIG, frozenset({"model.layers.2.mlp.experts"}))
        heads, nope, rope, value = 4, CONFIG.qk_nope_head_dim, CONFIG.qk_rope_head_dim, CONFIG.v_head_dim
        q_lora, kv_lora = CONFIG.q_lora_rank, CONFIG.kv_lora_rank
        expert = CONFIG.moe_intermediate_size
        cases = {
            "model.layers.0.self_attn.linear_qkv.weight": (
                (q_lora + kv_lora + rope, 32), [(q_lora, 32), (kv_lora, 32), (rope, 32)]),
            "model.layers.0.self_attn.q_b_proj.weight": (
                (heads * (nope + rope), 16), [(heads * nope, 16), (heads * rope, 16)]),
            "model.layers.0.self_attn.kv_b_proj.weight": (
                (heads * (nope + value), 8), [(heads * nope, 8), (heads * value, 8)]),
            "model.layers.1.mlp.experts.gate_up_proj": (
                (4, 2 * expert, 32), [(4, expert, 32), (4, expert, 32)]),
            "model.layers.2.mlp.experts.gate_up_proj": (
                (4, 32, 2 * expert), [(4, 32, expert), (4, 32, expert)]),
        }
        for name, (shape, part_shapes) in cases.items():
            with self.subTest(name=name):
                update = torch.randn(shape)
                split = transform(name, update)
                self.assertEqual([tuple(part.shape) for part in split.tensors], part_shapes)
                restored = torch.empty_like(update)
                split.restore(split.tensors, restored)
                torch.testing.assert_close(restored, update, rtol=0, atol=0)
        query = torch.randn(heads * (nope + rope), 16)
        query_parts = transform("model.layers.0.self_attn.q_b_proj.weight", query).tensors
        per_head = query.view(heads, nope + rope, 16)
        torch.testing.assert_close(query_parts[0], per_head[:, :nope].reshape(-1, 16), rtol=0, atol=0)
        torch.testing.assert_close(query_parts[1], per_head[:, nope:].reshape(-1, 16), rtol=0, atol=0)
        self.assertIsNone(transform("model.layers.0.self_attn.o_proj.weight", torch.randn(8, 8)))


if __name__ == "__main__":
    unittest.main()
