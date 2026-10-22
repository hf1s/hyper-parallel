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
"""QK clipping applied by the JT optimizer post-update hook."""
# The Transformers config signature is generated dynamically; keyword checks are false positives here.
# pylint: disable=unexpected-keyword-arg
from collections.abc import Callable
from typing import Any, Optional
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.models.jt_deepseek_v3.adapter.runtime import jt_optimizer
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (
    JTDeepseekV3Config, JTDeepseekV3ForCausalLM, JTDeepseekV3MLAAttention, JTDeepseekV3MoE,
)
from hyper_parallel.models.replacement import apply_module_replacements, compile_module_replacements
from hyper_parallel.trainer.config import entries_to_module_replacements
from tests.common.jt_config import jt_test_config
from tests.common.mark_utils import arg_mark

THRESHOLD = 100.0


def replaced_model(dtype: torch.dtype) -> JTDeepseekV3ForCausalLM:
    """Build a CPU-sized JT model whose attention uses the MLA replacement.

    Two heads with 4 nope, 4 rope and 4 value channels give 8 projection rows per head.
    """
    torch.manual_seed(7)
    config = JTDeepseekV3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, moe_intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2, n_group=1, topk_group=1,
        q_lora_rank=8, kv_lora_rank=8, qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=4,
        mlp_layer_types=["dense", "sparse"], max_position_embeddings=32, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        architectures=["JTDeepseekV3ForCausalLM"], num_nextn_predict_layers=1, use_pad_tokens=True,
        norm_topk_prob=True, routed_scaling_factor=1.0, moe_aux_loss_coeff=0.01, mtp_loss_factor=0.3,
        moe_router_enable_expert_bias=False)
    config.rope_interleave = True
    model = JTDeepseekV3ForCausalLM(config)
    rules = entries_to_module_replacements(jt_test_config().plan_overrides)
    apply_module_replacements(model, compile_module_replacements(model, rules), weights_mapping=[])
    return model.to(dtype)


def attention_modules(model: torch.nn.Module) -> list[JTDeepseekV3MLAAttention]:
    """Return the replaced trunk and MTP attention modules in model order."""
    return [module for module in model.modules() if isinstance(module, JTDeepseekV3MLAAttention)]


def row_factors(head_scales: list[float], rope_scaled: bool) -> torch.Tensor:
    """Expected per-row multipliers of a q_b_proj (``rope_scaled``) or kv_b_proj weight.

    QK clipping scales a head's query and key nope rows by sqrt(scale) and its query rope
    rows by scale; key value rows are unchanged.
    """
    rows = []
    for scale in head_scales:
        rows += [scale ** 0.5] * 4 + [scale if rope_scaled else 1.0] * 4
    return torch.tensor(rows)[:, None]


def two_rank_all_reduce(peer: torch.Tensor) -> Callable[..., None]:
    """Emulate an all-reduce whose only other rank contributes ``peer``."""
    def all_reduce(tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM, group: Any = None) -> None:
        """Reduce ``tensor`` in place with the peer contribution under ``op``."""
        del group
        tensor.copy_(torch.maximum(tensor, peer) if op == dist.ReduceOp.MAX else tensor + peer)
    return all_reduce


class TestQKClip(unittest.TestCase):
    """QK clipping applied after the optimizer step must agree across replicas."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_optimizer_step_clips_projections(self):
        """Feature: QK clipping in the optimizer post-update hook.

        Description: Step the recipe optimizer behind the mixed-precision wrapper with FP32 parameters
            while head 0 exceeds the threshold 4x and head 1 stays below it.
        Expectation: Head 0 is scaled exactly once; head 1 is unchanged.
        """
        model = replaced_model(torch.float32)
        builder = jt_optimizer.build_optimizer(
            model=model, qk_clip_threshold=THRESHOLD,
            muon_config={"lr": 0.0}, adamw_config={"adamw_lr": 0.0})
        optimizer = Float16OptimizerWithFloat16Params(builder.get_optimizer(), model)
        modules = attention_modules(model)
        originals = []
        for module in modules:
            module.max_logits_val = torch.tensor([4 * THRESHOLD, THRESHOLD / 2])
            originals.append((module.q_b_proj.weight.detach().clone(), module.kv_b_proj.weight.detach().clone()))
        optimizer.step()
        for module, (query, key_value) in zip(modules, originals):
            self.assertTrue(torch.equal(module.q_b_proj.weight, query * row_factors([0.25, 1.0], rope_scaled=True)))
            self.assertTrue(torch.equal(module.kv_b_proj.weight,
                                        key_value * row_factors([0.25, 1.0], rope_scaled=False)))
            self.assertTrue(torch.equal(module.max_logits_val, torch.zeros(2)))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_replicas_clip_with_shared_maximum(self):
        """Feature: QK clipping across data-parallel replicas.

        Description: Two replicas of the same heads see different heads exceed the threshold;
            each clips under an all-reduce emulating the other replica.
        Expectation: Both replicas clip both heads by the shared maximum and stay identical.
        """
        replicas = [replaced_model(torch.float32), replaced_model(torch.float32)]
        originals = [(module.q_b_proj.weight.detach().clone(), module.kv_b_proj.weight.detach().clone())
                     for module in attention_modules(replicas[0])]
        local_maxima = ([4 * THRESHOLD, THRESHOLD / 2], [THRESHOLD / 2, 4 * THRESHOLD])
        contributions = []
        for model, maxima in zip(replicas, local_maxima):
            model.qk_clip_group = object()
            for module in attention_modules(model):
                module.max_logits_val = torch.tensor(maxima)
            contributions.append(torch.cat([module.max_logits_val for module in attention_modules(model)]))
        for model, peer in zip(replicas, reversed(contributions)):
            with patch.object(dist, "all_reduce", side_effect=two_rank_all_reduce(peer)):
                jt_optimizer.clip_qk(model, THRESHOLD)
        for replica in replicas:
            for module, (query, key_value) in zip(attention_modules(replica), originals):
                self.assertTrue(torch.equal(module.q_b_proj.weight,
                                            query * row_factors([0.25, 0.25], rope_scaled=True)),
                                "every replica must clip q_b_proj with the shared maximum")
                self.assertTrue(torch.equal(module.kv_b_proj.weight,
                                            key_value * row_factors([0.25, 0.25], rope_scaled=False)),
                                "every replica must clip kv_b_proj with the shared maximum")


class TestAdamWOptimizer(unittest.TestCase):
    """Tests for the AdamW-only JT post-update hook."""

    @staticmethod
    def _embedding_param(model: JTDeepseekV3ForCausalLM) -> torch.nn.Parameter:
        """Return the token embedding parameter."""
        return next(p for n, p in model.named_parameters() if n.endswith("embed_tokens.weight"))

    def _setup(self, *, enable_bias: Optional[bool], rate: float = 0.1,
               counts: Optional[tuple[tuple[int, ...], ...]] = None
               ) -> tuple[JTDeepseekV3ForCausalLM, list[JTDeepseekV3MoE],
                          Float16OptimizerWithFloat16Params, torch.nn.Parameter, torch.Tensor]:
        """Build a CPU JT model, MoE state, and wrapped AdamW optimizer."""
        model = replaced_model(torch.float32)
        config = model.config
        if enable_bias is None:
            config.__dict__.pop("moe_router_enable_expert_bias", None)
        else:
            config.moe_router_enable_expert_bias = enable_bias
        config.moe_router_bias_update_rate = rate
        model.expert_load_group = None
        modules = [m for m in model.modules() if isinstance(m, JTDeepseekV3MoE)]
        for module in modules:
            module.sequence_partition_group = None
        if counts is not None:
            for module, values in zip(modules, counts):
                module.tokens_per_expert = torch.tensor(values, dtype=torch.float32)
                module.gate.e_score_correction_bias.zero_()
        builder = jt_optimizer.build_adamw_optimizer(model=model, adamw_config={
            "adamw_lr": 0.01, "adamw_betas": (0.9, 0.95),
            "adamw_eps": 1e-8, "adamw_weight_decay": 0.0})
        optimizer = Float16OptimizerWithFloat16Params(builder.get_optimizer(), model)
        embed = self._embedding_param(model)
        return model, modules, optimizer, embed, embed.detach().clone()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_bias_enabled_matches_torch_adamw(self):
        """Feature: enabled AdamW hook updates biases without QK clipping."""
        _, modules, optimizer, embed, initial = self._setup(
            enable_bias=True, counts=((1, 2, 3, 4), (4, 3, 2, 1)))
        self.assertEqual(len(modules), 2)
        with patch.object(jt_optimizer, "clip_qk", side_effect=AssertionError("clip_qk")), \
             patch.object(jt_optimizer, "sync_and_update_expert_bias",
                          wraps=jt_optimizer.sync_and_update_expert_bias) as sync_spy:
            embed.grad = torch.ones_like(embed)
            optimizer.step()
        self.assertEqual(sync_spy.call_count, 2)
        for module in modules:
            self.assertEqual(sum(c.args[0] is module for c in sync_spy.call_args_list), 1)
            self.assertIsNone(module.tokens_per_expert)
        self.assertTrue(torch.equal(modules[0].gate.e_score_correction_bias, torch.tensor([.1, .1, -.1, -.1])))
        self.assertTrue(torch.equal(modules[1].gate.e_score_correction_bias, torch.tensor([-.1, -.1, .1, .1])))
        ref = initial.clone().requires_grad_(True)
        ref_opt = torch.optim.AdamW([ref], lr=.01, betas=(.9, .95), eps=1e-8, weight_decay=0.)
        ref.grad = torch.ones_like(ref)
        ref_opt.step()
        torch.testing.assert_close(embed.detach(), ref.detach(), atol=2e-7, rtol=1e-6)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_bias_disabled_or_missing_preserves_counts(self):
        """Feature: disabled or missing config flag skips sync and clipping."""
        for case, enable_bias in (("disabled", False), ("missing", None)):
            with self.subTest(case=case):
                _, modules, optimizer, embed, initial = self._setup(
                    enable_bias=enable_bias, counts=((1, 2, 3, 4), (4, 3, 2, 1)))
                originals = [(m.gate.e_score_correction_bias.clone(), m.tokens_per_expert.clone()) for m in modules]
                with patch.object(jt_optimizer, "sync_and_update_expert_bias",
                                  side_effect=AssertionError("sync")), \
                     patch.object(jt_optimizer, "clip_qk", side_effect=AssertionError("clip")):
                    embed.grad = torch.ones_like(embed)
                    optimizer.step()
                self.assertFalse(torch.equal(embed.detach(), initial))
                for module, (bias, counts) in zip(modules, originals):
                    self.assertTrue(torch.equal(module.gate.e_score_correction_bias, bias))
                    self.assertTrue(torch.equal(module.tokens_per_expert, counts))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_group_passthrough_updates_real_bias(self):
        """Feature: AdamW hook forwards MoE groups and updates bias locally."""
        model, modules, optimizer, embed, _ = self._setup(
            enable_bias=True, counts=((1, 2, 3, 4), (4, 3, 2, 1)))
        self.assertEqual(len(modules), 2)
        model.expert_load_group = object()
        for module in modules:
            module.sequence_partition_group = object()
        calls: list[tuple[JTDeepseekV3MoE, float, Optional[object], Optional[object]]] = []

        def fake_sync(module: JTDeepseekV3MoE, lr: float, tp_group: Optional[object] = None,
                      dp_group: Optional[object] = None) -> None:
            """Record kwargs and run only the local expert-bias update."""
            calls.append((module, lr, tp_group, dp_group))
            module.update_expert_bias(lr)

        with patch.object(jt_optimizer, "sync_and_update_expert_bias", side_effect=fake_sync), \
             patch.object(jt_optimizer, "clip_qk", side_effect=AssertionError("clip")):
            embed.grad = torch.ones_like(embed)
            optimizer.step()
        self.assertEqual(len(calls), 2)
        for module in modules:
            matches = [call for call in calls if call[0] is module]
            self.assertEqual(len(matches), 1)
            _, lr, tp_group, dp_group = matches[0]
            self.assertEqual(lr, .1)
            self.assertIs(tp_group, module.sequence_partition_group)
            self.assertIs(dp_group, model.expert_load_group)
            self.assertIsNone(module.tokens_per_expert)
        self.assertTrue(torch.equal(modules[0].gate.e_score_correction_bias, torch.tensor([.1, .1, -.1, -.1])))
        self.assertTrue(torch.equal(modules[1].gate.e_score_correction_bias, torch.tensor([-.1, -.1, .1, .1])))


if __name__ == "__main__":
    unittest.main()
