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
from pathlib import Path
from typing import Any
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.models.jt_deepseek_v3.adapter.runtime import jt_optimizer
from hyper_parallel.models.jt_deepseek_v3.configuration_jt_deepseek_v3 import JTDeepseekV3Config
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (
    JTDeepseekV3ForCausalLM, JTDeepseekV3MLAAttention,
)
from hyper_parallel.models.replacement import apply_module_replacements, compile_module_replacements
from hyper_parallel.trainer.config import entries_to_module_replacements
from hyper_parallel.trainer.config.parser import parse_training_args
from tests.common.mark_utils import arg_mark

THRESHOLD = 100.0
RECIPE = Path(__file__).resolve().parents[4] / "examples/training_demo/jt_deepseek_v3/jt_deepseek_v3.yaml"


def replaced_model(dtype: torch.dtype) -> JTDeepseekV3ForCausalLM:
    """Build a CPU-sized JT model whose attention uses the recipe's MLA replacement.

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
    rules = entries_to_module_replacements(parse_training_args([str(RECIPE)]).plan_overrides)
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
    """QK clipping must survive fp32 main-param copy-back and agree across replicas."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_clip_survives_fp32_main_param_copy_back(self):
        """Feature: QK clipping with fp32 main parameters.

        Description: Step the recipe optimizer behind the mixed-precision wrapper while head 0
            exceeds the threshold 4x and head 1 stays below it.
        Expectation: Head 0 is scaled exactly once in the main and the model parameter; head 1 is unchanged.
        """
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                model = replaced_model(dtype)
                builder = jt_optimizer.build_optimizer(
                    model=model, qk_clip_threshold=THRESHOLD,
                    muon_config={"lr": 0.0}, adamw_config={"adamw_lr": 0.0})
                optimizer = Float16OptimizerWithFloat16Params(builder.get_optimizer(), model)
                modules = attention_modules(model)
                originals = []
                for module in modules:
                    module.max_logits_val = torch.tensor([4 * THRESHOLD, THRESHOLD / 2])
                    originals.append((module.q_b_proj.weight.detach().float().clone(),
                                      module.kv_b_proj.weight.detach().float().clone()))
                optimizer.step()
                for module, (query, key_value) in zip(modules, originals):
                    for weight, expected in (
                            (module.q_b_proj.weight, query * row_factors([0.25, 1.0], rope_scaled=True)),
                            (module.kv_b_proj.weight, key_value * row_factors([0.25, 1.0], rope_scaled=False))):
                        self.assertTrue(torch.equal(weight.main_param, expected),
                                        "fp32 main parameter must hold the clipped projection")
                        self.assertTrue(torch.equal(weight, expected.to(dtype)),
                                        "copy-back must not restore the unclipped projection")
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


if __name__ == "__main__":
    unittest.main()
