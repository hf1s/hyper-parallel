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
"""DeepSeek-V3.2 adapter contracts: TP roles, recipe topology and DSA math."""

import copy
import fnmatch
import unittest
from pathlib import Path

import torch
from transformers import DeepseekV32Config, DeepseekV32ForCausalLM

from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.recipe_spec import TP
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.models.deepseek_v32.adapter.conversion.module_replacement import (
    replace_deepseek_v32_dsa_attention,
)
from hyper_parallel.models.deepseek_v32.adapter.data.runtime import DeepseekV32PackedSequence
from tests.common.dsa_reference_kernels import reference_dsa_kernels
from tests.common.mark_utils import arg_mark
from tests.ut.auto_models.distributed.conftest import FakeDeviceMesh

_RECIPE = Path(__file__).resolve().parents[5] / "hyper_parallel/models/deepseek_v32/recipes/train.yaml"


def _tiny_config(**overrides) -> DeepseekV32Config:
    """Return a small DeepSeek-V3.2 configuration with one dense and two MoE layers."""
    values = {
        "vocab_size": 96, "hidden_size": 48, "intermediate_size": 64, "moe_intermediate_size": 16,
        "num_hidden_layers": 3, "first_k_dense_replace": 1, "num_attention_heads": 4,
        "num_key_value_heads": 4, "n_shared_experts": 1, "n_routed_experts": 8, "n_group": 2,
        "topk_group": 1, "num_experts_per_tok": 2, "kv_lora_rank": 16, "q_lora_rank": 24,
        "qk_rope_head_dim": 8, "v_head_dim": 16, "qk_nope_head_dim": 16, "index_topk": 4,
        "index_head_dim": 16, "index_n_heads": 4, "max_position_embeddings": 64,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0}, "attn_implementation": "sdpa",
    }
    values.update(overrides)
    config = DeepseekV32Config.from_dict(values)
    config.architectures = ["DeepseekV32ForCausalLM"]
    return config


def _attention_inputs(model, batch_size=2, seq_len=12, seed=3):
    """Return hidden states, rotary embeddings and position IDs for layer 0."""
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(batch_size, seq_len, model.config.hidden_size, generator=generator)
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    return hidden, model.model.rotary_emb(hidden, positions), positions


def _dsa_from(attention):
    """Replace one Transformers attention and copy its fused latent projection."""
    replacement = replace_deepseek_v32_dsa_attention(
        module=copy.deepcopy(attention), module_fqn="model.layers.0.self_attn", context={}
    )
    with torch.no_grad():
        replacement.linear_qkv.weight.copy_(
            torch.cat([attention.q_a_proj.weight, attention.kv_a_proj_with_mqa.weight])
        )
    return replacement


class TestDeepseekV32TensorParallelRoles(unittest.TestCase):
    """The planner must own every DeepSeek-V3.2 attention parameter under TP."""

    @staticmethod
    def _attention_placements(model):
        """Plan TP2 and return the TP placement of every layer-0 attention parameter."""
        plan = ShardingPlanner().plan(model, FakeDeviceMesh((2,), ("tp",)), tp_size=2, sequence_parallel=True)
        spec = plan.modules["model.layers.0.self_attn"]
        return {name: placement[TP] for name, placement in spec.params.items()}

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_reference_attention_roles_cover_mla_and_indexer(self):
        """Feature: DeepSeek-V3.2 TP roles on the Transformers attention.

        Description: Plan TP2 for a dense-only model without any replacement.
        Expectation: Latent and indexer projections replicate and head projections shard.
        """
        with torch.device("meta"):
            model = DeepseekV32ForCausalLM(_tiny_config(num_hidden_layers=2, first_k_dense_replace=2))
        placements = self._attention_placements(model)
        expected = {
            "q_a_proj.weight": Replicate(), "kv_a_proj_with_mqa.weight": Replicate(),
            "q_b_proj.weight": Shard(0), "kv_b_proj.weight": Shard(0), "o_proj.weight": Shard(1),
            "indexer.wq_b.weight": Replicate(), "indexer.wk.weight": Replicate(),
            "indexer.weights_proj.weight": Replicate(),
        }
        self.assertEqual(placements, expected)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dsa_replacement_keeps_fused_latent_projection_replicated(self):
        """Feature: DeepSeek-V3.2 TP roles on the DSA replacement.

        Description: Plan TP2 after fusing q_a_proj and kv_a_proj_with_mqa into linear_qkv.
        Expectation: linear_qkv replicates while head-major projections shard.
        """
        with reference_dsa_kernels():
            with torch.device("meta"):
                model = DeepseekV32ForCausalLM(_tiny_config(num_hidden_layers=2, first_k_dense_replace=2))
                for layer in model.model.layers:
                    layer.self_attn = replace_deepseek_v32_dsa_attention(
                        module=layer.self_attn, module_fqn="", context={}
                    )
            placements = self._attention_placements(model)
        self.assertEqual(placements["linear_qkv.weight"], Replicate())
        self.assertEqual(placements["q_b_proj.weight"], Shard(0))
        self.assertEqual(placements["kv_b_proj.weight"], Shard(0))
        self.assertEqual(placements["o_proj.weight"], Shard(1))
        self.assertNotIn("q_a_proj.weight", placements)


class TestDeepseekV32RecipeTopology(unittest.TestCase):
    """Recipe parallel entries must target exactly the intended modules."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_ep_entries_target_exactly_the_moe_layers(self):
        """Feature: DeepSeek-V3.2 EP plan overrides.

        Description: Activate the recipe at EP>1 and match its MLP globs against the released depth.
        Expectation: Only MoE layers receive the EP compute; CP and EP stay off at size one.
        """
        released = DeepseekV32Config()
        mlps = [f"model.layers.{index}.mlp" for index in range(released.num_hidden_layers)]
        sparse = [name for name, kind in zip(mlps, released.mlp_layer_types) if kind == "sparse"]
        with reference_dsa_kernels():
            from hyper_parallel.trainer.config import (  # pylint: disable=import-outside-toplevel
                entries_to_plan_overrides,
            )
            from hyper_parallel.trainer.config.parser import (  # pylint: disable=import-outside-toplevel
                parse_training_args,
            )
            entries = parse_training_args([str(_RECIPE)]).plan_overrides
            self.assertEqual(entries_to_plan_overrides(entries), {})
            active = entries_to_plan_overrides(entries, cp_size=2, ep_size=2)
            ep_globs = [key for key, spec in active.items() if spec.local_compute_fn is not None]
            matched = [name for name in mlps if any(fnmatch.fnmatchcase(name, glob) for glob in ep_globs)]
            self.assertEqual(matched, sparse)
            self.assertIsNotNone(active["*.self_attn"].inner_wrapper)


class TestDeepseekV32DSAAttention(unittest.TestCase):
    """The DSA replacement must reproduce Transformers DeepSeek-V3.2 attention."""

    def _assert_matches_reference(self, attention, dsa, hidden, rope, positions, reference_mask, **kwargs):
        """Compare outputs, input gradients and mapped parameter gradients in FP32."""
        reference_input = hidden.clone().requires_grad_(True)
        expected, _ = attention(reference_input, position_embeddings=rope, attention_mask=reference_mask)
        expected.square().sum().backward()
        candidate_input = hidden.clone().requires_grad_(True)
        actual, _ = dsa(candidate_input, position_embeddings=rope, position_ids=positions, **kwargs)
        actual.square().sum().backward()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(candidate_input.grad, reference_input.grad, rtol=1e-4, atol=1e-6)
        fused_grad = torch.cat([attention.q_a_proj.weight.grad, attention.kv_a_proj_with_mqa.weight.grad])
        torch.testing.assert_close(dsa.linear_qkv.weight.grad, fused_grad, rtol=1e-4, atol=1e-6)
        for name in ("q_b_proj", "kv_b_proj", "o_proj", "q_a_layernorm", "kv_a_layernorm"):
            torch.testing.assert_close(
                getattr(dsa, name).weight.grad, getattr(attention, name).weight.grad, rtol=1e-4, atol=1e-6
            )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_sparse_and_dense_selection_match_transformers(self):
        """Feature: DSA replacement parity.

        Description: Run sparse (top-k below length) and dense selection against the Transformers module.
        Expectation: Outputs and every attention gradient match in FP32.
        """
        for index_topk in (4, 64):
            with self.subTest(index_topk=index_topk), reference_dsa_kernels():
                torch.manual_seed(0)
                model = DeepseekV32ForCausalLM(_tiny_config(index_topk=index_topk))
                attention = model.model.layers[0].self_attn
                dsa = _dsa_from(attention)
                hidden, rope, positions = _attention_inputs(model)
                causal = torch.ones(12, 12, dtype=torch.bool).tril().expand(2, 1, 12, 12)
                self._assert_matches_reference(attention, dsa, hidden, rope, positions, causal)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_packed_documents_match_block_diagonal_attention(self):
        """Feature: DSA packed-document boundaries.

        Description: Pass global cu_seq_lens through the runtime adapter for documents crossing rows.
        Expectation: DSA equals Transformers attention under the matching block-diagonal mask.
        """
        boundaries = torch.tensor([0, 5, 12, 15, 24], dtype=torch.int32)
        runtime = DeepseekV32PackedSequence().build_runtime_inputs(
            batch={"cu_seq_lens": boundaries}, context=None
        )
        mask = torch.zeros(2, 1, 12, 12, dtype=torch.bool)
        for start, end in zip(boundaries[:-1].tolist(), boundaries[1:].tolist()):
            row, first = divmod(start, 12)
            last = end - row * 12
            mask[row, 0, first:last, first:last] = torch.ones(last - first, last - first, dtype=torch.bool).tril()
        with reference_dsa_kernels():
            torch.manual_seed(1)
            model = DeepseekV32ForCausalLM(_tiny_config(index_topk=3))
            attention = model.model.layers[0].self_attn
            dsa = _dsa_from(attention)
            hidden, rope, positions = _attention_inputs(model)
            self._assert_matches_reference(attention, dsa, hidden, rope, positions, mask, **runtime)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_indexer_loss_rejects_tensor_parallelism(self):
        """Feature: DSA indexer KL loss ownership.

        Description: Request the trainable-indexer loss with and without TP.
        Expectation: TP raises before construction; TP1 builds the replacement.
        """
        with reference_dsa_kernels():
            attention = DeepseekV32ForCausalLM(_tiny_config()).model.layers[0].self_attn
            attention.config.dsa_loss_coeff = 0.01
            with self.assertRaisesRegex(ValueError, "tp_size=1"):
                replace_deepseek_v32_dsa_attention(module=attention, module_fqn="layer", context={"tp": True})
            replacement = replace_deepseek_v32_dsa_attention(module=attention, module_fqn="layer", context={})
            self.assertEqual(replacement.dsa_loss_coeff, 0.01)


if __name__ == "__main__":
    unittest.main()
