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
"""DeepSeek-V3.2 training objectives: balance loss, expert bias, MTP and checkpoint names."""

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import DeepseekV32Config
from transformers.masking_utils import create_causal_mask
from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32TopkRouter

from hyper_parallel.models.deepseek_v32.modeling_deepseek_v32 import (
    DeepseekV32ForCausalLM,
    DeepseekV32Router,
)
from tests.common.dsa_reference_kernels import reference_dsa_kernels
from tests.common.mark_utils import arg_mark


def _tiny_config(**overrides) -> DeepseekV32Config:
    """Return one dense layer, one MoE layer and one MoE MTP depth."""
    values = {
        "vocab_size": 96, "hidden_size": 48, "intermediate_size": 64, "moe_intermediate_size": 16,
        "num_hidden_layers": 2, "first_k_dense_replace": 1, "num_attention_heads": 4,
        "num_key_value_heads": 4, "n_shared_experts": 1, "n_routed_experts": 8, "n_group": 2,
        "topk_group": 1, "num_experts_per_tok": 2, "kv_lora_rank": 16, "q_lora_rank": 24,
        "qk_rope_head_dim": 8, "v_head_dim": 16, "qk_nope_head_dim": 16, "index_topk": 4,
        "index_head_dim": 16, "index_n_heads": 4, "max_position_embeddings": 64,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0}, "attn_implementation": "sdpa",
        "num_nextn_predict_layers": 1,
    }
    values.update(overrides)
    config = DeepseekV32Config.from_dict(values)
    config.architectures = ["DeepseekV32ForCausalLM"]
    return config


def _router(config, **options):
    """Return a training router over a randomly initialized Transformers router."""
    torch.manual_seed(1)
    source = DeepseekV32TopkRouter(config)
    with torch.no_grad():
        source.weight.normal_(std=0.2)
        source.e_score_correction_bias.normal_(std=0.05)
    return source, DeepseekV32Router.from_router(source, config, **options)


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
        with patch("hyper_parallel.models.deepseek_v32.modeling_deepseek_v32.is_recomputing", return_value=True):
            router(hidden)
        counts = torch.bincount(indices.reshape(-1), minlength=config.n_routed_experts)
        torch.testing.assert_close(router.tokens_per_expert, counts)
        before = router.e_score_correction_bias.clone()
        router.update_expert_bias(lr=1e-3)
        expected = before + 1e-3 * torch.sign(counts.float().mean() - counts.float())
        torch.testing.assert_close(router.e_score_correction_bias, expected)
        self.assertEqual(int(router.tokens_per_expert.sum()), 0)


class TestDeepseekV32MultiTokenPrediction(unittest.TestCase):
    """MTP depth 1 predicts token i + 2 from the trunk state at i and token i + 1."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_mtp_loss_matches_truncated_depth(self):
        """Feature: DeepSeek-V3.2 MTP loss.

        Description: Compare the full-length MTP loss with the depth evaluated on the first
        S - 1 positions, whose last target is the final label.
        Expectation: Equal losses, scaled by 0.3, with the LM loss under ``foundation_loss``.
        """
        torch.manual_seed(2)
        model = DeepseekV32ForCausalLM(_tiny_config()).float().train()
        tokens = torch.randint(1, 96, (2, 13))
        inputs, labels = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
        captured = {}
        hook = model.model.norm.register_forward_pre_hook(lambda _, args: captured.setdefault("hidden", args[0]))
        with torch.no_grad():
            output = model(input_ids=inputs, labels=labels, shift_labels=labels, use_cache=False)
            hook.remove()
            length = inputs.shape[1] - 1
            positions = torch.arange(length).unsqueeze(0)
            embeds = model.model.embed_tokens(inputs[:, 1:])
            mask = create_causal_mask(config=model.config, inputs_embeds=embeds, attention_mask=None,
                                      past_key_values=None, position_ids=positions, allow_is_causal_skip=False)
            hidden = model.mtp.layers[0](
                captured["hidden"][:, :-1], embeds, attention_mask=mask, position_ids=positions,
                position_embeddings=model.model.rotary_emb(embeds, positions),
            )
            logits = model.lm_head(hidden)
            expected = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                         labels[:, 1:].reshape(-1))
        torch.testing.assert_close(output.loss["foundation_loss/mtp"], 0.3 * expected)
        torch.testing.assert_close(output.extra_metrics["mtp_1_loss"], expected)
        self.assertIn("foundation_loss", output.loss)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_released_mtp_layer_round_trips(self):
        """Feature: DeepSeek-V3.2 MTP checkpoint names.

        Description: Save the MTP depth under the released layer names, with per-expert
        weights and its own embedding and head copies, load the training model and save it.
        Expectation: Every MTP parameter is loaded and the copies are not; the saved
        checkpoint has the released architecture and weight names.
        """
        config = _tiny_config()
        torch.manual_seed(3)
        source = DeepseekV32ForCausalLM(config).float()
        layer = config.num_hidden_layers
        state = {}
        for name, tensor in source.state_dict().items():
            name = name.replace("mtp.layers.0.mtp_block.", f"model.layers.{layer}.")
            name = name.replace("mtp.layers.0.post_norm.", f"model.layers.{layer}.shared_head.norm.")
            name = name.replace("mtp.layers.0.", f"model.layers.{layer}.")
            if name.endswith("mlp.experts.gate_up_proj"):
                gate, up = tensor.chunk(2, dim=1)
                for expert in range(tensor.shape[0]):
                    state[name.replace("gate_up_proj", f"{expert}.gate_proj.weight")] = gate[expert].clone()
                    state[name.replace("gate_up_proj", f"{expert}.up_proj.weight")] = up[expert].clone()
            elif name.endswith("mlp.experts.down_proj"):
                for expert in range(tensor.shape[0]):
                    state[name.replace("down_proj", f"{expert}.down_proj.weight")] = tensor[expert].clone()
            else:
                state[name] = tensor.clone().contiguous()
        state[f"model.layers.{layer}.embed_tokens.weight"] = torch.zeros(config.vocab_size, config.hidden_size)
        state[f"model.layers.{layer}.shared_head.head.weight"] = torch.zeros(config.vocab_size, config.hidden_size)
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            save_file(state, f"{directory}/model.safetensors")
            loaded, info = DeepseekV32ForCausalLM.from_pretrained(
                directory, torch_dtype=torch.float32, output_loading_info=True,
            )
            loaded.save_pretrained(f"{directory}/export")
            with open(f"{directory}/export/config.json", encoding="utf-8") as handle:
                architectures = json.load(handle)["architectures"]
            with safe_open(f"{directory}/export/model.safetensors", "pt") as handle:
                exported = set(handle.keys())
        self.assertEqual(info["missing_keys"], set())
        expected = source.state_dict()
        for name, tensor in loaded.state_dict().items():
            if name.startswith("mtp.") or name.startswith("model.embed_tokens"):
                torch.testing.assert_close(tensor, expected[name], msg=name)
        self.assertEqual(architectures, ["DeepseekV32ForCausalLM"])
        copies = {f"model.layers.{layer}.embed_tokens.weight", f"model.layers.{layer}.shared_head.head.weight"}
        self.assertEqual(exported, set(state) - copies)


class TestDeepseekV32IndexerOptions(unittest.TestCase):
    """Indexer rotation and the tensor-parallel KL teacher."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_hadamard_rotation_preserves_scores(self):
        """Feature: Indexer Hadamard rotation.

        Description: Rotate indexer queries and keys.
        Expectation: The Sylvester matrix is orthogonal up to its size and query-key dot
        products are unchanged.
        """
        with reference_dsa_kernels():
            # pylint: disable=import-outside-toplevel,protected-access
            from hyper_parallel.models.deepseek_v32.adapter.conversion.dsa_attention import (
                _hadamard_matrix,
                rotate_activation,
            )

            matrix = _hadamard_matrix(16, torch.float64, torch.device("cpu"))
            torch.testing.assert_close(matrix @ matrix.T, 16 * torch.eye(16, dtype=torch.float64))
            query, key = torch.randn(5, 3, 16, dtype=torch.float64), torch.randn(7, 16, dtype=torch.float64)
            torch.testing.assert_close(rotate_activation(query) @ rotate_activation(key).T, query @ key.T)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_kl_teacher_requires_head_group_under_tensor_parallelism(self):
        """Feature: DSA indexer KL under tensor parallelism.

        Description: Resolve the KL head group for whole and TP-sharded attention heads.
        Expectation: Whole heads need no group; sharded heads use the bound group and
        fail without one.
        """
        with reference_dsa_kernels():
            # pylint: disable=import-outside-toplevel,protected-access
            from hyper_parallel.models.deepseek_v32.adapter.conversion.dsa_attention import (
                DeepseekV32TrainingDSAAttention,
            )

            resolve = DeepseekV32TrainingDSAAttention._kl_head_group
            self.assertIsNone(resolve(SimpleNamespace(num_heads=4, kl_head_group=None)))
            sharded = SimpleNamespace(num_heads=2, kl_head_group=None, _hp_full_head_counts={"num_heads": 4})
            with self.assertRaises(RuntimeError):
                resolve(sharded)
            sharded.kl_head_group = "tp"
            self.assertEqual(resolve(sharded), "tp")


if __name__ == "__main__":
    unittest.main()
