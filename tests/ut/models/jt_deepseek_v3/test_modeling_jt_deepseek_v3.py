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
"""Complete standalone model construction and optional acceleration boundaries."""
# pylint: disable=protected-access

from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (
    JTDeepseekV3Config,
    JTDeepseekV3ForCausalLM, JTDeepseekV3Decoder, JTDeepseekV3MoE,
    JTDeepseekV3Attention, JTDeepseekV3MLAAttention, JTDeepseekV3RotaryEmbedding,
)
from hyper_parallel.components.modules.mtp import DeepseekV3MTPExecution, MultiTokenPredictionLayer
from hyper_parallel.data.batching import TextParallelBatch
from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.distributed.activation_checkpoint import _apply_activation_checkpointing
from hyper_parallel.models.build_options import FSDP2Config
from hyper_parallel.models.replacement import compile_module_replacements, apply_module_replacements
from hyper_parallel.models.jt_deepseek_v3.adapter.jt_builder import _load_reference_state
from hyper_parallel.models.jt_deepseek_v3.adapter.distributed.fsdp import JTFSDP2Manager
from hyper_parallel.models.registry import get_model_adapter
from hyper_parallel.trainer.config import entries_to_module_replacements
from hyper_parallel.trainer.config.parser import parse_training_args
from tests.common.mark_utils import arg_mark


def small_config() -> JTDeepseekV3Config:
    """Build a CPU-sized fixture without changing the production validation recipe."""
    config = JTDeepseekV3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, moe_intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2, n_group=1, topk_group=1,
        q_lora_rank=8, kv_lora_rank=8, qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=4,
        mlp_layer_types=["dense", "sparse"], max_position_embeddings=32, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        architectures=["JTDeepseekV3ForCausalLM"],
        num_nextn_predict_layers=1, use_pad_tokens=True, norm_topk_prob=True,
        routed_scaling_factor=1.0, moe_aux_loss_coeff=0.01, mtp_loss_factor=0.3)
    config.rope_interleave = True
    return config


class TestCompleteModel(unittest.TestCase):
    """Exercise model semantics without a builder, EP adapter or replacement pass."""

    def test_moe_combine_preserves_probability_precision_and_gradients(self):
        """Feature: Expert aggregation precision.

        Description: Combine BF16 expert values with non-BF16 routing probabilities.
        Expectation: FP32 products are summed before one BF16 cast; both inputs receive gradients.
        """
        moe = JTDeepseekV3MoE(small_config())
        values = torch.tensor([[[1.0], [2.0]]], dtype=torch.bfloat16, requires_grad=True)
        probabilities = torch.tensor([[0.501, 0.499]], requires_grad=True)
        actual = moe._combine_experts(values, probabilities)
        torch.testing.assert_close(actual, torch.tensor([[1.499]], dtype=torch.bfloat16), rtol=0, atol=0)
        actual.float().sum().backward()
        torch.testing.assert_close(probabilities.grad, torch.tensor([[1.0, 2.0]]), rtol=0, atol=0)
        torch.testing.assert_close(values.grad.squeeze(-1), probabilities.detach().to(torch.bfloat16), rtol=0, atol=0)

    def test_moe_residual_has_only_one_rounding_boundary(self):
        """Feature: MoE residual accumulation.

        Description: Sum small shared/routed branches with a BF16 residual.
        Expectation: A representable small branch is not lost to an intermediate BF16 cast.
        """
        config = small_config()
        config.use_pad_tokens = False
        moe = JTDeepseekV3MoE(config)
        hidden = torch.zeros(1, 1, 16, dtype=torch.bfloat16)
        routed = torch.full_like(hidden, 1.0, requires_grad=True)
        shared = torch.full_like(hidden, 0.003, requires_grad=True)
        residual = torch.full_like(hidden, -1.0, requires_grad=True)
        moe.ep_compute = lambda _hidden: routed
        with patch.object(moe.shared_experts, "forward", return_value=shared):
            actual = moe(hidden, residual=residual)
        torch.testing.assert_close(actual, shared, rtol=0, atol=0)
        actual.float().sum().backward()
        for value in (routed, shared, residual):
            torch.testing.assert_close(value.grad, torch.ones_like(value), rtol=0, atol=0)

    def test_partitioned_combine_validates_shape_and_preserves_gradients(self):
        """Feature: Reference reduction partitions.

        Description: Split token-expert work into ranges ending inside a token.
        Expectation: Local partial sums are merged without detaching values or probabilities.
        """
        config = small_config()
        config.moe_combine_num_partitions = 4
        moe = JTDeepseekV3MoE(config)
        values = torch.arange(42, dtype=torch.float32).reshape(7, 6, 1).requires_grad_()
        probabilities = torch.full((7, 6), 0.25, requires_grad=True)
        actual = moe._combine_experts(values, probabilities)
        expected = values.detach().sum(1) / 4
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.sum().backward()
        torch.testing.assert_close(values.grad, torch.full_like(values, 0.25), rtol=0, atol=0)
        torch.testing.assert_close(probabilities.grad, values.detach().squeeze(-1), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "token count"):
            moe._combine_experts(values[:1], probabilities[:1])
        for invalid in (0, True, 1.5):
            config.moe_combine_num_partitions = invalid
            with self.assertRaisesRegex(ValueError, "positive integer"):
                JTDeepseekV3MoE(config)

    def test_grouped_combine_preserves_rounding_and_gradients(self):
        """Feature: Grouped reference reduction.

        Description: Sum cancellation-sensitive products in two groups of three.
        Expectation: Group rounding is retained and gradients reach both inputs.
        """
        config = small_config()
        config.moe_combine_group_size = 3
        moe = JTDeepseekV3MoE(config)
        values = torch.tensor([[[2**24], [1], [-2**24], [1], [1], [1]]],
                              dtype=torch.bfloat16, requires_grad=True)
        probabilities = torch.ones(1, 6, requires_grad=True)
        actual = moe._combine_experts(values, probabilities)
        torch.testing.assert_close(actual, torch.tensor([[3.0]], dtype=torch.bfloat16), rtol=0, atol=0)
        actual.float().sum().backward()
        torch.testing.assert_close(values.grad, torch.ones_like(values), rtol=0, atol=0)
        torch.testing.assert_close(probabilities.grad, values.detach().float().squeeze(-1), rtol=0, atol=0)

    def test_grouped_combine_partitions_and_validation(self):
        """Feature: Grouped reduction partitions and validation.

        Description: Partition grouped work across token boundaries and reject invalid sizes.
        Expectation: Partitions count groups and unsupported group sizes fail explicitly.
        """
        config = small_config()
        config.moe_combine_group_size = 3
        config.moe_combine_num_partitions = 4
        moe = JTDeepseekV3MoE(config)
        values = torch.tensor([2**24, 1, -2**24], dtype=torch.float32).repeat(7, 4).reshape(7, 12, 1)
        probabilities = torch.ones(7, 12)
        torch.testing.assert_close(moe._combine_experts(values, probabilities), torch.zeros(7, 1), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "must divide"):
            moe._combine_experts(values[:, :5], probabilities[:, :5])
        for invalid in (0, -1, True, 1.5):
            config.moe_combine_group_size = invalid
            with self.assertRaisesRegex(ValueError, "positive integer"):
                JTDeepseekV3MoE(config)

    def test_precision_units_survive_checkpoint_wrappers(self):
        """Feature: Mixed precision with recomputation.

        Description: Discover FP32 units after native checkpoint wrappers normalize module names.
        Expectation: Every returned FQN resolves uniquely to the intended precision-sensitive module.
        """
        model = JTDeepseekV3ForCausalLM(small_config())
        _apply_activation_checkpointing(model, "full")
        manager = JTFSDP2Manager(FSDP2Config(), SimpleNamespace(fsdp_moe_mesh=None))
        units = manager._find_wrap_modules(model)
        names = {unit.fqn for unit in units}
        self.assertIn("mtp.layers.0.transformer_layer.self_attn.q_a_layernorm", names)
        self.assertEqual(len(units), len(names))
        self.assertEqual(len(units), len({id(unit.module) for unit in units}))

    def test_recomputed_model_runs_consecutive_backward_steps(self):
        """Feature: Recomputed router state lifetime.

        Description: Run two complete forward/backward steps through checkpoint wrappers.
        Expectation: Router scores are returned directly and cannot be shadowed by stale wrapper attributes.
        """
        model = JTDeepseekV3ForCausalLM(small_config())
        _apply_activation_checkpointing(model, "full")
        tokens = torch.arange(8).unsqueeze(0)
        for _ in range(2):
            output = model(tokens, (tokens + 1) % 32)
            sum(output.loss.values()).backward()
            router = model.mtp.layers[0].transformer_layer.mlp.gate
            self.assertIsNotNone(router.weight.grad)
            self.assertTrue(torch.isfinite(router.weight.grad).all())
            model.zero_grad()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_rotary_preserves_fp32_tables_and_activation_dtype(self):
        """Feature: Rotary mixed precision.

        Description: Rotate BF16 activations using angles that are not representable in BF16.
        Expectation: The FP32 rotation is rounded once to the activation dtype and retains gradients.
        """
        values = torch.tensor([[[[1.5, 1.25, -0.5, 0.75]]]], dtype=torch.bfloat16, requires_grad=True)
        angles = torch.tensor([[[0.37, 1.29, 0.37, 1.29]]], dtype=torch.float32)
        cosine, sine = angles.cos(), angles.sin()
        ordered = torch.tensor([[[[1.5, -0.5, 1.25, 0.75]]]], dtype=torch.float32)
        rotated = torch.tensor([[[[-1.25, -0.75, 1.5, -0.5]]]], dtype=torch.float32)
        expected = (ordered * cosine.unsqueeze(1) + rotated * sine.unsqueeze(1)).to(torch.bfloat16)

        actual = JTDeepseekV3RotaryEmbedding()(values, cosine, sine)

        self.assertEqual(actual.dtype, values.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.float().sum().backward()
        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(values.grad).all())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_standalone_forward_backward(self):
        """Feature: Complete model.

        Description: Construct directly while replacement compilation is forbidden.
        Expectation: Trunk, MTP and router execute and backpropagate without an EP adapter.
        """
        torch.manual_seed(11)
        with patch("hyper_parallel.models.replacement.compile_module_replacements", side_effect=AssertionError):
            model = JTDeepseekV3ForCausalLM(small_config())
        self.assertIsInstance(model.model.layers[0], JTDeepseekV3Decoder)
        self.assertIsInstance(model.model.layers[1].mlp, JTDeepseekV3MoE)
        self.assertEqual(model.model.layers[1].mlp.ep_compute, model.model.layers[1].mlp.local_routed_forward)
        self.assertIsInstance(model.mtp.layers[0], MultiTokenPredictionLayer)
        self.assertIs(type(model.mtp.execution), DeepseekV3MTPExecution)
        tokens = torch.arange(8).unsqueeze(0)
        output = model(tokens, (tokens + 1) % 32)
        self.assertEqual(
            set(output.loss),
            {"foundation_loss/lm", "foundation_loss/mtp", "foundation_loss/aux"},
        )
        total_loss = sum(output.loss.values())
        self.assertTrue(torch.isfinite(total_loss))
        total_loss.backward()
        for name in ["model.embed_tokens.weight", "model.layers.1.mlp.gate.weight", "mtp.layers.0.eh_proj.weight"]:
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_standard_policy_bfloat16_forward_closes_activations(self):
        """Feature: Framework BF16 policy.

        Description: Emulate FSDP param_dtype=bfloat16 compute copies over FP32 storage; buffers stay as stored.
        Expectation: Activations stay BF16 without FSDP input casts; losses and the expert bias stay FP32.
        """
        model = JTDeepseekV3ForCausalLM(small_config())
        for parameter in model.parameters():
            parameter.data = parameter.data.to(torch.bfloat16)
        biases = [
            module.e_score_correction_bias
            for module in model.modules()
            if hasattr(module, "e_score_correction_bias")
        ]
        self.assertTrue(biases)
        self.assertTrue(all(bias.dtype == torch.float32 for bias in biases))

        activations = {}

        def capture(name: str) -> Any:
            """Record a selected module output by name."""

            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                """Store the tensor output after tuple unwrapping."""
                output = output[0] if isinstance(output, tuple) else output
                activations[name] = output

            return hook

        handles = [
            model.model.layers[0].self_attn.register_forward_hook(capture("attention")),
            model.model.layers[1].mlp.register_forward_hook(capture("moe")),
            model.model.layers[1].register_forward_hook(capture("decoder")),
        ]
        try:
            tokens = torch.arange(8).unsqueeze(0)
            with torch.no_grad():
                output = model(tokens, (tokens + 1) % 32)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual({value.dtype for value in activations.values()}, {torch.bfloat16})
        self.assertTrue(all(value.dtype == torch.float32 for value in output.loss.values()))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_replacement_selects_attention(self):
        """Feature: Acceleration boundary.

        Description: Apply the optional high-performance attention replacement.
        Expectation: Attention changes while decoder, MLP, norm and MTP weights survive.
        """
        model = JTDeepseekV3ForCausalLM(small_config())
        previous = dict(model.named_modules())
        self.assertEqual(sum(isinstance(m, JTDeepseekV3Attention) for m in previous.values()), 3)
        original_q = model.model.layers[0].self_attn.q_a_proj.weight.detach().clone()
        original_kv = model.model.layers[0].self_attn.kv_a_proj_with_mqa.weight.detach().clone()
        recipe_path = Path(__file__).resolve().parents[4] / (
            "examples/training_demo/jt_deepseek_v3/jt_deepseek_v3.yaml")
        recipe = parse_training_args([str(recipe_path)])
        rules = entries_to_module_replacements(recipe.plan_overrides)
        self.assertEqual(len(rules), 1)
        plan = compile_module_replacements(model, rules)
        apply_module_replacements(model, plan, weights_mapping=[])
        self.assertTrue(torch.equal(model.model.layers[0].self_attn.linear_qkv.weight,
                                    torch.cat((original_q, original_kv))))
        current = dict(model.named_modules())
        self.assertEqual(sum(isinstance(m, JTDeepseekV3MLAAttention) for m in current.values()), 3)
        for name in ["model.layers.0", "model.layers.1.mlp", "model.norm", "mtp.layers.0"]:
            self.assertIs(current[name], previous[name])
        self.assertIs(model.model.layers[0].self_attn.q_a_layernorm,
                      previous["model.layers.0.self_attn.q_a_layernorm"])


    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_meta_replacement_loads_converted_reference_state(self):
        """Feature: Offline checkpoint loading.

        Description: Load arrays that already use the model's final parameter layout.
        Expectation: Every materialized parameter is loaded exactly.
        """
        config = small_config()
        recipe_path = Path(__file__).resolve().parents[4] / (
            "examples/training_demo/jt_deepseek_v3/jt_deepseek_v3.yaml")
        rules = entries_to_module_replacements(parse_training_args([str(recipe_path)]).plan_overrides)
        with torch.device("meta"):
            candidate = JTDeepseekV3ForCausalLM(config)
            plan = compile_module_replacements(candidate, rules)
            apply_module_replacements(candidate, plan, weights_mapping=[])
        candidate.to_empty(device="cpu")
        arrays = {name: value.detach().numpy().copy() for name, value in candidate.state_dict().items()}
        loaded = _load_reference_state(candidate, arrays)
        self.assertEqual(set(loaded), set(arrays))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_family_registration_does_not_replace_standard_deepseek(self):
        """Feature: Independent family registration.

        Description: Resolve both families and the custom architecture identity.
        Expectation: JT owns its spec; the original V3/V2 identities stay registered independently.
        """
        standard = get_model_adapter("deepseek_v3")
        custom = get_model_adapter("jt_deepseek_v3")
        self.assertIs(get_model_adapter("JTDeepseekV3ForCausalLM"), custom)
        self.assertIsNot(standard, custom)
        self.assertEqual(standard.architecture, "DeepseekV3ForCausalLM")
        self.assertEqual(custom.model_type, "jt_deepseek_v3")
        self.assertEqual(small_config().model_type, "jt_deepseek_v3")
        self.assertIs(get_model_adapter(small_config().architectures[0]), custom)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_fsdp_discovery_uses_jt_identity_and_declared_mtp_unit(self):
        """Feature: FSDP unit discovery.

        Description: Resolve JT units and apply FP32 only through the model-local manager.
        Expectation: The shared manager has ordinary block units; JT alone adds its precision units.
        """
        model = JTDeepseekV3ForCausalLM(small_config())
        self.assertEqual(FSDP2Manager._get_model_adapter_spec(model).model_type, "jt_deepseek_v3")
        manager = FSDP2Manager(FSDP2Config(), SimpleNamespace(fsdp_moe_mesh=None))
        units = manager._find_wrap_modules(model)
        self.assertEqual(
            sorted(unit.fqn for unit in units),
            ["model.layers.0", "model.layers.1", "mtp.layers.0.transformer_layer"],
        )
        manager = JTFSDP2Manager(FSDP2Config(), SimpleNamespace(fsdp_moe_mesh=None))
        units = manager._find_wrap_modules(model)
        fp32_units = [unit for unit in units if getattr(unit.module, "keep_compute_in_fp32", False)]
        fp32_names = {unit.fqn for unit in fp32_units}
        self.assertIn("model.layers.1.mlp.gate", fp32_names)
        self.assertIn("model.norm", fp32_names)
        self.assertIn("mtp.layers.0.hnorm", fp32_names)
        self.assertEqual(len({id(unit.module) for unit in units}), len(units))

        manager.fp32_main_params = True
        policy = manager._build_mixed_precision_policy()
        with patch("hyper_parallel.distributed._builder.fsdp_adapter.fully_shard") as shard:
            manager._parallelize_child_units(units, {}, None, None, {"mp_policy": policy})
        fp32_ids = {id(unit.module) for unit in fp32_units}
        for call in shard.call_args_list:
            applied = call.kwargs["mp_policy"]
            self.assertTrue(applied.apply_grad_on_fp32_main_grad)
            if id(call.args[0]) in fp32_ids:
                self.assertEqual(applied.param_dtype, torch.float32)
                self.assertFalse(applied.cast_forward_inputs)
                self.assertIsNone(applied.output_dtype)
            else:
                self.assertIs(applied, policy)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_native_recipe_config_roundtrip(self):
        """Feature: Native model configuration.

        Description: Build and serialize the recipe's HF configuration without a reference YAML.
        Expectation: Model dimensions, JT options and independent adapter identity survive.
        """
        recipe_path = Path(__file__).resolve().parents[4] / (
            "examples/training_demo/jt_deepseek_v3/jt_deepseek_v3.yaml")
        recipe = parse_training_args([str(recipe_path)])
        config = JTDeepseekV3Config(**recipe.model.config)
        self.assertIs(type(config), JTDeepseekV3Config)
        self.assertFalse(hasattr(recipe.model, "reference_yaml"))
        self.assertFalse(hasattr(config, "jt_config"))
        restored = JTDeepseekV3Config.from_dict(config.to_dict())
        self.assertEqual(restored.to_dict(), config.to_dict())
        self.assertEqual(restored.mlp_layer_types, ["dense", "sparse"])
        self.assertEqual(restored.num_nextn_predict_layers, 1)
        self.assertEqual(restored.mtp_loss_factor, 0.3)
        restored.n_group = 2
        with self.assertRaisesRegex(ValueError, "n_group=topk_group=1"):
            JTDeepseekV3ForCausalLM(restored)
        self.assertEqual(restored.moe_aux_loss_coeff, 0.0001)
        self.assertEqual(restored.rope_parameters["rope_theta"], 5000000)
        self.assertIs(get_model_adapter(restored.architectures[0]), get_model_adapter("jt_deepseek_v3"))


    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_shared_batch_inputs_define_lm_and_mtp_objectives(self):
        """Feature: Native Trainer input and objective contract.

        Description: Forward the shared text batch's model inputs unchanged; the final target is the valid ID 0.
        Expectation: LM and MTP losses equal dense CE over the batch mask; MTP has no target past the end.
        """
        torch.manual_seed(5)
        model = JTDeepseekV3ForCausalLM(small_config())
        tokens = torch.arange(8).unsqueeze(0)
        labels = torch.roll(tokens, -1, 1)
        labels[0, [2, 5]] = -100
        mesh = SimpleNamespace(cp_size=1, pp_size=1, tp_size=1, dp_size=1, dp_rank=0, device_mesh=None)
        batch = TextParallelBatch(mesh, torch.device("cpu"), None, {}, False,
                                  source_type="indexed", attention_mode="compressed")
        model_inputs, loss_inputs = batch(iter([{"tokens": tokens, "labels": labels}]))
        head_outputs = []
        handle = model.lm_head.register_forward_hook(lambda _module, _inputs, output: head_outputs.append(output))
        try:
            output = model(**model_inputs, use_cache=False)
        finally:
            handle.remove()
        lm_logits, mtp_logits = (logits[0].detach() for logits in head_outputs)
        targets, mask = loss_inputs["shift_labels"][0], loss_inputs["loss_mask"][0].bool()
        torch.testing.assert_close(output.loss["foundation_loss/lm"], F.cross_entropy(lm_logits[mask], targets[mask]))
        # The single MTP depth predicts each next target, scaled by the MTP loss factor.
        expected_mtp = F.cross_entropy(mtp_logits[:-1][mask[1:]], targets[1:][mask[1:]]) * model.config.mtp_loss_factor
        torch.testing.assert_close(output.loss["foundation_loss/mtp"], expected_mtp)
        with self.assertRaisesRegex(ValueError, "attention_mask=None"):
            model(**(model_inputs | {"attention_mask": torch.ones_like(tokens)}))
        with self.assertRaisesRegex(ValueError, "explicit shift_labels"):
            model(input_ids=tokens, labels=labels)
