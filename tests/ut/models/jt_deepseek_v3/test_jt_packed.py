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
"""Packed JT objectives keep document semantics with chunked output projection."""

import copy
from types import SimpleNamespace
import unittest

import torch
from torch.utils.checkpoint import DefaultDeviceType

from hyper_parallel.models.jt_deepseek_v3.adapter.distributed.context_parallel import configure_context_parallel
from hyper_parallel.data.batching.get_batch import TextParallelBatch
from hyper_parallel.models.jt_deepseek_v3.adapter.data.runtime import JTSequenceRuntime
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (
    JTDeepseekV3Config, JTDeepseekV3ForCausalLM,
)
from hyper_parallel.trainer.runtime.flops import TransformerFlopsEstimator, transformer_flops
from tests.common.mark_utils import arg_mark
from tests.ut.models.jt_deepseek_v3.test_modeling_jt_deepseek_v3 import small_config


class TestJTPacked(unittest.TestCase):
    """Compare actual model losses and gradients across projection policies."""

    def setUp(self) -> None:
        """Keep CPU checkpoint recomputation independent of installed accelerators."""
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_packed_lm_mtp_equal_independent_documents(self) -> None:
        """Packed losses and gradients equal token-weighted independent document runs."""
        torch.manual_seed(8)
        config = small_config()
        config.moe_aux_loss_coeff = 0.0
        config.num_nextn_predict_layers = 2
        packed = JTDeepseekV3ForCausalLM(config)
        separate = copy.deepcopy(packed)
        tokens = torch.arange(1, 9).unsqueeze(0)
        labels = torch.tensor([[2, 3, 4, -100, 6, 7, 8, -100]])
        output = packed(tokens, labels, actual_seq_len=(4, 8)).loss
        left = separate(tokens[:, :4], labels[:, :4]).loss
        right = separate(tokens[:, 4:], labels[:, 4:]).loss
        keys = ("foundation_loss/lm", "foundation_loss/mtp")
        expected = {key: (left[key] + right[key]) / 2 for key in keys}
        for key in keys:
            torch.testing.assert_close(output[key], expected[key], rtol=2e-5, atol=2e-6)
        sum(output[key] for key in keys).backward()
        sum(expected.values()).backward()
        torch.testing.assert_close(
            {name: parameter.grad for name, parameter in packed.named_parameters()},
            {name: parameter.grad for name, parameter in separate.named_parameters()}, rtol=3e-4, atol=2e-6)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_chunked_lm_mtp_preserves_packed_losses_and_gradients(self) -> None:
        """Two MTP depths, masked documents and uneven chunks use eager token weighting."""
        torch.manual_seed(9)
        config = small_config()
        config.num_nextn_predict_layers = 2
        eager = JTDeepseekV3ForCausalLM(config)
        chunked = copy.deepcopy(eager)
        chunked.loss_chunk_size = 3
        inputs = torch.arange(9).unsqueeze(0)
        targets = torch.tensor([[-100, -100, -100, 4, 5, -100, 7, 8, -100]])
        expected = eager(inputs, targets, actual_seq_len=(3, 6, 9)).loss
        actual = chunked(inputs, targets, actual_seq_len=(3, 6, 9)).loss
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=2e-5, atol=2e-6)
        sum(actual.values()).backward()
        sum(expected.values()).backward()
        torch.testing.assert_close(
            {name: parameter.grad for name, parameter in chunked.named_parameters()},
            {name: parameter.grad for name, parameter in eager.named_parameters()}, rtol=3e-4, atol=2e-6)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_chunking_rejects_cp_before_installing_runtime(self) -> None:
        """Local chunk denominators must not bypass the upstream global CP normalization."""
        model = JTDeepseekV3ForCausalLM(small_config())
        model.loss_chunk_size = 4
        with self.assertRaisesRegex(ValueError, "loss_chunk_size=0 for CP"):
            configure_context_parallel(model, SimpleNamespace(cp_size=2))
        self.assertFalse(hasattr(model, "jt_cp_context"))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_diagnostic_losses_preserve_objective_and_gradients(self) -> None:
        """Raw MTP means and unscaled per-layer aux means never join the backward sum."""
        for chunk_size in (0, 3):
            with self.subTest(chunk_size=chunk_size):
                torch.manual_seed(21)
                config = small_config()
                config.num_nextn_predict_layers = 2
                config.loss_chunk_size = chunk_size
                reference = JTDeepseekV3ForCausalLM(config)
                logged = copy.deepcopy(reference)
                inputs = torch.arange(8).unsqueeze(0)
                targets = torch.tensor([[1, 2, 3, -100, 5, 6, 7, -100]])
                expected = reference.compute_jt_losses(inputs, targets, targets >= 0, actual_seq_len=(4, 8))
                output = logged(inputs, targets, actual_seq_len=(4, 8))
                metrics = output.loss_metrics
                self.assertEqual(set(metrics), {"lm_loss", "mtp_1_loss", "mtp_2_loss", "load_balancing_loss"})
                self.assertTrue(all(not value.requires_grad and value.grad_fn is None for value in metrics.values()))
                torch.testing.assert_close(metrics["lm_loss"], expected["lm_loss"], rtol=0, atol=0)
                torch.testing.assert_close((metrics["mtp_1_loss"] + metrics["mtp_2_loss"]) * 0.15,
                                           expected["mtp_loss"], rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(metrics["load_balancing_loss"] * (0.01 * 3),
                                           expected["aux_loss"], rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(sum(output.loss.values()), sum(expected.values()), rtol=0, atol=0)
                sum(output.loss.values()).backward()
                sum(expected.values()).backward()
                torch.testing.assert_close(
                    {name: parameter.grad for name, parameter in logged.named_parameters()},
                    {name: parameter.grad for name, parameter in reference.named_parameters()}, rtol=0, atol=0,
                )

    def test_disabled_auxiliary_objectives_have_no_diagnostic_entries(self) -> None:
        """Dense training with MTP and router loss disabled only exposes LM loss."""
        config = small_config()
        config.num_nextn_predict_layers = 0
        config.moe_aux_loss_coeff = 0.0
        config.mlp_layer_types = ["dense"] * config.num_hidden_layers
        model = JTDeepseekV3ForCausalLM(config)
        tokens = torch.arange(8).unsqueeze(0)
        output = model(tokens, tokens)
        self.assertEqual(set(output.loss_metrics), {"lm_loss"})
        torch.testing.assert_close(output.loss_metrics["lm_loss"], output.loss["foundation_loss/lm"],
                                   rtol=0, atol=0)

    def test_executed_jt_geometry_and_indexed_packed_flops(self) -> None:
        """Real JT config and indexed runtime describe full MLA, MTP and document boundaries."""
        config = JTDeepseekV3Config.from_dict(small_config().to_dict())
        self.assertEqual(config.index_topk, 0)
        self.assertEqual(config.layer_types, ["full_attention", "full_attention"])
        model = JTDeepseekV3ForCausalLM(config)
        estimator = TransformerFlopsEstimator(config, model)
        self.assertIsNotNone(estimator.coefficients)
        mesh = SimpleNamespace(dp_size=1, cp_size=1, tp_size=1, pp_size=1)
        get_batch = TextParallelBatch(
            mesh_context=mesh, device=torch.device("cpu"), tokenizer=SimpleNamespace(eod=7),
            data_config={}, pp_shared_data=False, source_type="indexed", attention_mode="compressed",
            runtime_input_adapter=JTSequenceRuntime(),
        )
        tokens = torch.tensor([[0, 1, 7, 2, 3, 4, 5, 6]])
        model_inputs, loss_inputs = get_batch(iter([{"tokens": tokens, "labels": tokens}]))
        self.assertFalse(get_batch.attention_runtime.reset_mask)
        self.assertEqual(model_inputs["actual_seq_len"], (3, 8))
        self.assertNotIn("cu_seq_lens", loss_inputs)
        expected = transformer_flops(config, mtp_layer_types=(True,)).estimate(8, 34)
        self.assertEqual(estimator({**model_inputs, **loss_inputs}), expected)
        self.assertGreater(expected, transformer_flops(config).estimate(8, 34))
