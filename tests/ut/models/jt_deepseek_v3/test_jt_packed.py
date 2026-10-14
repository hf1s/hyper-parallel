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
from itertools import product
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import DefaultDeviceType

from hyper_parallel.distributed.activation_checkpoint import _apply_activation_checkpointing
from hyper_parallel.models.jt_deepseek_v3.adapter.distributed.context_parallel import configure_context_parallel
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3ForCausalLM, JTDeepseekV3MoE
from tests.common.mark_utils import arg_mark
from tests.ut.models.jt_deepseek_v3.test_modeling_jt_deepseek_v3 import small_config


def reference_grouped_swiglu(inputs: torch.Tensor, gate_up: torch.Tensor,
                            down: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Use differentiable CPU expert math to test loss policy independently of NPU kernels."""
    outputs = []
    for index, tokens in enumerate(torch.split(inputs, counts.tolist())):
        gate, up = F.linear(tokens, gate_up[index]).chunk(2, dim=-1)
        outputs.append(F.linear(F.silu(gate) * up, down[index]))
    return torch.cat(outputs)


class TestJTPacked(unittest.TestCase):
    """Compare actual model losses and gradients across projection policies."""

    def setUp(self) -> None:
        """Keep CPU checkpoint recomputation independent of installed accelerators."""
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")
        self.enterContext(patch(
            "hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3.npu_grouped_swiglu",
            side_effect=reference_grouped_swiglu))

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
        chunked_template = copy.deepcopy(eager)
        inputs = torch.arange(9).unsqueeze(0)
        targets = torch.tensor([[-100, -100, -100, 4, 5, -100, 7, 8, -100]])
        expected = eager(inputs, targets, actual_seq_len=(3, 6, 9)).loss
        sum(expected.values()).backward()
        for recompute in (False, True):
            with self.subTest(recompute=recompute):
                chunked = copy.deepcopy(chunked_template)
                chunked.loss_chunk_size = 3
                chunked.loss_recompute = recompute
                actual = chunked(inputs, targets, actual_seq_len=(3, 6, 9)).loss
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], rtol=2e-5, atol=2e-6)
                sum(actual.values()).backward()
                torch.testing.assert_close(
                    {name: parameter.grad for name, parameter in chunked.named_parameters()},
                    {name: parameter.grad for name, parameter in eager.named_parameters()}, rtol=3e-4, atol=2e-6)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_layer_and_loss_checkpoint_gradients(self) -> None:
        """Real full/selective layer wrappers compose with both LM/MTP loss policies."""
        torch.manual_seed(19)
        config = small_config()
        reference = JTDeepseekV3ForCausalLM(config)
        self.assertTrue(reference.loss_recompute)
        tokens = torch.arange(9).unsqueeze(0)
        targets = torch.tensor([[-100, -100, 3, 4, -100, 6, 7, 8, -100]])
        expected = reference(tokens, targets, actual_seq_len=(3, 6, 9)).loss
        sum(expected.values()).backward()
        for mode, recompute in product(("full", "selective"), (False, True)):
            with self.subTest(mode=mode, recompute=recompute):
                chunk_config = copy.deepcopy(config)
                chunk_config.loss_chunk_size = 3
                chunk_config.loss_recompute = recompute
                model = JTDeepseekV3ForCausalLM(chunk_config)
                model.load_state_dict(reference.state_dict())
                self.assertEqual(model.loss_recompute, recompute)
                model = _apply_activation_checkpointing(model, mode)
                head_calls = []
                handle = model.lm_head.register_forward_hook(
                    lambda _module, _args, output: head_calls.append(output.shape[1]))
                self.addCleanup(handle.remove)
                actual = model(tokens, targets, actual_seq_len=(3, 6, 9)).loss
                for key, loss in expected.items():
                    torch.testing.assert_close(actual[key], loss, rtol=2e-5, atol=2e-6)
                auxiliary_caches = [(module, module.auxiliary_loss) for module in model.modules()
                                    if isinstance(module, JTDeepseekV3MoE)]
                forward_calls = len(head_calls)
                sum(actual.values()).backward()
                self.assertEqual(len(head_calls), forward_calls * (2 if recompute else 1))
                for module, forward_aux in auxiliary_caches:
                    self.assertIs(module.auxiliary_loss, forward_aux)
                for parameter, original in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(parameter.grad, original.grad, rtol=3e-4, atol=2e-6)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_loss_recompute_rejects_non_boolean_config(self) -> None:
        """A quoted YAML false must not accidentally enable checkpointing."""
        for value in ("false", "off", 0, 1, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "loss_recompute must be a bool"):
                config = small_config()
                config.loss_recompute = value
                JTDeepseekV3ForCausalLM(config)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_chunking_rejects_cp_before_installing_runtime(self) -> None:
        """Local chunk denominators must not bypass the upstream global CP normalization."""
        model = JTDeepseekV3ForCausalLM(small_config())
        model.loss_chunk_size = 4
        with self.assertRaisesRegex(ValueError, "loss_chunk_size=0 for CP"):
            configure_context_parallel(model, SimpleNamespace(cp_size=2))
        self.assertFalse(hasattr(model, "jt_cp_context"))
