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
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3ForCausalLM
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
