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
"""JT LM/MTP projection chunking preserves the native losses and parameter gradients."""

import copy
import unittest

import torch

from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3ForCausalLM
from tests.common.mark_utils import arg_mark
from tests.ut.models.jt_deepseek_v3.test_modeling_jt_deepseek_v3 import small_config


class TestJTChunkedLoss(unittest.TestCase):
    """Compare all model objectives and gradients, including uneven packed documents."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_native_and_chunked_loss_gradients(self) -> None:
        """Two MTP depths share the head without changing boundary or normalization semantics."""
        for boundaries in (None, (0, 1, 4, 9)):
            with self.subTest(boundaries=boundaries):
                torch.manual_seed(9)
                config = small_config()
                config.num_nextn_predict_layers = 2
                native = JTDeepseekV3ForCausalLM(config)
                chunked = copy.deepcopy(native)
                chunked.loss_chunk_size = 2
                tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
                targets = torch.tensor([[2, -100, 4, -100, 6, 7, -100, 9, 10]])
                expected = native(tokens, targets, cu_seq_lens=boundaries).loss
                actual = chunked(tokens, targets, cu_seq_lens=boundaries).loss
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key])
                (sum(actual.values()) * 0.37).backward()
                (sum(expected.values()) * 0.37).backward()
                for (name, parameter), (_, reference) in zip(chunked.named_parameters(), native.named_parameters()):
                    with self.subTest(parameter=name):
                        torch.testing.assert_close(parameter.grad, reference.grad, atol=1e-6, rtol=1e-5)
