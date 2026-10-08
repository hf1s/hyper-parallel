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
"""Public MTP composition and gradient contracts."""

import unittest

import torch
from torch import nn

from hyper_parallel.components.modules.mtp import MultiTokenPrediction, MultiTokenPredictionLayer
from tests.common.mark_utils import arg_mark


class _Scale(nn.Module):
    """Output normalization stand-in that multiplies by a fixed factor."""

    def __init__(self, factor: float) -> None:
        """Store the multiplication factor."""
        super().__init__()
        self.factor = factor

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Scale ``values``."""
        return values * self.factor


class TestMultiTokenPrediction(unittest.TestCase):
    """Exercise injected non-DeepSeek components without distributed state."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_composition_and_gradient(self):
        """Both trunk states and future embeddings receive the expected gradients.

        Feature: mtp.
        Description: Both trunk states and future embeddings receive the expected gradients.
        Expectation: The asserted values and state transitions hold.
        """
        projection = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            projection.weight.copy_(torch.tensor([[1., 0., 2., 0.], [0., 1., 0., 2.]]))
        layer = MultiTokenPredictionLayer(embedding_norm=nn.Identity(), hidden_norm=nn.Identity(),
                                         projection=projection, decoder=nn.Identity(), output_norm=nn.Identity())
        hidden = torch.tensor([[[3., 4.]]], requires_grad=True)
        embedding = torch.tensor([[[5., 6.]]], requires_grad=True)
        result = layer(hidden, embedding)
        torch.testing.assert_close(result, hidden + 2 * embedding)
        result.sum().backward()
        torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))
        torch.testing.assert_close(embedding.grad, 2 * torch.ones_like(embedding))
        container = MultiTokenPrediction([layer])
        self.assertIn("layers.0.eh_proj.weight", container.state_dict())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_empty_depth_registers_no_parameters(self):
        """Models without MTP do not allocate a dummy prediction layer.

        Feature: mtp.
        Description: Models without MTP do not allocate a dummy prediction layer.
        Expectation: The asserted values and state transitions hold.
        """
        self.assertEqual(dict(MultiTokenPrediction([]).named_parameters()), {})


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_fusion_preserves_input_precision(self):
        """Preserve precision in the public fusion block.

        Feature: MTP fusion.
        Description: FP32 values survive fusion.
        Expectation: No BF16 rounding.
        """
        layer = MultiTokenPredictionLayer(embedding_norm=nn.Identity(), hidden_norm=nn.Identity(),
                                         projection=nn.Identity(), decoder=nn.Identity(), output_norm=nn.Identity())
        hidden = torch.tensor([[[1.001, 2.003]]], requires_grad=True)
        embedding = torch.tensor([[[3.005, 4.007]]], requires_grad=True)
        result = layer(hidden, embedding)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, torch.cat((hidden, embedding), dim=-1), rtol=0, atol=0)
        self.assertFalse(torch.equal(result, result.bfloat16().float()))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_depths_chain_normalized_outputs(self):
        """Chain depths over the normalized output, as Megatron and the reference do.

        Feature: MTP depth recurrence.
        Description: Run two depths whose output norms scale by 2 and 3, with an
            embedding that returns the token ID and a projection h + 10 * e.
        Expectation: Depth k embeds the IDs shifted left by k without wrapping;
            the head and the next depth both receive the depth's normalized output.
        """
        def depth(scale: float) -> MultiTokenPredictionLayer:
            """Build one depth computing ``scale * (h + 10 * e)``."""
            projection = nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                projection.weight.copy_(torch.tensor([[1., 10.]]))
            return MultiTokenPredictionLayer(
                embedding_norm=nn.Identity(), hidden_norm=nn.Identity(), projection=projection,
                decoder=nn.Identity(), output_norm=_Scale(scale))

        embedding = nn.Embedding(5, 1)
        with torch.no_grad():
            embedding.weight.copy_(torch.arange(5.).unsqueeze(-1))
        hidden = torch.tensor([[[0.5], [1.5], [2.5], [3.5]]])
        output = MultiTokenPrediction([depth(2.), depth(3.)])(
            hidden, torch.tensor([[1, 2, 3, 4]]), embedding=embedding, head=nn.Identity())

        first = 2 * (hidden + 10 * torch.tensor([[[2.], [3.], [4.], [0.]]]))
        second = 3 * (first + 10 * torch.tensor([[[3.], [4.], [0.], [0.]]]))
        self.assertEqual(len(output.logits), 2)
        torch.testing.assert_close(output.logits[0], first)
        torch.testing.assert_close(output.logits[1], second)
        torch.testing.assert_close(output.hidden_states, second)
