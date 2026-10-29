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
"""CPU contracts for explicit MLA callbacks and Xing40 rotary precision."""
from typing import Any
import unittest
import torch
from transformers.models.deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave
from hyper_parallel.distributed.context_parallel import mla_context_parallel as mla
from hyper_parallel.models.xing40.adapter.distributed.context_parallel import xing40_rotary


class _FalseCallable:
    """A callable's truth value must not select an unrelated default."""
    def __bool__(self) -> bool:
        """Return false while retaining an explicitly supplied callback."""
        return False

    def __call__(self, value: torch.Tensor, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        """Return the input unchanged for callback selection tests."""
        return value


class TestXing40CPCallbacks(unittest.TestCase):
    """Cover new callback selection and the family-specific rotary contract."""

    def test_optional_callbacks_are_selected_by_none(self):
        """
        Feature: Optional MLA local compute callbacks.
        Description: Preserve defaults and explicitly supplied callable objects.
        Expectation: Only None selects defaults; noncallables are rejected.
        """
        plan = mla.MLACPPlan(mla.MLADimensions(4, 5, 6, 3, 2, 4), 1)
        default = mla.MLAContextParallel(plan, None)
        self.assertIs(default.rotary_fn, mla._apply_mla_rope)
        self.assertIs(default.attention_interface, mla.npu_fusion_attention_forward)
        callback = _FalseCallable()
        custom = mla.MLAContextParallel(plan, None, rotary_fn=callback, attention_interface=callback)
        self.assertIs(custom.rotary_fn, callback)
        self.assertIs(custom.attention_interface, callback)
        with self.assertRaisesRegex(TypeError, 'rotary_fn must be callable'):
            mla.MLAContextParallel(plan, None, rotary_fn=1)
        with self.assertRaisesRegex(TypeError, 'attention_interface must be callable'):
            mla.MLAContextParallel(plan, None, attention_interface=1)

    def test_xing40_rotary_matches_family_reference_and_gradient(self):
        """
        Feature: Xing40 interleaved rotary dtype contract.
        Description: Compare explicit FP32 family rotation and its input gradient.
        Expectation: The callback rounds once to the input dtype and differentiates correctly.
        """
        value = torch.linspace(-2, 2, 48).reshape(2, 3, 1, 8).to(torch.bfloat16).requires_grad_()
        reference = value.detach().clone().requires_grad_()
        angle = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8) / 19
        positions = angle.cos(), angle.sin()
        output = xing40_rotary(value, positions, interleaved=True)
        expected, _ = apply_rotary_pos_emb_interleave(reference.float(), reference.float(), *positions, unsqueeze_dim=2)
        expected = expected.to(reference.dtype)
        self.assertEqual(output.dtype, torch.bfloat16)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        output.float().square().sum().backward()
        expected.float().square().sum().backward()
        torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'position embeddings'):
            xing40_rotary(value.detach(), None, interleaved=True)
        with self.assertRaisesRegex(ValueError, 'interleaved'):
            xing40_rotary(value.detach(), positions)
