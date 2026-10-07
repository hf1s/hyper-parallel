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
"""Public supervision forwarding and unmodified model-computed objectives."""

from types import SimpleNamespace
import unittest

import torch

from hyper_parallel.components.losses.model_output import ModelOutputLoss
from tests.common.mark_utils import arg_mark


class TestModelOutputLoss(unittest.TestCase):
    """Verify forwarding without renaming, shifting or numerical compensation."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_named_loss_mapping_bypasses_scalar_label_guard(self):
        """Feature: Model-owned named objective.

        Description: Return a named objective while all labels are masked.
        Expectation: The mapping is returned unchanged and remains differentiable.
        """
        value = torch.tensor(2., requires_grad=True)
        losses = {"foundation_loss/aux": value}
        result = ModelOutputLoss()(
            model_output=SimpleNamespace(loss=losses),
            labels=torch.full((1, 2), -100),
        )
        self.assertIs(result, losses)
        result["foundation_loss/aux"].backward()
        self.assertEqual(value.grad.item(), 1.)


    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_default_causal_label_check_and_gradient(self):
        """Feature: Existing causal objective behavior.

        Description: Supply valid, ignored and first-position-only causal labels.
        Expectation: Only a valid shifted target retains the loss and its gradient.
        """
        for labels, expected in (([[1, 2]], 2.), ([[-100, -100]], 0.), ([[1, -100]], 0.)):
            with self.subTest(labels=labels):
                value = torch.tensor(2., requires_grad=True)
                result = ModelOutputLoss()(model_output=SimpleNamespace(loss=value), labels=torch.tensor(labels))
                self.assertEqual(result.item(), expected)
                result.backward()
                self.assertEqual(value.grad.item(), expected / 2.)
