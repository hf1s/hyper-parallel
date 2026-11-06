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
"""Auxiliary gradients must follow the scalar objective under recomputation."""

from concurrent.futures import ThreadPoolExecutor
import unittest

import torch
from torch.utils.checkpoint import DefaultDeviceType, checkpoint

from hyper_parallel.components.functional.aux_loss import (
    aux_loss_auto_scale,
    aux_loss_scale_context,
    bind_aux_loss_scale,
    set_aux_loss_scale,
)
from tests.common.mark_utils import arg_mark


class TestAuxLossScale(unittest.TestCase):
    """Compare injected gradients against an explicit scalar objective."""

    def setUp(self) -> None:
        """Keep CPU checkpoint tests independent of the registered accelerator."""
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")

    def tearDown(self) -> None:
        """Restore the standalone API's default scale."""
        set_aux_loss_scale(torch.tensor(1.0))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_weighting_and_checkpoint_match_explicit_loss(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: Token weighting, loss scaling and checkpointing affect both losses.
        Expectation: Eager and both checkpoint modes match the explicitly weighted auxiliary gradient.
        """
        for use_checkpoint in (False, True):
            for reentrant in (False, True):
                for scale in (0.0, 0.125, 3.0):
                    with self.subTest(checkpoint=use_checkpoint, reentrant=reentrant, scale=scale):
                        value = torch.tensor([0.5, 2.0], requires_grad=True)

                        def forward(inputs: torch.Tensor) -> torch.Tensor:
                            """Attach a scalar penalty to an unchanged activation."""
                            return aux_loss_auto_scale(inputs.square(), inputs.pow(3).mean())

                        with aux_loss_scale_context():
                            output = (
                                checkpoint(forward, value, use_reentrant=reentrant)
                                if use_checkpoint else forward(value)
                            )
                            loss = output.sum()
                            bind_aux_loss_scale(loss)
                            (loss * scale).backward()
                        expected = scale * (2 * value.detach() + 1.5 * value.detach().square())
                        torch.testing.assert_close(value.grad, expected)
                        torch.testing.assert_close(output, value.detach().square())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_contexts_and_manual_scale_do_not_leak(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: Nested microsteps restore their context and preserve manual callers.
        Expectation: Nested contexts remain independent and the manual fallback retains its own scale.
        """
        set_aux_loss_scale(torch.tensor(7.0))
        outer = torch.tensor(2.0, requires_grad=True)
        inner = torch.tensor(3.0, requires_grad=True)
        with aux_loss_scale_context():
            outer_loss = aux_loss_auto_scale(outer * 0, outer.square())
            bind_aux_loss_scale(outer_loss)
            with aux_loss_scale_context():
                inner_loss = aux_loss_auto_scale(inner * 0, inner.square())
                bind_aux_loss_scale(inner_loss)
                (inner_loss * 0.5).backward()
            (outer_loss * 0.25).backward()
        torch.testing.assert_close(outer.grad, torch.tensor(1.0))
        torch.testing.assert_close(inner.grad, torch.tensor(3.0))
        standalone = torch.tensor(2.0, requires_grad=True)
        aux_loss_auto_scale(standalone * 0, standalone.square()).backward()
        torch.testing.assert_close(standalone.grad, torch.tensor(28.0))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_no_auxiliary_loss_leaves_main_gradient_unchanged(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: Ordinary models retain the same scalar objective.
        Expectation: The main gradient remains exactly that of the task loss.
        """
        value = torch.tensor(2.0, requires_grad=True)
        with aux_loss_scale_context():
            loss = value.square()
            bind_aux_loss_scale(loss)
            (loss * 0.25).backward()
        torch.testing.assert_close(value.grad, torch.tensor(1.0))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_tp_replica_averaging_is_explicit(self) -> None:
        """
        Feature: Replicated auxiliary means under loss parallelism.
        Description: A replicated mean and a partial objective share one backward context.
        Expectation: Only the declared mean is averaged, only for TP loss parallelism.
        """
        for loss_parallel, tp_size, divisor in ((True, 2, 2), (False, 2, 1), (True, 1, 1)):
            with self.subTest(loss_parallel=loss_parallel, tp_size=tp_size):
                main = torch.tensor(4.0, requires_grad=True)
                replicated = torch.tensor(2.0, requires_grad=True)
                partial = torch.tensor(3.0, requires_grad=True)
                with aux_loss_scale_context(loss_parallel=loss_parallel, tp_size=tp_size):
                    output = aux_loss_auto_scale(main, replicated.square(), tp_replicated=True)
                    output = aux_loss_auto_scale(output, partial.square())
                    loss = output.square()
                    bind_aux_loss_scale(loss)
                    (loss * 5).backward()
                torch.testing.assert_close(output, main.detach())
                torch.testing.assert_close(main.grad, torch.tensor(40.0))
                torch.testing.assert_close(replicated.grad, torch.tensor(20.0 / divisor))
                torch.testing.assert_close(partial.grad, torch.tensor(30.0))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_replica_metadata_survives_checkpoint_and_backward_thread(self) -> None:
        """
        Feature: Replicated auxiliary-loss lifecycle.
        Description: Non-reentrant recomputation and backward run in a different Python context.
        Expectation: The original autograd context retains both the multiplier and TP degree.
        """
        value = torch.tensor(2.0, requires_grad=True)

        def forward(inputs: torch.Tensor) -> torch.Tensor:
            return aux_loss_auto_scale(inputs.square(), inputs.pow(3), tp_replicated=True)

        with aux_loss_scale_context(loss_parallel=True, tp_size=2):
            loss = checkpoint(forward, value, use_reentrant=False)
            bind_aux_loss_scale(loss)
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit((loss * 7).backward).result()
        torch.testing.assert_close(value.grad, torch.tensor(70.0))
