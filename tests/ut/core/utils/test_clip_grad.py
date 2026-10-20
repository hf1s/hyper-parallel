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
"""Archived MF reduction schedule of the ``clip_grad_norm_`` L2 norm on single-card gradients.

Every expected value is derived by hand in FP32. Squares of ``ROOT_EPS = 2 ** -12`` are
``EPS = 2 ** -24``: ``1 + EPS`` rounds back to ``1`` (ties to even) while ``EPS + EPS`` survives,
so whether the small squares meet each other before they meet ``1`` exposes the accumulation order.
"""
import math
import unittest

import numpy as np
import torch

from hyper_parallel.core.utils.clip_grad import clip_grad_norm_, gradient_square_sum, sequential_sum
from tests.common.mark_utils import arg_mark

ROOT_EPS = 2.0 ** -12
# A threshold far above every norm below: the clip coefficient clamps to 1 and only the norm is observed.
NO_CLIP = 10.0


def sparse_gradient(count: int, values: dict[int, float]) -> torch.Tensor:
    """Return ``count`` FP32 zeros holding ``values`` at the given flat indices."""
    gradient = torch.zeros(count, dtype=torch.float32)
    for index, value in values.items():
        gradient[index] = value
    return gradient


def parameters_with(gradients: list[torch.Tensor]) -> list[torch.nn.Parameter]:
    """Return plain parameters, in the given order, whose gradients are copies of ``gradients``."""
    parameters = []
    for gradient in gradients:
        parameter = torch.nn.Parameter(torch.zeros_like(gradient))
        parameter.grad = gradient.clone()
        parameters.append(parameter)
    return parameters


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
class TestArchivedL2Schedule(unittest.TestCase):
    """The p=2 norm follows the archived per-tensor SumX schedule and sequential host accumulation."""

    def test_tensor_sums_accumulate_sequentially_in_parameter_order(self):
        """Per-tensor square sums are added in parameter order with one FP32 rounding per addition."""
        one = sparse_gradient(64, {0: 1.0})
        smalls = [sparse_gradient(64, {0: ROOT_EPS}) for _ in range(8)]
        # 1 first: every EPS meets 1 alone and rounds away, although the exact norm is 1 + 2 ** -22.
        norm = clip_grad_norm_(parameters_with([one, *smalls]), NO_CLIP).item()
        self.assertEqual(norm, 1.0, f"1 before eight EPS: got={norm!r}, expected=1.0")
        # 1 last: the eight EPS first sum to 2 ** -21; sqrt(1 + 2 ** -21) rounds to 1 + 2 ** -22.
        norm = clip_grad_norm_(parameters_with([*smalls, one]), NO_CLIP).item()
        self.assertEqual(norm, 1.0 + 2.0 ** -22, f"eight EPS before 1: got={norm!r}, expected={1.0 + 2.0 ** -22!r}")

    def test_single_tile_halves_rows_before_summing_lanes(self):
        """Up to 5120 elements form one 64-lane tile whose rows meet by halves: row j with row j + rows // 2."""

        def tile(eps_rows: list[int]) -> torch.Tensor:
            """5120 elements = 80 rows x 64 lanes; 1 in row 0 and EPS squares in ``eps_rows``, all in lane 0."""
            return sparse_gradient(5120, {0: 1.0, **{64 * row: ROOT_EPS for row in eps_rows}})

        # Rows 1+41 and 21+61 pair, then meet each other, then row 0: 1 + 4 * EPS = 1 + 2 ** -22, whose sqrt
        # rounds to 1 + 2 ** -23. Adding the rows in order would give 1.
        norm = clip_grad_norm_(parameters_with([tile([1, 41, 21, 61])]), NO_CLIP).item()
        self.assertEqual(norm, 1.0 + 2.0 ** -23, f"paired EPS rows: got={norm!r}, expected={1.0 + 2.0 ** -23!r}")
        # Rows 40, 20, 10 and 5 are row 0's partners at the first four halvings: each EPS meets 1 alone.
        norm = clip_grad_norm_(parameters_with([tile([40, 20, 10, 5])]), NO_CLIP).item()
        self.assertEqual(norm, 1.0, f"EPS rows partnering row 0: got={norm!r}, expected=1.0")

    def test_large_tensors_use_count_selected_tiles_on_strided_cores(self):
        """40960 elements use 1024-element tiles; other counts above 5120 use 8144/16 tiles on 40 cores."""
        values = {0: 1.0, 1024: ROOT_EPS, 2048: ROOT_EPS}
        # 40960: the values sit in tiles 0, 1 and 2, i.e. on cores 0, 1 and 2, which add in order.
        total = gradient_square_sum(sparse_gradient(40960, values))
        self.assertEqual(total, np.float32(1.0), f"40960 elements: got={total!r}, expected=1.0")
        # Any other count keeps them in 8144-element tile 0 (rows 0, 64, 128 of 16 lanes), where the halvings
        # pair the two EPS before they meet 1.
        total = gradient_square_sum(sparse_gradient(40960 + 64, values))
        self.assertEqual(total, np.float32(1.0 + 2.0 ** -23),
                         f"40960 + 64 elements: got={total!r}, expected={1.0 + 2.0 ** -23!r}")
        # 42 tiles of 8144: tile i accumulates on core i % 40, so tiles 1 and 41 share core 1 while tile 40
        # joins tile 0 (holding 1) on core 0.
        for second_tile, expected in [(41, 1.0 + 2.0 ** -23), (40, 1.0)]:
            gradient = sparse_gradient(8144 * 42, {0: 1.0, 8144: ROOT_EPS, 8144 * second_tile: ROOT_EPS})
            total = gradient_square_sum(gradient)
            self.assertEqual(total, np.float32(expected),
                             f"EPS in tiles 1 and {second_tile}: got={total!r}, expected={expected!r}")

    def test_norm_is_the_sqrt_of_archived_sums_and_scales_every_gradient(self):
        """The norm is exactly sqrt(sequential_sum(gradient_square_sum)) and every gradient is clipped by it."""
        generator = torch.Generator().manual_seed(0)
        shapes = [(64,), (80, 64), (8, 5120), (100, 100), (1000, 700)]
        gradients = [torch.randn(shape, generator=generator) for shape in shapes]
        parameters = parameters_with(gradients)
        sums = np.asarray([gradient_square_sum(gradient) for gradient in gradients], dtype=np.float32)
        expected = torch.tensor(sequential_sum(sums), dtype=torch.float32).sqrt()
        max_norm = expected.item() / 4
        norm = clip_grad_norm_(parameters, max_norm)
        self.assertTrue(torch.equal(norm, expected), f"norm: got={norm.item()!r}, expected={expected.item()!r}")
        exact = math.sqrt(sum(gradient.double().square().sum().item() for gradient in gradients))
        self.assertLess(abs(norm.item() - exact), 1e-6 * exact, f"norm: got={norm.item()!r}, FP64={exact!r}")
        coefficient = torch.clamp(max_norm / (expected + 1e-6), max=1.0)
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            clipped = gradient * coefficient
            self.assertTrue(torch.equal(parameter.grad, clipped),
                            f"gradient {index}: max |got - expected|={(parameter.grad - clipped).abs().max()!r}")


if __name__ == "__main__":
    unittest.main()
