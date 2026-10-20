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
"""Archived FP32 reduction schedules: coverage of every element and the fixed pairing order."""
import unittest

import numpy as np
import torch

from hyper_parallel.models.jt_deepseek_v3.archived_reductions import (
    adjacent_tree,
    core_sum,
    gradient_square_sum,
    half_tree,
    sequential_sum,
    tiled_square_sum,
    weighted_sum,
)
from tests.common.mark_utils import arg_mark

# 1 + EPS rounds back to 1 in FP32 (ties to even), while EPS + EPS = 2 * EPS survives;
# whether two EPS terms meet before the 1 therefore exposes the pairing order.
EPS = 2.0 ** -24
ONE_PLUS_TWO_EPS = 1.0 + 2.0 ** -23


def lanes(*values: float) -> torch.Tensor:
    """Build one FP32 row from explicit lane values."""
    return torch.tensor(values, dtype=torch.float32)


class TestArchivedReductions(unittest.TestCase):
    """Lock the archived pairing order and check that every element contributes once."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_reduction_includes_odd_tail_and_partial_tiles(self):
        """Every element contributes, including non-power-of-two widths and final partial tiles."""
        values = torch.arange(1, 6, dtype=torch.float32)
        self.assertEqual(half_tree(values).item(), 15)
        self.assertEqual(adjacent_tree(values).item(), 15)
        ones = torch.ones(8144 * 2 + 17, dtype=torch.float32)
        self.assertEqual(tiled_square_sum(ones, 8144, 16), np.float32(ones.numel()))
        ones[-1] = 2
        self.assertEqual(tiled_square_sum(ones, 8144, 16), np.float32(ones.numel() + 3))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_half_tree_pairs_halves_and_folds_the_odd_tail_into_lane_zero(self):
        """Entry j meets entry j + n // 2; an odd tail joins the first partial sum."""
        self.assertEqual(half_tree(lanes(1.0, EPS, 0.0, EPS)).item(), ONE_PLUS_TWO_EPS)
        # Folding the tail into lane 1 instead would give (1) + (EPS + EPS) = 1 + 2 * EPS.
        self.assertEqual(half_tree(lanes(1.0, 0.0, 0.0, EPS, EPS)).item(), 1.0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_adjacent_tree_pairs_neighbours_and_zero_pads(self):
        """Lane 2k meets lane 2k + 1; non-power-of-two widths are padded with zeros."""
        self.assertEqual(adjacent_tree(lanes(1.0, EPS, 0.0, EPS)).item(), 1.0)
        self.assertEqual(adjacent_tree(lanes(EPS, EPS, 1.0)).item(), ONE_PLUS_TWO_EPS)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_host_sums_round_after_every_addition_in_core_order(self):
        """Scalar sums round after each addition; core sums stride tiles over the cores first."""
        partials = np.asarray([1.0, EPS, 0.0, EPS], dtype=np.float32)
        self.assertEqual(sequential_sum(partials), np.float32(1.0))
        # Tile i accumulates on core i % cores: core 0 holds 1 + 0, core 1 holds EPS + EPS.
        self.assertEqual(core_sum(partials, cores=2), np.float32(ONE_PLUS_TWO_EPS))
        self.assertEqual(core_sum(partials, cores=1), np.float32(1.0))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_square_sum_tile_reduces_rows_by_halves_then_lanes_natively(self):
        """Element r * width + l is row r of lane l; rows meet by halves, then lanes sum natively."""
        rows = torch.zeros(64, dtype=torch.float32)
        rows[[0, 16, 48]] = torch.tensor([1.0, 2.0 ** -12, 2.0 ** -12])  # squares 1, EPS, EPS in lane 0
        self.assertEqual(tiled_square_sum(rows, 64, 16), np.float32(ONE_PLUS_TWO_EPS))
        lanes_only = torch.zeros(16, dtype=torch.float32)
        lanes_only[[0, 1, 3]] = torch.tensor([1.0, 2.0 ** -12, 2.0 ** -12])  # squares 1, EPS, EPS in row 0
        self.assertEqual(tiled_square_sum(lanes_only, 16, 16), np.float32(1.0))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_gradient_square_sum_selects_the_archived_tiling_by_element_count(self):
        """Up to 5120 elements form one 64-lane tile; exactly 40960 use 1024/64 tiles, not the 8144/16 default."""
        single_tile = torch.zeros(5120, dtype=torch.float32)
        single_tile[[0, 64, 192]] = torch.tensor([1.0, 2.0 ** -12, 2.0 ** -12])  # lane 0, rows 0, 1, 3
        self.assertEqual(gradient_square_sum(single_tile), np.float32(ONE_PLUS_TWO_EPS))
        separate_tiles = torch.zeros(40960, dtype=torch.float32)
        separate_tiles[[0, 1024, 2048]] = torch.tensor([1.0, 2.0 ** -12, 2.0 ** -12])  # tiles 0, 1, 2 of 1024
        self.assertEqual(gradient_square_sum(separate_tiles), np.float32(1.0))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_square_sums_match_double_precision(self):
        """Every SumX tiling stays within the FP32 rounding bound of the FP64 sum of squares."""
        torch.manual_seed(253)
        # The longest path (8144/16 tiles, 601 tiles over 40 cores) rounds at most 76 times, squaring included.
        bound = 128 * 2.0 ** -24
        for count in (64, 5120, 40960, 8144 * 600 + 5):
            value = torch.randn(count, dtype=torch.float32)
            expected = value.double().square().sum().item()
            actual = float(gradient_square_sum(value))
            self.assertLess(abs(actual - expected), bound * expected, f"count={count}")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_weighted_sum_follows_80_row_half_tree_then_native_64_lane_sum(self):
        """Hidden position r * 64 + l is row r of lane l: rows meet by halves (locked, archived order);
        lanes sum natively (issue #252 arm R10 found the archived adjacent-pair lane tree numerically
        inert for this layout). Native torch sum isn't naive left-to-right accumulation either, so this
        same two-EPS-before-the-1 construction still survives across lanes, not just across rows."""
        rows = torch.zeros(1, 5120, dtype=torch.bfloat16)
        rows[0, [0, 64, 192]] = torch.tensor([1.0, EPS, EPS], dtype=torch.bfloat16)  # lane 0, rows 0, 1, 3
        self.assertEqual(weighted_sum(rows).item(), ONE_PLUS_TWO_EPS)
        lanes_only = torch.zeros(1, 5120, dtype=torch.bfloat16)
        lanes_only[0, [0, 1, 3]] = torch.tensor([1.0, EPS, EPS], dtype=torch.bfloat16)  # row 0, lanes 0, 1, 3
        self.assertEqual(weighted_sum(lanes_only).item(), 1.0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_weighted_sum_matches_double_precision_and_includes_every_position(self):
        """Random rows stay within 13 FP32 additions of FP64; every hidden position contributes once."""
        torch.manual_seed(255)
        product = torch.randn(6, 5120).bfloat16()
        actual = weighted_sum(product)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.shape, (6, 1))
        gamma = 13 * 2.0 ** -24 / (1 - 13 * 2.0 ** -24)
        bound = gamma * product.double().abs().sum(-1).max().item()
        torch.testing.assert_close(actual.double(), product.double().sum(-1, keepdim=True), rtol=0, atol=bound)
        torch.testing.assert_close(weighted_sum(torch.eye(5120, dtype=torch.bfloat16)), torch.ones(5120, 1),
                                   rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
