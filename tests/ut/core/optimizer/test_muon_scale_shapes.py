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
"""Model-defined RMS shapes preserve native Muon update arithmetic."""

import math
import unittest

import torch

from hyper_parallel.core.optimizer.muon import Muon, NSInputTransform
from tests.common.mark_utils import arg_mark


class TestMuonScaleShapes(unittest.TestCase):
    """Separate matrix partitioning, scale selection and BF16 restoration contracts."""

    def setUp(self) -> None:
        """Prepare unequal blocks and a fixed NS direction to isolate RMS scaling."""
        self.parameter = torch.nn.Parameter(torch.zeros(6, 2))
        self.parameter.model_name = "projection"
        self.optimizer = object.__new__(Muon)
        self.optimizer.reshape_fn = None
        self.ns_value = 1.0390625
        self.optimizer.zeropower_fn = self._fixed_ns
        self.group = {"lr": 0.01, "matched_adamw_rms": 0.2, "ns_steps": 4, "ns_variant": "legacy",
                      "ns_coefficients": None, "ns_epsilon": 1e-7, "zero_rms_scale_mode": "zero",
                      "apply_lr_in_update": False}

    def _fixed_ns(self, value: torch.Tensor, steps: int) -> torch.Tensor:
        """Provide deterministic BF16 values without coupling this scale test to NS.

        Args:
            value: Batched logical matrix input.
            steps: Configured iteration count, unused by this isolated direction.

        Returns:
            A BF16 direction with the original logical shape.
        """
        del steps
        return torch.full_like(value, self.ns_value)

    def _updates(self, shapes=None) -> torch.Tensor:
        """Run the existing native update assembly with one model-provided transform.

        Args:
            shapes: Optional per-block logical scale dimensions.

        Returns:
            Restored BF16 update in the original parameter layout.
        """
        def transform(_name: str, value: torch.Tensor) -> NSInputTransform:
            """Split contiguous blocks while allowing an independent scale basis.

            Args:
                _name: Parameter name, unused by this generic transform.
                value: Complete NS input.

            Returns:
                Two reversible blocks with the requested scale shapes.
            """
            def restore(parts: list[torch.Tensor], output: torch.Tensor) -> None:
                """Restore distinct block updates without an extra rescaling pass.

                Args:
                    parts: Logical NS updates.
                    output: Original-layout destination.
                """
                output[:4].copy_(parts[0])
                output[4:].copy_(parts[1])
            return NSInputTransform([value[:4], value[4:]], restore, scale_shapes=shapes)
        self.optimizer.ns_transform_fn = transform
        inputs = {self.parameter: torch.ones(6, 2, dtype=torch.bfloat16)}
        return self.optimizer._compute_batched_ns_updates(  # pylint: disable=protected-access
            [self.parameter], inputs, self.group, no_shard=True)[self.parameter]

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_omitted_shapes_keep_each_blocks_original_scale(self):
        """
        Feature: Backward-compatible transform scaling.
        Description: Split one tensor into unequal row blocks without scale metadata.
        Expectation: Each block retains the native scale based on its own dimensions.
        """
        actual = self._updates()
        expected = torch.full_like(actual, self.ns_value)
        expected[:4].mul_(0.4)
        expected[4:].mul_(math.sqrt(2) * 0.2)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.dtype, torch.bfloat16)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_scale_metadata_selects_shape_without_changing_ns_blocks(self):
        """
        Feature: Model-defined RMS scale dimensions.
        Description: Use the last block shape for both unequal NS matrices.
        Expectation: The restored update uses the shared scale and stays BF16.
        """
        actual = self._updates([(2, 2), (2, 2)])
        expected = torch.full_like(actual, self.ns_value).mul_(math.sqrt(2) * 0.2)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.dtype, torch.bfloat16)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_shared_scale_is_applied_before_bf16_rounding(self):
        """
        Feature: One-pass BF16 RMS scaling.
        Description: Select a BF16 direction where post-restore compensation would round twice.
        Expectation: Output equals direct scaling and differs from the compensated native scale.
        """
        candidates = torch.arange(128, 256, dtype=torch.float32).to(torch.bfloat16) / 128
        direct = candidates * (math.sqrt(2) * 0.2)
        compensated = (candidates * 0.4) * math.sqrt(0.5)
        different = torch.nonzero(direct != compensated).flatten()
        self.assertGreater(different.numel(), 0)
        self.ns_value = float(candidates[int(different[0])])
        actual = self._updates([(2, 2), (2, 2)])
        self.assertTrue(torch.equal(actual, torch.full_like(actual, self.ns_value) * (math.sqrt(2) * 0.2)))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_invalid_scale_metadata_fails_before_ns(self):
        """
        Feature: Logical scale shape validation.
        Description: Supply wrong counts, ranks or nonpositive/noninteger matrix dimensions.
        Expectation: Invalid metadata produces a descriptive ValueError before NS execution.
        """
        for shapes in ([], [(2, 2)], [(2,), (2, 2)], [(2, 2, 2), (2, 2)],
                       [(0, 2), (2, 2)], [(-1, 2), (2, 2)], [(2.0, 2), (2, 2)],
                       [(True, 2), (2, 2)], [None, (2, 2)]):
            with self.subTest(shapes=shapes), self.assertRaisesRegex(ValueError, "scale_shapes"):
                self._updates(shapes)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_zero_rms_modes_and_lr_placement_remain_native(self):
        """
        Feature: Existing zero-RMS and learning-rate behavior.
        Description: Apply custom scale shapes with zero RMS, then fold LR into the update.
        Expectation: Native zero/use_lr fallbacks and LR placement remain unchanged.
        """
        self.group["matched_adamw_rms"] = 0
        self.assertTrue(torch.equal(self._updates([(2, 2)] * 2), torch.zeros(6, 2, dtype=torch.bfloat16)))
        self.group.update(zero_rms_scale_mode="use_lr", apply_lr_in_update=True)
        expected = torch.full((6, 2), self.ns_value, dtype=torch.bfloat16) * -self.group["lr"]
        self.assertTrue(torch.equal(self._updates([(2, 2)] * 2), expected))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_linear_batch_schedule_retains_group_order_and_empty_groups(self):
        """
        Feature: Existing HSDP assignment schedule.
        Description: Flatten mixed empty and nested parameter-group assignment batches.
        Expectation: Group order, assignment order and the maximum batch count are unchanged.
        """
        self.optimizer._hsdp_assignment_batches = {  # pylint: disable=protected-access
            1: {"batch_groups": [{"sub_batches": ["first", "second"]}, {}, {"sub_batches": ["third"]}]},
            2: {"batch_groups": []}, 3: None}
        batches, count = self.optimizer._get_linear_hsdp_batches(4)  # pylint: disable=protected-access
        self.assertEqual(batches, {0: [], 1: ["first", "second", "third"], 2: [], 3: []})
        self.assertEqual(count, 3)
        self.assertEqual(self.optimizer._get_linear_hsdp_batches(0), ({}, 0))  # pylint: disable=protected-access
