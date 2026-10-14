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
"""CPU tests for the NPU matmul K-shift policy in full-determinism seeding.

Merged from two independent changes:
* "leave K-shift to the process environment" (determinism never implies a matmul tiling), and
* "allow an explicit K-shift policy" (cross-framework bitwise parity).
The merged contract is: ``close_matmul_k_shift=None`` (default) never touches
``CLOSE_MATMUL_K_SHIFT``; an explicit ``True``/``False`` sets it to ``1``/``0`` on the NPU path.
"""

import os
from types import SimpleNamespace
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

from hyper_parallel.trainer.runtime import random as runtime_random
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
class TestMatmulDeterminism(unittest.TestCase):
    """Exercise the real seeding policy while replacing hardware operations."""

    def setUp(self) -> None:
        """Isolate environment, RNG state, backend flags, and device seed calls."""
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.context.enter_context(patch.dict(os.environ, {"CLOSE_MATMUL_K_SHIFT": "inherited"}))
        self.context.enter_context(patch.object(runtime_random, "IS_NPU_AVAILABLE", True))
        self.context.enter_context(patch.object(runtime_random.random, "seed"))
        self.context.enter_context(patch.object(runtime_random.np.random, "seed"))
        self.context.enter_context(patch.object(runtime_random.torch, "manual_seed"))
        self.context.enter_context(patch.object(runtime_random.torch.cuda, "manual_seed"))
        self.context.enter_context(patch.object(runtime_random.torch.cuda, "manual_seed_all"))
        self.context.enter_context(patch.object(runtime_random.torch.backends, "cudnn", SimpleNamespace()))
        self.context.enter_context(patch.object(
            runtime_random.torch, "npu", SimpleNamespace(manual_seed=Mock(), manual_seed_all=Mock()), create=True))
        self.deterministic = self.context.enter_context(
            patch.object(runtime_random.torch, "use_deterministic_algorithms"))

    def test_default_keeps_inherited_k_shift(self) -> None:
        """Full determinism alone never rewrites the inherited K-shift policy."""
        runtime_random.enable_full_determinism(42)
        self.assertEqual(os.environ["CLOSE_MATMUL_K_SHIFT"], "inherited")
        self.deterministic.assert_called_once_with(True, warn_only=True)

    def test_default_leaves_k_shift_absent(self) -> None:
        """When the process never set K-shift, seeding does not invent a value."""
        os.environ.pop("CLOSE_MATMUL_K_SHIFT", None)
        runtime_random.enable_full_determinism(42)
        self.assertNotIn("CLOSE_MATMUL_K_SHIFT", os.environ)

    def test_explicit_false_overrides_inherited_value(self) -> None:
        """An explicit compatibility policy wins over the inherited value."""
        runtime_random.set_seed(42, True, close_matmul_k_shift=False)
        self.assertEqual(os.environ["CLOSE_MATMUL_K_SHIFT"], "0")
        self.deterministic.assert_called_once_with(True, warn_only=True)
        runtime_random.torch.npu.manual_seed.assert_called_once_with(42)

    def test_explicit_true_sets_k_shift(self) -> None:
        """An explicit default selection sets the matmul tiling policy."""
        runtime_random.set_seed(42, True, close_matmul_k_shift=True)
        self.assertEqual(os.environ["CLOSE_MATMUL_K_SHIFT"], "1")

    def test_non_npu_backend_preserves_npu_environment(self) -> None:
        """CPU and CUDA seeding do not rewrite an NPU-specific policy."""
        with patch.object(runtime_random, "IS_NPU_AVAILABLE", False):
            runtime_random.set_seed(42, True, close_matmul_k_shift=False)
        self.assertEqual(os.environ["CLOSE_MATMUL_K_SHIFT"], "inherited")
        self.deterministic.assert_called_once_with(True, warn_only=True)

    def test_regular_seeding_does_not_apply_deterministic_policy(self) -> None:
        """The policy only applies when full determinism is requested."""
        with patch.object(runtime_random, "set_seed_func") as ordinary_seed:
            runtime_random.set_seed(42, False, close_matmul_k_shift=False)
        ordinary_seed.assert_called_once_with(42)
        self.assertEqual(os.environ["CLOSE_MATMUL_K_SHIFT"], "inherited")
        self.deterministic.assert_not_called()

    def test_missing_seed_keeps_existing_noop(self) -> None:
        """Calling the utility without a seed does not mutate process policy."""
        runtime_random.set_seed(None, True, close_matmul_k_shift=False)
        self.assertEqual(os.environ["CLOSE_MATMUL_K_SHIFT"], "inherited")
        self.deterministic.assert_not_called()
