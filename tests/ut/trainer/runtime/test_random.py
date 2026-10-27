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
"""Unit tests for Trainer determinism and backend precision controls."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hyper_parallel.trainer.runtime import random as random_module
from tests.common.mark_utils import arg_mark

MODULE = "hyper_parallel.trainer.runtime.random"


class TestEnableFullDeterminism(unittest.TestCase):
    """Verify CLOSE_MATMUL_K_SHIFT is left to the process environment, not HP."""

    def setUp(self) -> None:
        """Isolate each test from any CLOSE_MATMUL_K_SHIFT set outside this process."""
        self._saved = os.environ.pop("CLOSE_MATMUL_K_SHIFT", None)

    def tearDown(self) -> None:
        """Restore whatever CLOSE_MATMUL_K_SHIFT state existed before the test."""
        os.environ.pop("CLOSE_MATMUL_K_SHIFT", None)
        if self._saved is not None:
            os.environ["CLOSE_MATMUL_K_SHIFT"] = self._saved

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_npu_path_does_not_set_k_shift_when_absent(self) -> None:
        """Leave CLOSE_MATMUL_K_SHIFT unset on NPU when the caller never set it."""
        with patch(f"{MODULE}.IS_NPU_AVAILABLE", True), patch(f"{MODULE}.torch"):
            random_module.enable_full_determinism(42)

        self.assertNotIn("CLOSE_MATMUL_K_SHIFT", os.environ)
        # The other NPU-specific determinism knob this function owns is untouched.
        self.assertEqual(os.environ["NCCL_DETERMINISTIC"], "true")

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_npu_path_preserves_a_caller_set_value(self) -> None:
        """Never clobber a value the caller (or CANN's own default) already set."""
        os.environ["CLOSE_MATMUL_K_SHIFT"] = "0"

        with patch(f"{MODULE}.IS_NPU_AVAILABLE", True), patch(f"{MODULE}.torch"):
            random_module.enable_full_determinism(42)

        self.assertEqual(os.environ["CLOSE_MATMUL_K_SHIFT"], "0")

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_non_npu_path_is_unaffected(self) -> None:
        """Take no position on CLOSE_MATMUL_K_SHIFT at all off the NPU path."""
        with patch(f"{MODULE}.IS_NPU_AVAILABLE", False), patch(f"{MODULE}.torch"):
            random_module.enable_full_determinism(42)

        self.assertNotIn("CLOSE_MATMUL_K_SHIFT", os.environ)
        self.assertEqual(os.environ["NCCL_DETERMINISTIC"], "1")


class TestHighPrecision(unittest.TestCase):
    """Verify backend precision policy without initializing an accelerator."""

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_npu_disables_hf32(self) -> None:
        """Disable HF32 through the supported NPU option.

        Feature: NPU high precision policy.
        Description: Start with HF32 enabled or disabled and apply the policy.
        Expectation: HF32 is disabled and CUDA-only options are not introduced.
        """
        for initial in (True, False):
            with self.subTest(initial_allow_hf32=initial):
                matmul = SimpleNamespace(allow_hf32=initial)
                backend = SimpleNamespace(npu=SimpleNamespace(matmul=matmul))
                with (
                    patch.object(random_module, "IS_NPU_AVAILABLE", True),
                    patch.object(random_module, "IS_CUDA_AVAILABLE", False),
                    patch.object(random_module, "torch", backend),
                ):
                    random_module.enable_high_precision_for_bf16()
                expected = {"allow_hf32": False}
                self.assertEqual(vars(matmul), expected, f"expected={expected}, got={vars(matmul)}")

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_cuda_disables_reduced_precision(self) -> None:
        """Preserve the CUDA precision policy.

        Feature: CUDA high precision policy.
        Description: Apply the policy with both reduced precision options enabled.
        Expectation: TF32 and reduced precision BF16 reductions are disabled without accessing an NPU API.
        """
        matmul = SimpleNamespace(allow_tf32=True, allow_bf16_reduced_precision_reduction=True)
        backend = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul)))
        with (
            patch.object(random_module, "IS_NPU_AVAILABLE", False),
            patch.object(random_module, "IS_CUDA_AVAILABLE", True),
            patch.object(random_module, "torch", backend),
        ):
            random_module.enable_high_precision_for_bf16()
        expected = {"allow_tf32": False, "allow_bf16_reduced_precision_reduction": False}
        self.assertEqual(vars(matmul), expected, f"expected={expected}, got={vars(matmul)}")

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_cpu_does_not_access_accelerator_options(self) -> None:
        """Keep CPU-only setup independent of accelerator option namespaces.

        Feature: CPU high precision policy.
        Description: Apply the policy when neither accelerator is available.
        Expectation: No accelerator attribute is accessed.
        """
        with (
            patch.object(random_module, "IS_NPU_AVAILABLE", False),
            patch.object(random_module, "IS_CUDA_AVAILABLE", False),
            patch.object(random_module, "torch", object()),
        ):
            self.assertIsNone(random_module.enable_high_precision_for_bf16())
