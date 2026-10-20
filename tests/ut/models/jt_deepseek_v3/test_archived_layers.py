# Copyright 2026 Huawei Technologies Co., Ltd.
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
"""Archived norm and output head: FP32 gradient joins, cast boundaries and the scoped head GEMM."""
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.models.jt_deepseek_v3 import archived_layers, legacy_gemm
from hyper_parallel.models.jt_deepseek_v3.archived_layers import ArchivedHead, ArchivedNorm
from tests.common.mark_utils import arg_mark

RESIDUAL_GRADIENT = 2.0 ** -8


def rms_norm(values: torch.Tensor, weight: torch.Tensor, *, epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU stand-in for ``torch_npu.npu_rms_norm``: normalized values and the reciprocal RMS."""
    rstd = torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon)
    return values * rstd * weight, rstd


def rms_norm_backward(gradient: torch.Tensor, values: torch.Tensor, weight: torch.Tensor,
                      rstd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU stand-in for ``npu_rms_norm_backward``: the exact RMSNorm input and weight derivatives."""
    normalized = values * rstd
    scaled = gradient * weight
    input_gradient = rstd * (scaled - normalized * (scaled * normalized).mean(-1, keepdim=True))
    return input_gradient, (gradient * normalized).reshape(-1, values.shape[-1]).sum(0)


class SetterOnlyConfig:
    """TorchNPU 2.12 ``npu.config``: ``allow_internal_format`` can be written but not read back."""

    def __init__(self, runtime: "FakeNpuRuntime") -> None:
        """Forward every write to ``runtime``."""
        self._runtime = runtime

    def _write(self, enabled: bool) -> None:
        self._runtime.set_internal_format(enabled)

    allow_internal_format = property(fset=_write)


class FakeNpuRuntime:
    """Process-global TorchNPU switches read and written around the archived head GEMM."""

    def __init__(self, *, jit_compile: bool = False, internal_option: Optional[bytes] = b"disable") -> None:
        """Start from the caller's compile mode and internal-format option."""
        self.jit_compile = jit_compile
        self.internal_option = internal_option
        self.npu = SimpleNamespace(is_jit_compile_false=lambda: not self.jit_compile,
                                   set_compile_mode=self.set_compile_mode, config=SetterOnlyConfig(self),
                                   device=lambda _: nullcontext())
        self._C = SimpleNamespace(_npu_getOption=self.get_option)

    def state(self) -> tuple[bool, Optional[bytes]]:
        """Return the compile mode and the backing internal-format option."""
        return self.jit_compile, self.internal_option

    def get_option(self, name: str) -> Optional[bytes]:
        """Read a backing runtime option; only the internal-format option is modelled."""
        if name != "ALLOW_INTERNAL_FORMAT":
            raise KeyError(name)
        return self.internal_option

    def set_compile_mode(self, *, jit_compile: bool) -> None:
        """Switch JIT compilation for the whole process."""
        self.jit_compile = jit_compile

    def set_internal_format(self, enabled: bool) -> None:
        """Back the setter-only ``npu.config.allow_internal_format`` attribute."""
        self.internal_option = b"enable" if enabled else b"disable"


class RecordingKernel:
    """NZ GEMM stand-in returning BF16 ``gradient.T @ inputs`` and recording the switches it ran under."""

    def __init__(self, runtime: FakeNpuRuntime, error: Optional[Exception] = None) -> None:
        """Observe ``runtime``; raise ``error`` instead of computing when it is given."""
        self.runtime = runtime
        self.error = error
        self.seen = []

    def __call__(self, gradient: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Record the runtime switches, then fail or return the product rounded once to BF16."""
        self.seen.append(self.runtime.state())
        if self.error is not None:
            raise self.error
        return (gradient.double().T @ inputs.double()).to(torch.bfloat16)


@contextmanager
def archived_gemm(runtime: FakeNpuRuntime, kernel: RecordingKernel) -> Iterator[None]:
    """Install ``runtime`` as TorchNPU (``torch.npu`` is ``torch_npu.npu``) and ``kernel`` as the NZ GEMM."""
    with patch.object(legacy_gemm, "torch_npu", runtime), patch.object(torch, "npu", runtime.npu, create=True), \
            patch.object(legacy_gemm, "_nz_matmul", kernel):
        yield


class TestArchivedNorm(unittest.TestCase):
    """RMSNorm arithmetic stays FP32 until the declared activation boundary."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_residual_norm_rounds_once_after_joining_both_input_gradients(self):
        """The FP32 RMSNorm and residual input derivatives are added before the single BF16 rounding."""
        values = torch.tensor([[1.0, -1.0, 1.0, -1.0]], dtype=torch.bfloat16, requires_grad=True)
        weight = torch.full((4,), 1.0 + 2.0 ** -8, dtype=torch.float64, requires_grad=True)
        norm = ArchivedNorm(4, 0.0, residual=True)
        with torch.no_grad():
            norm.weight.copy_(weight)
        with patch.object(archived_layers.torch_npu, "npu_rms_norm", rms_norm), \
                patch.object(torch.ops.npu, "npu_rms_norm_backward", rms_norm_backward, create=True):
            normalized, residual = norm(values)
            (normalized.float().sum() + residual.float().mul(RESIDUAL_GRADIENT).sum()).backward()
        # Unit RMS makes the norm derivative exactly 1 + 2**-8, so the joined derivative 1 + 2**-7 is a BF16
        # value; rounding the norm derivative first ties it to 1, and adding 2**-8 in BF16 ties back to 1.
        reference = values.detach().double().requires_grad_()
        norm_part, weight_gradient = torch.autograd.grad(
            rms_norm(reference, weight, epsilon=0.0)[0].sum(), (reference, weight))
        residual_part = torch.full_like(norm_part, RESIDUAL_GRADIENT)
        joined = (norm_part + residual_part).to(torch.bfloat16)
        rounded_first = norm_part.to(torch.bfloat16) + residual_part.to(torch.bfloat16)
        self.assertFalse(torch.equal(joined, rounded_first))
        torch.testing.assert_close(values.grad, joined, rtol=0, atol=0)
        torch.testing.assert_close(norm.weight.grad, weight_gradient.float(), rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_norm_casts_fp32_result_once_to_the_selected_output_dtype(self):
        """Without a residual the FP32 result is cast to ``output_dtype`` when given, else to the input dtype."""
        # RMS 2 normalizes to +-1, so the result is +-weight: 1 + 2**-8 + 2**-10 is exact in FP32 and
        # rounds up to 1 + 2**-7 in BF16.
        values = torch.tensor([[2.0, -2.0, 2.0, -2.0]])
        weight = torch.full((4,), 1.0 + 2.0 ** -8 + 2.0 ** -10)
        cases = ((values, torch.bfloat16, torch.bfloat16), (values, None, torch.float32),
                 (values.bfloat16(), None, torch.bfloat16))
        for inputs, output_dtype, dtype in cases:
            with self.subTest(input_dtype=inputs.dtype, output_dtype=output_dtype):
                norm = ArchivedNorm(4, 0.0, output_dtype=output_dtype)
                with torch.no_grad():
                    norm.weight.copy_(weight)
                with patch.object(archived_layers.torch_npu, "npu_rms_norm", rms_norm):
                    actual = norm(inputs)
                expected = rms_norm(inputs.double(), weight.double(), epsilon=0.0)[0].to(dtype)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class TestArchivedHead(unittest.TestCase):
    """The output head keeps ordinary derivatives except for its archived weight-gradient GEMM."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_head_sums_per_use_archived_weight_gradients_in_fp32(self):
        """Each use's BF16 weight derivative comes from the NZ GEMM and is summed in the FP32 parameter."""
        head = ArchivedHead(16, 32, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.arange(512.0).reshape(32, 16) % 9 - 4)
        main = torch.eye(16, dtype=torch.bfloat16).unsqueeze(0).requires_grad_()
        mtp = (torch.eye(16, dtype=torch.bfloat16) * 2.0 ** -8).unsqueeze(0).requires_grad_()
        runtime = FakeNpuRuntime()
        kernel = RecordingKernel(runtime)
        with archived_gemm(runtime, kernel):
            (head(main).float().sum() + head(mtp).float().sum()).backward()
        # Unit upstream gradients make each use's weight derivative ones.T @ (s * I) = s: summed in FP32 the
        # MTP share 2**-8 survives next to the main share 1, whereas a BF16 sum ties back to 1.
        upstream = torch.ones(16, 32, dtype=torch.float64)
        main_part, mtp_part = ((upstream.T @ values.detach().double()[0]).to(torch.bfloat16) for values in (main, mtp))
        self.assertEqual(len(kernel.seen), 2)
        self.assertFalse(torch.equal((main_part + mtp_part).float(), main_part.float() + mtp_part.float()))
        torch.testing.assert_close(head.weight.grad, main_part.float() + mtp_part.float(), rtol=0, atol=0)
        input_gradient = (upstream @ head.weight.detach().bfloat16().double()).to(torch.bfloat16).unsqueeze(0)
        torch.testing.assert_close(main.grad, input_gradient, rtol=0, atol=0)
        torch.testing.assert_close(mtp.grad, input_gradient, rtol=0, atol=0)


class TestArchivedHeadGemm(unittest.TestCase):
    """The NZ head GEMM scopes the global compiler switches it needs."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_head_gradient_restores_compile_and_format_switches_after_success_and_failure(self):
        """The kernel runs with JIT compilation and internal formats; the caller's switches return either way."""
        gradient = torch.ones(32, 16, dtype=torch.bfloat16)
        inputs = torch.ones(32, 48, dtype=torch.bfloat16)
        for jit_compile, option in ((False, b"disable"), (True, b"enable")):
            for error in (None, RuntimeError("MatMulV2/NZ: CANN status 507011")):
                with self.subTest(jit_compile=jit_compile, option=option, kernel_fails=error is not None):
                    runtime = FakeNpuRuntime(jit_compile=jit_compile, internal_option=option)
                    kernel = RecordingKernel(runtime, error)
                    with archived_gemm(runtime, kernel):
                        if error is None:
                            legacy_gemm.head_weight_gradient(gradient, inputs)
                        else:
                            with self.assertRaisesRegex(RuntimeError, "507011"):
                                legacy_gemm.head_weight_gradient(gradient, inputs)
                    self.assertEqual(kernel.seen, [(True, b"enable")])
                    self.assertEqual(runtime.state(), (jit_compile, option))


if __name__ == "__main__":
    unittest.main()
