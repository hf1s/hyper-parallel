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
"""The portable mhc_pre path must follow the mHC pre-processing reference."""

import unittest

import torch

from hyper_parallel.components.functional.mhc_pre import mhc_pre
from tests.common.mark_utils import arg_mark
from tests.common.multicore.mega_mhc_golden import _predict_mappings


class TestMhcPrePortable(unittest.TestCase):
    """Compare CPU mhc_pre with the independent mHC mapping reference."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_cpu_outputs_match_reference_mappings(self) -> None:
        """
        Feature: mhc_pre without the NPU operator.
        Description: Run mhc_pre on CPU streams in FP32 and BF16.
        Expectation: The coefficients match the reference mappings, the block input is the
            pre-weighted stream sum in the input dtype, and the residual buffer is the input.
        """
        num_stream, hidden, eps = 4, 8, 1e-6
        generator = torch.Generator().manual_seed(0)
        phi = torch.randn((num_stream + 2) * num_stream, num_stream * hidden, generator=generator) * 0.1
        alpha = torch.tensor([0.5, 1.5, 2.0])
        beta = torch.randn((num_stream + 2) * num_stream, generator=generator)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.randn(2, 3, num_stream * hidden, generator=generator).to(dtype)
                block_input, h_post, h_res, residual = mhc_pre(x, phi, alpha, beta, num_stream, 20, eps, eps)
                pre_mix, post_mix, res_mix = _predict_mappings(
                    x.unflatten(-1, (num_stream, hidden)), phi, alpha, beta,
                    hc_eps=eps, norm_eps=eps, num_iters=20)
                expected_input = (pre_mix.unsqueeze(-1) * x.float().unflatten(-1, (num_stream, hidden))).sum(2)
                torch.testing.assert_close(h_post, post_mix, msg="post coefficients differ")
                torch.testing.assert_close(h_res, res_mix, msg="residual mixing differs")
                torch.testing.assert_close(block_input, expected_input.to(dtype), msg="block input differs")
                self.assertIs(residual, x, "residual buffer is not the input streams")


if __name__ == "__main__":
    unittest.main()
