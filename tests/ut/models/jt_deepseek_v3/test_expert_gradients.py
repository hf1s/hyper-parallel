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
"""JT expert precision configuration reaches the shared local/EP kernel boundary."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3Experts
from tests.ut.models.jt_deepseek_v3.test_modeling_jt_deepseek_v3 import small_config


class TestExpertGradientPolicy(unittest.TestCase):
    """Keep the existing recipe by default and require a real boolean opt-in."""

    def test_default_and_enabled_kernel_arguments(self) -> None:
        """Both routed execution paths share the same explicit FP32 opt-in."""
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                config = small_config()
                if enabled:
                    config.expert_fp32_weight_gradients = True
                experts = JTDeepseekV3Experts(config)
                inputs = torch.zeros(4, config.hidden_size)
                counts = torch.ones(config.n_routed_experts, dtype=torch.int64)
                with patch("hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3.npu_grouped_swiglu") as kernel:
                    experts.forward_expert_major(inputs, counts)
                expected = {"fp32_weight_gradients": True} if enabled else {}
                kernel.assert_called_once_with(inputs, experts.gate_up_proj, experts.down_proj, counts, **expected)

    def test_invalid_option_is_rejected(self) -> None:
        """A string or integer must not silently enable a numerical recipe change."""
        for value in ("true", 1, None):
            with self.subTest(value=value):
                config = small_config()
                config.expert_fp32_weight_gradients = value
                with self.assertRaisesRegex(TypeError, "expert_fp32_weight_gradients"):
                    JTDeepseekV3Experts(config)


if __name__ == "__main__":
    unittest.main()
