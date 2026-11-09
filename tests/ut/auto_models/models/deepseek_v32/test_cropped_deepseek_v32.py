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
"""Single-card DeepSeek-V3.2 crop: per-layer MLP pattern and routing constraints."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import DeepseekV32Config, DeepseekV32ForCausalLM
from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32MLP, DeepseekV32MoE

from tests.common.mark_utils import arg_mark

# Load the example by path: `examples` is also the RL package name that
# tests/ut/rl/conftest.py puts on sys.path, so importing it by name would
# bind whichever package a pytest session imports first.
_BUILDER = Path(__file__).resolve().parents[5] / "examples/training_demo/deepseek_v32/cropped_deepseek_v32.py"
_SPEC = importlib.util.spec_from_file_location("_cropped_deepseek_v32_example", _BUILDER)
_EXAMPLE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EXAMPLE)
build_cropped_deepseek_v32_config = _EXAMPLE.build_cropped_deepseek_v32_config


def _save_released_shape(directory: str) -> None:
    """Save a narrow configuration with the released depth pattern, routing and FP8 storage."""
    DeepseekV32Config.from_dict({
        "vocab_size": 96, "hidden_size": 48, "intermediate_size": 64, "moe_intermediate_size": 16,
        "num_hidden_layers": 5, "first_k_dense_replace": 3, "num_attention_heads": 4, "num_key_value_heads": 4,
        "n_shared_experts": 1, "n_routed_experts": 32, "n_group": 8, "topk_group": 4, "num_experts_per_tok": 8,
        "kv_lora_rank": 16, "q_lora_rank": 24, "qk_rope_head_dim": 8, "v_head_dim": 16, "qk_nope_head_dim": 16,
        "index_topk": 4, "index_head_dim": 16, "index_n_heads": 4,
        "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]},
    }).save_pretrained(directory)


class TestCroppedDeepseekV32Config(unittest.TestCase):
    """The crop must re-derive the dense/MoE layout instead of keeping the released one."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_crop_builds_one_dense_and_one_moe_layer(self):
        """
        Feature: Single-card DeepSeek-V3.2 crop.
        Description: Crop a released-shaped configuration to two layers, the first dense, with 16 experts.
        Expectation: Layer 1 is a 16-expert MoE block, widths are unchanged and the FP8 storage is dropped.
        """
        with tempfile.TemporaryDirectory() as directory:
            _save_released_shape(directory)
            config = build_cropped_deepseek_v32_config(
                directory, num_hidden_layers=2, first_k_dense_replace=1, n_routed_experts=16,
            )
        with torch.device("meta"):
            model = DeepseekV32ForCausalLM(config)
        self.assertIsInstance(model.model.layers[0].mlp, DeepseekV32MLP)
        self.assertIsInstance(model.model.layers[1].mlp, DeepseekV32MoE)
        self.assertEqual(model.model.layers[1].mlp.experts.gate_up_proj.shape[0], 16)
        self.assertEqual(config.hidden_size, 48)
        self.assertIsNone(getattr(config, "quantization_config", None))
        self.assertFalse(config.use_cache)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rejects_expert_counts_that_break_group_routing(self):
        """
        Feature: Single-card DeepSeek-V3.2 crop.
        Description: Request expert counts below two per group or not divisible by n_group.
        Expectation: The crop fails before any model is built.
        """
        with tempfile.TemporaryDirectory() as directory:
            _save_released_shape(directory)
            for experts in (8, 12):
                with self.subTest(n_routed_experts=experts), self.assertRaisesRegex(ValueError, "n_group=8"):
                    build_cropped_deepseek_v32_config(
                        directory, num_hidden_layers=2, first_k_dense_replace=1, n_routed_experts=experts,
                    )


if __name__ == "__main__":
    unittest.main()
