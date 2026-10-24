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
"""Check coordinate mappings independently of destination parameter loading."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.checkpoint_conversion.convert_mf_ckpt_to_hp import convert_model, convert_optimizer, write_artifacts
from scripts.checkpoint_conversion.resume_state import validate_materialization_config, validate_resume_state


class TestCheckpointConversionLayout(unittest.TestCase):
    """Check fused model/moment layouts and reject unmapped source state."""

    def test_dense_and_shared_swiglu_use_gate_then_up_rows(self):
        fused = np.array([[11, 12], [21, 22], [13, 14], [23, 24]], np.float32)
        expected = np.array([[11, 12], [13, 14], [21, 22], [23, 24]], np.float32)
        pairs = (
            ("decoder.layers.0.mlp", "model.layers.0.mlp"),
            ("decoder.layers.1.mlp.shared_experts", "model.layers.1.mlp.shared_experts"),
            ("mtp.layers.0.transformer_layer.mlp.shared_experts", "mtp.layers.0.transformer_layer.mlp.shared_experts"),
        )
        for source_prefix, target_prefix in pairs:
            source = {f"{source_prefix}.linear_fc1.weight": fused}
            for family, offset in (("adam_m", 100), ("adam_v", 200)):
                source[f"{family}.{source_prefix}.linear_fc1.weight"] = fused + offset
            with self.subTest(source=source_prefix):
                actual = convert_model(source, mlp_layout="swiglu")
                np.testing.assert_array_equal(actual[f"{target_prefix}.linear_fc1.weight"], expected)
                moments = convert_optimizer(source, mlp_layout="swiglu")
                for family, offset in (("adam_m", 100), ("adam_v", 200)):
                    actual_moment = moments[f"{family}.{target_prefix}.linear_fc1.weight"]
                    np.testing.assert_array_equal(actual_moment, expected + offset)
                separate = convert_model(source)
                np.testing.assert_array_equal(separate[f"{target_prefix}.gate_proj.weight"], expected[:2])
                np.testing.assert_array_equal(separate[f"{target_prefix}.up_proj.weight"], expected[2:])

    def test_expert_moments_use_model_router_shape(self):
        prefix = "decoder.layers.1.mlp"
        first = np.arange(48, dtype=np.float32).reshape(6, 8)
        second = np.arange(24, dtype=np.float32).reshape(8, 3)
        source = {f"{prefix}.router.weight": np.ones((2, 3), np.float32),
                  f"{prefix}.experts.weight1": first, f"{prefix}.experts.weight2": second,
                  f"muon_m.{prefix}.router.weight": np.array([0], np.float32),
                  f"muon_m.{prefix}.experts.weight1": first + 100,
                  f"muon_m.{prefix}.experts.weight2": second + 200}
        moments = convert_optimizer(source)
        actual_first = moments["muon_m.model.layers.1.mlp.experts.gate_up_proj"]
        actual_second = moments["muon_m.model.layers.1.mlp.experts.down_proj"]
        self.assertEqual(actual_first.shape, (2, 8, 3))
        self.assertEqual(actual_second.shape, (2, 3, 4))
        for expert in range(2):
            for hidden in range(3):
                for feature in range(8):
                    self.assertEqual(actual_first[expert, feature, hidden], first[expert * 3 + hidden, feature] + 100)
                for feature in range(4):
                    self.assertEqual(actual_second[expert, hidden, feature], second[expert * 4 + feature, hidden] + 200)

    def test_fused_qkv_distinguishes_placeholders_and_one_element_parameters(self):
        prefix = "decoder.layers.0.self_attention"
        keys = [f"{prefix}.linear_q_down_proj.weight", f"{prefix}.linear_kv_down_proj.weight"]
        source = {name: np.ones((2, 3), np.float32) for name in keys}
        source.update({f"muon_m.{name}": np.array([0], np.float32) for name in keys})
        output = "muon_m.model.layers.0.self_attn.linear_qkv.weight"
        self.assertEqual(convert_optimizer(source)[output].shape, (1,))
        source[f"muon_m.{keys[1]}"] = np.ones((2, 3), np.float32)
        with self.assertRaises(ValueError):
            convert_optimizer(source)
        scalar_source = {name: np.array([[1]], np.float32) for name in keys}
        scalar_source.update({f"adam_m.{name}": np.array([[2]], np.float32) for name in keys})
        actual = convert_optimizer(scalar_source)["adam_m.model.layers.0.self_attn.linear_qkv.weight"]
        np.testing.assert_array_equal(actual, np.array([[2], [2]], np.float32))

    def test_unmapped_or_incomplete_source_is_rejected(self):
        for source in (
                {"decoder.layers.2.input_layernorm.weight": np.ones(2, np.float32)},
                {"decoder.layers.0.self_attention.linear_q_down_proj.weight": np.ones((2, 3), np.float32)},
                {"unknown_parameter": np.ones(2, np.float32)},
        ):
            with self.subTest(keys=list(source)), self.assertRaises(ValueError):
                convert_model(source)
        with self.assertRaises(ValueError):
            convert_optimizer({"adam_m.unknown_parameter": np.ones(2, np.float32)})


class TestCheckpointResumeMetadata(unittest.TestCase):
    """Reject unsupported continuation instead of inventing its history."""

    def test_resume_counters_are_consistent(self):
        valid = {"global_step": 200, "step_num": 200, "global_batch_size": 1, "consumed_samples": 200,
                 "loss_scale": 1.0, "data_order": "sequential"}
        validate_resume_state(valid, 200)
        for field, value in (("global_step", 199), ("step_num", 201), ("consumed_samples", 0),
                             ("global_batch_size", 2), ("loss_scale", 128), ("loss_scale", True),
                             ("data_order", "random")):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_resume_state({**valid, field: value}, 200)

    def test_recipe_validation(self):
        state_dir = Path("offline")
        config = {
            "training": {"global_batch_size": 1, "micro_batch_size": 1},
            "accelerator": {"tp_size": 1},
            "dataloader": {"_target_": "hyper_parallel.data.batching.FixedBatchDataLoader", "sampler_type": "single"},
            "lr_scheduler": {"_target_": "hyper_parallel.components.optim.lr_scheduler.MultiLRScheduler",
                             "lr_decay_style": "constant", "lr_config": {"lr_warmup_steps": 0}},
            "model": {"reference_weights": str(state_dir)},
            "checkpoint": {"save_optimizer": True, "save_train_state": True},
        }
        validate_materialization_config(config, state_dir)
        cases = (("training", "global_batch_size", 2), ("accelerator", "tp_size", 2),
                 ("dataloader", "sampler_type", "cyclic"), ("dataloader", "num_workers", 2),
                 ("dataloader", "data_rearrange_map", "mapping.npy"),
                 ("lr_scheduler", "lr_decay_style", "cosine"), ("model", "reference_weights", "other"),
                 ("checkpoint", "save_optimizer", False))
        for section, key, value in cases:
            invalid = copy.deepcopy(config)
            invalid[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_materialization_config(invalid, state_dir)

    def test_invalid_conversion_metadata_writes_no_artifacts(self):
        model = {"weight": np.ones(2, np.float32)}
        optimizer = {"adam_m.weight": np.ones(2, np.float32)}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step in (np.array(-1), np.array(200.5), np.array(True)):
                with self.subTest(step=step), self.assertRaises(ValueError):
                    write_artifacts({"step_num": step, "global_batch_size": np.array(1)}, model, optimizer,
                                    root / "output", source_checkpoint=root / "source.safetensors")
                self.assertFalse((root / "output").exists())
