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
"""Read existing manifest-named states without rewriting their tensor archives."""

import copy
from contextlib import contextmanager
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scripts.checkpoint_conversion.manifest_input import load_conversion_manifest, map_manifest_slots
from scripts.checkpoint_conversion.materialize_hp_native_checkpoint import load_optimizer_npz, main


@contextmanager
def _temporary_root():
    with tempfile.TemporaryDirectory() as temporary:
        yield Path(temporary)


class TestCheckpointManifestInput(unittest.TestCase):
    """Cover the persisted experiment schema, corruption, and key-map validation."""

    def setUp(self) -> None:
        """Create small archives using the actual step396 manifest schema."""
        self.root = self.enterContext(_temporary_root())
        self.weights = {"weight": np.array([[1.0017, .9131]], np.float32)}
        self.moments = {"weight::exp_avg": np.array([[.2, .3]], np.float32),
                        "weight::exp_avg_sq": np.array([[.4, .5]], np.float32)}
        np.savez(self.root / "model.npz", **self.weights)
        np.savez(self.root / "optimizer.npz", **self.moments)
        self.manifest = {
            "format_version": 1, "optimizer_type": "AdamW", "completed_steps": 396,
            "optimizer_global_step": 396, "mapped_model_tensors": 1, "optimizer_state_tensors": 2,
            "optimizer_manifest": [{"name": "weight", "use_muon": False,
                                    "state": {"exp_avg": "weight::exp_avg", "exp_avg_sq": "weight::exp_avg_sq"}}],
            "sha256": {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                       for name in ("model.npz", "optimizer.npz")},
        }
        self._write_manifest(self.manifest)

    def _write_manifest(self, manifest):
        (self.root / "CONVERSION.json").write_text(json.dumps(manifest), encoding="utf8")

    def test_actual_schema_restores_without_rewriting_archives(self):
        manifest = load_conversion_manifest(self.root, 396)
        model = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters())
        trainer = SimpleNamespace(base=SimpleNamespace(model=model, optimizer=optimizer))
        self.assertEqual(load_optimizer_npz(trainer, self.root, 396, optimizer_manifest=manifest), (2, 2))
        torch.testing.assert_close(model.weight, torch.from_numpy(self.weights["weight"]), rtol=0, atol=0)
        np.testing.assert_array_equal(optimizer.state[model.weight]["exp_avg"].numpy(), self.moments["weight::exp_avg"])
        self.assertEqual(optimizer.state[model.weight]["step"].item(), 396)
        for name, digest in self.manifest["sha256"].items():
            self.assertEqual(hashlib.sha256((self.root / name).read_bytes()).hexdigest(), digest)
        self.assertFalse((self.root / "state.json").exists())

    def test_counter_mismatch_is_rejected(self):
        for field, value in (("completed_steps", 395), ("optimizer_global_step", 0),
                             ("format_version", True), ("optimizer_type", "SGD")):
            invalid = {**self.manifest, field: value}
            self._write_manifest(invalid)
            with self.subTest(field=field), self.assertRaises(ValueError):
                load_conversion_manifest(self.root, 396)

    def test_corrupted_archive_is_rejected(self):
        with (self.root / "optimizer.npz").open("ab") as stream:
            stream.write(b"unexpected trailing bytes")
        with self.assertRaises(ValueError):
            load_conversion_manifest(self.root, 396)

    def test_optimizer_parameter_requires_a_model_tensor(self):
        invalid = copy.deepcopy(self.manifest)
        invalid["optimizer_manifest"][0]["name"] = "missing_model_weight"
        self._write_manifest(invalid)
        with self.assertRaisesRegex(ValueError, "no corresponding model tensor"):
            load_conversion_manifest(self.root, 396)

    def test_no_missing_or_ignored_slot_is_accepted(self):
        invalid_cases = []
        invalid = copy.deepcopy(self.manifest)
        del invalid["optimizer_manifest"][0]["state"]["exp_avg_sq"]
        invalid_cases.append(invalid)
        invalid = copy.deepcopy(self.manifest)
        invalid["optimizer_manifest"][0]["state"]["exp_avg_sq"] = "weight::exp_avg"
        invalid_cases.append(invalid)
        invalid = copy.deepcopy(self.manifest)
        invalid["optimizer_manifest"].append(copy.deepcopy(invalid["optimizer_manifest"][0]))
        invalid_cases.append(invalid)
        invalid = copy.deepcopy(self.manifest)
        invalid["optimizer_manifest"][0]["use_muon"] = True
        invalid_cases.append(invalid)
        for index, invalid in enumerate(invalid_cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                map_manifest_slots(self.moments, invalid)
        with self.assertRaises(ValueError):
            map_manifest_slots({**self.moments, "unknown::exp_avg": np.zeros(1, np.float32)}, self.manifest)

    def test_mapping_does_not_eagerly_read_arrays(self):
        class _CountingMapping(dict):
            reads = 0

            def __getitem__(self, key: str) -> np.ndarray:
                """Count array access independently from name enumeration."""
                self.reads += 1
                return super().__getitem__(key)

        arrays = _CountingMapping(self.moments)
        mapped = map_manifest_slots(arrays, self.manifest)
        self.assertEqual(arrays.reads, 0)
        self.assertEqual(set(mapped), {"adam_m.weight", "adam_v.weight"})
        self.assertIs(mapped["adam_m.weight"], self.moments["weight::exp_avg"])
        self.assertEqual(arrays.reads, 1)

    def test_manifest_input_never_infers_missing_resume_metadata(self):
        arguments = ["materialize", "recipe.yaml", "--state-dir", str(self.root),
                     "--output", str(self.root / "output"), "--step", "396", "--optimizer-key-format", "manifest"]
        with patch("sys.argv", arguments), self.assertRaisesRegex(ValueError, "requires explicit --resume-state"):
            main()
        self.assertFalse((self.root / "output").exists())

    def test_family_prefix_input_cannot_override_its_stored_counters(self):
        arguments = ["materialize", "recipe.yaml", "--state-dir", str(self.root),
                     "--output", str(self.root / "output"), "--step", "396", "--resume-state", "other.json"]
        with patch("sys.argv", arguments), self.assertRaisesRegex(ValueError, "applies only to manifest"):
            main()
