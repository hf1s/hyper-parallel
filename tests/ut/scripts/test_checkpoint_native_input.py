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
"""Check actual native producer field schema with small CPU-only synthetic arrays."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file

from scripts.checkpoint_conversion.convert_mf_ckpt_to_hp import (
    convert_model, convert_optimizer, load_safetensors, main, write_artifacts,
)


class TestCheckpointNativeInput(unittest.TestCase):
    """Exercise the public CLI, including rejection before any artifact is written."""

    def setUp(self) -> None:
        """Build synthetic values using the captured producer's names, shapes and dtypes."""
        self.schema = json.loads((Path(__file__).parent / "fixtures/mf_native_train_cell.json").read_text())
        self.source = {}
        dtypes = {"F32": np.float32, "I32": np.int32, "I64": np.int64}
        for index, (name, info) in enumerate(self.schema["tensors"].items()):
            shape = tuple(size // 16 if size > 8 else size for size in info["shape"])
            zero = name.endswith((".seed", ".offset", ".fi_accu", ".expert_load"))
            self.source[name] = np.full(shape, 0 if zero else (index + 1) / 256, dtypes[info["dtype"]])
        for name, value in (("step_num", 200), ("global_step", 200), ("epoch_num", 1),
                            ("global_batch_size", 1), ("loss_scale", 1), ("scale_sense", 1)):
            self.source[name].fill(value)

    @contextmanager
    def _invoke(self, source, *, metadata=None, mode=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, output = root / "native.safetensors", root / "converted"
            save_file(source, str(checkpoint), metadata=self.schema["metadata"] if metadata is None else metadata)
            before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            argv = ["convert", "--ckpt", str(checkpoint), "--out-dir", str(output), "--mlp-layout", "swiglu"]
            if mode is not None:
                argv += ["--source-format", mode]
            with patch.object(sys, "argv", argv):
                main()
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), before)
            yield output

    def _reject(self, source, pattern, *, metadata=None, mode=None) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, output = root / "native.safetensors", root / "converted"
            save_file(source, str(checkpoint), metadata=self.schema["metadata"] if metadata is None else metadata)
            argv = ["convert", "--ckpt", str(checkpoint), "--out-dir", str(output)]
            if mode is not None:
                argv += ["--source-format", mode]
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, pattern):
                main()
            self.assertFalse(output.exists())

    @staticmethod
    def _family_source(source):
        aliases = {"global_step", "scale_sense", "lm_loss", "mtp_loss", "aux_loss"}
        return {name: value for name, value in source.items() if name not in aliases
                and not name.endswith((".seed", ".offset", ".fi_accu", ".expert_load"))}

    def test_native_schema_preserves_all_model_and_optimizer_arrays(self) -> None:
        """Runtime fields do not alter any of the 131 layout-converted payload arrays."""
        with self._invoke(self.source) as native:
            with self._invoke(self._family_source(self.source), metadata={}) as family:
                count = 0
                for filename, expected_count in (("model.npz", 45), ("optimizer.npz", 86)):
                    with np.load(native / filename, allow_pickle=False) as actual:
                        with np.load(family / filename, allow_pickle=False) as expected:
                            self.assertEqual(set(actual.files), set(expected.files))
                            self.assertEqual(len(actual.files), expected_count)
                            for name in actual.files:
                                np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
                                count += 1
                self.assertEqual(count, 131)
                state = json.loads((native / "state.json").read_text())
                self.assertEqual((state["global_step"], state["consumed_samples"], state["loss_scale"]), (200, 200, 1))
                manifest = json.loads((native / "CONVERSION.json").read_text())
                self.assertEqual(manifest["source_tensor_count"], len(self.source))
                self.assertEqual(manifest["source_input"]["source_format"], "mf-native")

    def test_counter_aliases_must_be_equal_nonnegative_integers(self) -> None:
        """A displayed step must not hide a stale or invalid optimizer step."""
        for value in (np.array([201], np.int32), np.array([-1], np.int32), np.array([200.0], np.float32)):
            with self.subTest(value=value):
                self._reject({**self.source, "global_step": value}, r"^Native.*(step|integer)")

    def test_scale_aliases_must_agree_and_be_unit(self) -> None:
        """Reject inconsistent wrapper scale and unsupported dynamic scaling."""
        for scale, sense in ((1.0, 2.0), (2.0, 2.0), (1.0, float("nan"))):
            with self.subTest(scale=scale, sense=sense):
                source = {**self.source, "loss_scale": np.array(scale, np.float32),
                          "scale_sense": np.array(sense, np.float32)}
                self._reject(source, r"^Native.*(scale|scaling)")

    def test_epoch_metadata_is_checked_before_writing_payloads(self) -> None:
        """An invalid informational epoch must not leave two large NPZ artifacts behind."""
        for value in (np.array([1, 2], np.int64), np.array(-1, np.int64), np.array(1.0, np.float32)):
            with self.subTest(value=value):
                self._reject({**self.source, "epoch_num": value}, r"^Native.*epoch_num")

    def test_invalid_provenance_metadata_is_rejected_before_file_creation(self) -> None:
        """A new optional input report cannot introduce late partial writes."""
        source = self._family_source(self.source)
        model, optimizer = convert_model(source), convert_optimizer(source)
        reports = ({}, {"source_tensor_count": False},
                   {"source_tensor_count": len(source), "value": np.array([1])})
        for report in reports:
            with self.subTest(report=report), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "converted"
                with self.assertRaisesRegex(ValueError, "metadata"):
                    write_artifacts(source, model, optimizer, output,
                                    source_checkpoint=Path(directory) / "source.safetensors", input_report=report)
                self.assertFalse(output.exists())

    def test_nonzero_rng_and_unconsumed_router_statistics_are_rejected(self) -> None:
        """Never discard continuation-relevant state merely because its name is known."""
        for name in ("decoder.layers.0.hidden_states_dropout.seed", "embedding.embedding_dropout.offset",
                     "decoder.layers.1.mlp.router.fi_accu"):
            with self.subTest(name=name):
                value = np.ones_like(self.source[name])
                self._reject({**self.source, name: value}, r"(Nonzero RNG|router statistics must be zero)")

    def test_router_statistic_shape_and_rng_type_are_checked(self) -> None:
        """Known field names do not bypass their producer's type and shape contracts."""
        self._reject({**self.source, "decoder.layers.1.mlp.router.fi_accu": np.zeros(1, np.float32)},
                     "router statistics must match")
        self._reject({**self.source, "embedding.embedding_dropout.seed": np.array(0, np.float32)},
                     "must be an integer")

    def test_optional_consumed_expert_load_is_accepted_only_when_zero(self) -> None:
        """A zero known callback counter is safe; a pending bias update is not."""
        name = "decoder.layers.1.mlp.router.expert_load"
        source = {**self.source, name: np.zeros(8, np.float32)}
        with self._invoke(source):
            pass
        self._reject({**source, name: np.ones(8, np.float32)}, "router statistics must be zero")

    def test_monitor_fields_must_be_finite_single_element_fp32(self) -> None:
        """Finite nonzero monitoring values are covered by the successful native fixture."""
        for value in (np.array([np.nan], np.float32), np.zeros(2, np.float32), np.array([1], np.int32)):
            with self.subTest(value=value):
                self._reject({**self.source, "lm_loss": value}, r"^Native checkpoint lm_loss")

    def test_unknown_scalar_and_foreign_rng_scope_remain_unmapped(self) -> None:
        """Do not revive the old behavior of ignoring arbitrary small tensors."""
        for name in ("foreign.counter", "foreign.seed"):
            with self.subTest(name=name):
                self._reject({**self.source, name: np.array(0, np.float32)}, "Unsupported or incomplete source mapping")

    def test_family_mode_and_non_ms_headers_do_not_hide_native_fields(self) -> None:
        """Producer handling is explicit and cannot be inferred from convenient key suffixes."""
        self._reject(self.source, "Unsupported or incomplete source mapping", mode="family-prefix")
        self._reject(self.source, "Unsupported or incomplete source mapping", metadata={"format": "pt"})
        self._reject(self.source, "mf-native requires", metadata={"format": "pt"}, mode="mf-native")

    def test_native_redundancy_removed_input_is_not_assumed_complete(self) -> None:
        """The adapter does not guess how to reassemble a partitioned native file."""
        self._reject(self.source, "mf-native requires", metadata={"format": "ms", "remove_redundancy": "True"})

    def test_sanitized_ms_family_source_remains_compatible(self) -> None:
        """Already-normalized valid files remain accepted even with a MindSpore header."""
        with self._invoke(self._family_source(self.source), metadata={"format": "ms"}):
            pass

    def test_raw_reader_contract_is_unchanged(self) -> None:
        """The existing low-level reader still returns every tensor, without normalization."""
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "native.safetensors"
            save_file(self.source, str(checkpoint), metadata=self.schema["metadata"])
            actual = load_safetensors(checkpoint)
            self.assertEqual(set(actual), set(self.source))
            for name, value in actual.items():
                np.testing.assert_array_equal(value, self.source[name], err_msg=name)
