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
# pylint: disable=C0415
"""Tests that an optimizer config key the optimizer cannot consume is rejected.

Dropping such a key made a run keep the constructor default instead of the value
written in the YAML, which is silently indistinguishable from a numeric
regression between parallel layouts.
"""
import unittest

from hyper_parallel.core.optimizer import _filter_optimizer_config


class _ToyOptimizer:
    """Minimal optimizer stand-in with a positional and a keyword constructor arg."""

    ADDITIONAL_CONFIG_KEYS = {"extra_option"}

    def __init__(self, params, lr=1e-3, momentum=0.9, **advanced):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.advanced = advanced


class TestOptimizerConfigFilter(unittest.TestCase):
    """Unsupported optimizer config keys must not be dropped silently."""

    def test_supported_keys_are_kept(self):
        """Declared constructor and additional keys survive filtering."""
        filtered = _filter_optimizer_config(
            "toy",
            _ToyOptimizer,
            {"lr": 0.02, "momentum": 0.95, "extra_option": True},
        )
        self.assertEqual(
            filtered,
            {"lr": 0.02, "momentum": 0.95, "extra_option": True},
        )

    def test_prefixed_keys_are_normalized(self):
        """A ``<optimizer>_`` prefix is stripped before the key is validated."""
        filtered = _filter_optimizer_config(
            "toy",
            _ToyOptimizer,
            {"toy_lr": 0.02, "toy_momentum": 0.95},
        )
        self.assertEqual(filtered, {"lr": 0.02, "momentum": 0.95})

    def test_unknown_key_raises(self):
        """An unrecognized key fails loudly instead of falling back to a default."""
        with self.assertRaises(ValueError) as ctx:
            _filter_optimizer_config("toy", _ToyOptimizer, {"lr": 0.02, "ns_dtype": "float32"})
        message = str(ctx.exception)
        self.assertIn("ns_dtype", message)
        self.assertIn("toy", message)
        self.assertIn("lr", message)

    def test_unknown_key_reported_after_prefix_normalization(self):
        """The error names the normalized key, not the prefixed spelling."""
        with self.assertRaises(ValueError) as ctx:
            _filter_optimizer_config("toy", _ToyOptimizer, {"toy_bogus": 1})
        self.assertIn("bogus", str(ctx.exception))

    def test_empty_config_is_allowed(self):
        """Optimizers configured entirely from defaults keep working."""
        self.assertEqual(_filter_optimizer_config("toy", _ToyOptimizer, None), {})
        self.assertEqual(_filter_optimizer_config("toy", _ToyOptimizer, {}), {})


if __name__ == "__main__":
    unittest.main()
