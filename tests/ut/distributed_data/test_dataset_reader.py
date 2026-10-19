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
"""Tests for index-preserving payload adapters used by planned reads."""

import unittest

from torch.utils.data._utils.pin_memory import (  # pylint: disable=forbidden-backend-import
    pin_memory as torch_pin_memory,
)

from hyper_parallel.distributed_data.dataset_reader import _IndexedPayload
from tests.common.mark_utils import arg_mark


class _PinnablePayload:
    def __init__(self, value: int) -> None:
        """Store a value and expose whether the walker visited it."""
        self.value = value
        self.pin_calls = 0

    def pin_memory(self) -> str:
        """Record recursive pinning without requiring accelerator hardware."""
        self.pin_calls += 1
        return f"pinned-{self.value}"


class TestIndexedPayload(unittest.TestCase):
    """Verify indexed payloads retain native DataLoader memory-pinning behavior."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_indexed_payload_participates_in_recursive_pin_memory_walk(self) -> None:
        """Feature: Indexed payload memory pinning.
        Description: Apply recursive pinning to an index-preserving payload wrapper.
        Expectation: Recursive pinning reaches the payload while preserving its index.
        """
        pinnable = _PinnablePayload(9)
        indexed = _IndexedPayload(dataset_index=17, payload={"nested": [pinnable]})

        pinned = torch_pin_memory(indexed)

        self.assertIsInstance(pinned, _IndexedPayload)
        self.assertEqual(pinned.dataset_index, 17)
        self.assertEqual(pinned.payload, {"nested": ["pinned-9"]})
        self.assertEqual(pinnable.pin_calls, 1)
