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
"""Async EP collective lifetime, scheduling and autograd contracts."""

from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.distributed.expert_parallel.collectives import ep_all_to_all


class TestAsyncExchange(unittest.TestCase):
    """Model pending communication without an accelerator or real process group."""

    def _exchange(self, backend, callback, *, empty=False):
        """Run a rank-local roundtrip, with deferred writes and reverse gradients."""
        events = []
        work = SimpleNamespace(wait=lambda: events.append("wait"))

        def exchange(output: Any, source: Any, **kwargs: Any) -> Any:
            """Model the real backend's untracked writes into receive storage."""
            if kwargs.get("async_op", False):
                events.append("launch")

                @torch.no_grad()
                def wait() -> None:
                    """Complete pending writes when the consumer joins communication."""
                    events.append("wait")
                    if isinstance(output, list):
                        for destination, origin in zip(output, source):
                            destination.copy_(origin)
                    else:
                        output.copy_(source)

                work.wait = wait
                return work
            if isinstance(output, list):
                for destination, origin in zip(output, source):
                    destination.copy_(origin)
            else:
                output.copy_(source)
            return None

        states = torch.randn(0 if empty else 3, 4, requires_grad=True)
        counts = [0, 0] if empty else [0, 3]

        def compute() -> None:
            """Execute independent compute before consuming the pending receive buffer."""
            events.append("compute")
            callback(states)

        with patch("hyper_parallel.distributed.expert_parallel.collectives.dist.get_backend",
                   return_value=backend), \
             patch("hyper_parallel.distributed.expert_parallel.collectives.dist.all_reduce"), \
             patch("hyper_parallel.distributed.expert_parallel.collectives.dist.all_to_all",
                   side_effect=exchange), \
             patch("hyper_parallel.distributed.expert_parallel.collectives.dist.all_to_all_single",
                   side_effect=exchange):
            try:
                result = ep_all_to_all(states, counts, counts, object(), overlap_fn=compute)
                torch.testing.assert_close(result, states)
                result.sum().backward()
                torch.testing.assert_close(states.grad, torch.ones_like(states))
            finally:
                self.assertEqual(events, ["launch", "compute", "wait"])

    def test_forward_backward_including_empty_splits(self):
        """HCCL, NCCL and padded Gloo paths wait before reads and preserve reverse exchange."""
        for backend in ("hccl", "nccl", "gloo"):
            for empty in (False, True):
                with self.subTest(backend=backend, empty=empty):
                    self._exchange(backend, lambda _states: None, empty=empty)

    def test_independent_compute_preserves_autograd_mode(self):
        """The callback is outside Function.forward and builds a regular autograd graph."""
        outputs = []
        self._exchange("hccl", lambda states: outputs.append(states.square().sum()))
        self.assertTrue(outputs[0].requires_grad)

    def test_exception_still_waits(self):
        """A failing shared expert cannot leave an outstanding collective behind."""
        def fail(_states: torch.Tensor) -> None:
            """Model a shared expert raising after communication has launched."""
            raise ValueError("shared failed")

        with self.assertRaisesRegex(ValueError, "shared failed"):
            self._exchange("hccl", fail)

    def test_no_grad_callback(self):
        """Inference callbacks inherit no_grad from the caller."""
        outputs = []
        with torch.no_grad():
            with patch("hyper_parallel.distributed.expert_parallel.collectives._EPAllToAllAsync.apply",
                       return_value=(torch.ones(1, 2), SimpleNamespace(wait=lambda: None, pad_to=0))):
                ep_all_to_all(torch.ones(1, 2), [1], [1], object(),
                              overlap_fn=lambda: outputs.append(torch.ones(1, requires_grad=True) * 2))
        self.assertFalse(outputs[0].requires_grad)
