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
"""Muon precision must survive momentum, NS, relay and checkpoint boundaries."""
# pylint: disable=protected-access

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer.muon_shard import build_pad_ns_inputs


class TestMuonNSPrecision(unittest.TestCase):
    """Exercise public configuration and real CPU optimizer updates."""

    def test_default_and_explicit_bfloat16_are_identical(self) -> None:
        """The new option keeps the existing default numerical recipe."""
        first = torch.nn.Parameter(torch.ones(2, 3))
        second = torch.nn.Parameter(first.detach().clone())
        default = Muon([first], ns_variant="legacy")
        explicit = Muon([second], ns_variant="legacy", ns_dtype="bfloat16")
        for _ in range(2):
            first.grad = torch.tensor([[0.1234567, -0.5678912, 0.7891234], [0.2345678, 0.8912345, -0.9123456]])
            second.grad = first.grad.clone()
            default.step()
            explicit.step()
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(default.param_groups[0]["ns_dtype"], "bfloat16")

    def test_float32_momentum_input_preserves_non_bfloat16_values(self) -> None:
        """FP32 products are not rounded before the NS implementation sees them."""
        parameter = torch.nn.Parameter(torch.ones(2, 2))
        parameter.grad = torch.tensor([[0.1234567, -0.5678912], [0.7891234, 0.2345678]])
        optimizer = Muon([parameter], ns_dtype="float32", momentum=0.95)
        updates = optimizer._update_muon_momentum(optimizer.param_groups[0], [parameter])
        expected = parameter.grad.mul(0.95).add(parameter.grad)
        self.assertEqual(updates[parameter].dtype, torch.float32)
        torch.testing.assert_close(updates[parameter], expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(updates[parameter], expected.bfloat16().float()))

    def test_custom_momentum_uses_each_groups_precision(self) -> None:
        """Callback output and per-group overrides obey the same dtype boundary."""
        first = torch.nn.Parameter(torch.ones(2, 2))
        second = torch.nn.Parameter(torch.ones(2, 2))
        optimizer = Muon([{"params": [first], "ns_dtype": "float32"}, {"params": [second]}],
                         momentum_update_fn=lambda grads, *_args: grads)
        for parameter, group, dtype in zip((first, second), optimizer.param_groups, (torch.float32, torch.bfloat16)):
            parameter.grad = torch.full_like(parameter, 0.1234567)
            update = optimizer._update_muon_momentum(group, [parameter])[parameter]
            self.assertEqual(update.dtype, dtype)
            torch.testing.assert_close(update, parameter.grad.to(dtype), rtol=0, atol=0)

    def test_custom_momentum_aliases_survive_two_real_steps(self) -> None:
        """Returning live gradients or momentum buffers must not expose them to NS writes."""
        for dtype in ("bfloat16", "float32"):
            for return_buffers in (False, True):
                with self.subTest(dtype=dtype, return_buffers=return_buffers):
                    first = torch.nn.Parameter(torch.ones(2, 3))
                    second = torch.nn.Parameter(first.detach().clone())

                    def momentum(grads: list, buffers: list, beta1: float, *_args: object) -> list:
                        """Update live momentum and return an intentionally aliased result."""
                        for grad, buffer in zip(grads, buffers):
                            buffer.mul_(beta1).add_(grad)
                        return buffers if return_buffers else grads

                    def copied_momentum(grads: list, buffers: list, beta1: float, *args: object) -> list:
                        """Independent-storage reference using the identical momentum equation."""
                        return [value.clone() for value in momentum(grads, buffers, beta1, *args)]

                    aliased = Muon([first], ns_dtype=dtype, momentum_update_fn=momentum)
                    copied = Muon([second], ns_dtype=dtype, momentum_update_fn=copied_momentum)
                    for step in range(2):
                        grad = torch.arange(6, dtype=torch.float32).view(2, 3).add_(1 + step).mul_(0.1234567)
                        first.grad, second.grad = grad.clone(), grad.clone()
                        aliased.step()
                        copied.step()
                        torch.testing.assert_close(first.grad, grad, rtol=0, atol=0)
                        torch.testing.assert_close(first, second, rtol=0, atol=0)
                        torch.testing.assert_close(aliased.state[first]["momentum_buffer"],
                                                   copied.state[second]["momentum_buffer"], rtol=0, atol=0)

    def test_missing_gradients_use_group_dtype_at_the_gather_boundary(self) -> None:
        """No-grad ranks and empty local shards match their FP32-input peer's collective."""
        for shape in ((2, 3), (0, 3)):
            with self.subTest(shape=shape):
                parameter = torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))
                optimizer = Muon([parameter], ns_dtype="float32")
                assignment = SimpleNamespace(
                    layout_spec=None, owned_records=[SimpleNamespace(param=parameter, index=0)],
                )
                captured = []

                def gather(inputs: list, *_args: object, **_kwargs: object) -> list:
                    """Capture the actual local collective payload, without distributed setup."""
                    captured.extend(inputs)
                    return [None]

                with patch.object(optimizer, "_get_shard_info", return_value=((2,), (1,), (None,), 2)), \
                        patch("hyper_parallel.core.optimizer.muon.fused_allgather_dtensor_params", side_effect=gather):
                    optimizer._gather_and_compute_shard_updates(
                        [parameter], {}, assignment, {0: (0,)}, optimizer.param_groups[0], None,
                    )
                peer = build_pad_ns_inputs([parameter], {parameter: torch.ones(shape)}, dtype=torch.float32)
                self.assertEqual(captured[0].dtype, peer[0].dtype)
                self.assertEqual(captured[0].shape, peer[0].shape)
                self.assertEqual(torch.count_nonzero(captured[0]).item(), 0)
        with self.assertRaisesRegex(ValueError, "collective dtype"):
            build_pad_ns_inputs([parameter], {parameter: torch.empty(shape, dtype=torch.bfloat16)},
                                dtype=torch.float32)

    def test_invalid_precision_fails_before_state_mutation(self) -> None:
        """Reject unsupported names and types at constructor, group and step boundaries."""
        parameter = torch.nn.Parameter(torch.ones(2, 2))
        for value in ("float16", "fp32", torch.float32, None, 1, []):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "ns_dtype"):
                    Muon([parameter], ns_dtype=value)
                with self.assertRaisesRegex(ValueError, "ns_dtype"):
                    Muon([{"params": [parameter], "ns_dtype": value}])
        optimizer = Muon([parameter])
        optimizer.param_groups[0]["ns_dtype"] = "invalid"
        parameter.grad = torch.ones_like(parameter)
        with self.assertRaisesRegex(ValueError, "ns_dtype"):
            optimizer.step()
        self.assertEqual(len(optimizer.state), 0)
        self.assertNotIn("step", optimizer.param_groups[0])

    def test_checkpoint_restores_precision_and_legacy_defaults(self) -> None:
        """A saved FP32 run resumes exactly; an old checkpoint retains BF16."""
        first = torch.nn.Parameter(torch.ones(2, 3))
        optimizer = Muon([first], ns_dtype="float32", ns_variant="legacy")
        first.grad = torch.arange(6, dtype=torch.float32).view(2, 3) * 0.1234567
        optimizer.step()
        saved = copy.deepcopy(optimizer.state_dict())
        second = torch.nn.Parameter(first.detach().clone())
        restored = Muon([second])
        restored.load_state_dict(saved)
        self.assertEqual(restored.param_groups[0]["ns_dtype"], "float32")
        second.grad = first.grad.clone()
        optimizer.step()
        restored.step()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        legacy = copy.deepcopy(saved)
        del legacy["param_groups"][0]["ns_dtype"]
        restored.load_state_dict(legacy)
        self.assertEqual(restored.param_groups[0]["ns_dtype"], "bfloat16")
        self.assertNotIn("ns_dtype", legacy["param_groups"][0])
        invalid = copy.deepcopy(saved)
        invalid["param_groups"][0]["ns_dtype"] = "invalid"
        with self.assertRaisesRegex(ValueError, "ns_dtype"):
            restored.load_state_dict(invalid)
        self.assertEqual(restored.param_groups[0]["ns_dtype"], "bfloat16")

    def test_fused_relay_preserves_fp32_updates_and_byte_alignment(self) -> None:
        """The owner packs and relays FP32 updates without a hidden BF16 cast."""
        parameter = torch.nn.Parameter(torch.zeros(2, 3))
        optimizer = Muon([parameter], ns_dtype="float32", lr=1.0, weight_decay=0.0)
        update = torch.full_like(parameter, 0.1234567)
        assignment = SimpleNamespace(layout_spec=None)
        buffers = []

        def relay(packs: dict, *_args: object) -> None:
            """Observe the actual buffer passed to the communication boundary."""
            buffers.extend(packs.values())

        with patch.object(optimizer, "_get_shard_info", return_value=((2,), (0,), (None,), 2)), \
                patch.object(optimizer, "_batched_relay_broadcast", side_effect=relay), \
                patch("hyper_parallel.core.optimizer.muon.chunk_update_by_layout", side_effect=lambda value, *_: value):
            with torch.no_grad():
                optimizer._fused_broadcast_and_apply([parameter], {parameter: update}, {parameter: (0,)},
                                                    optimizer.param_groups[0], assignment)
        self.assertEqual(len(buffers), 1)
        self.assertEqual(buffers[0].dtype, torch.float32)
        self.assertEqual(buffers[0].numel() * buffers[0].element_size(), 512)
        self.assertEqual(torch.count_nonzero(buffers[0][6:]).item(), 0)
        torch.testing.assert_close(parameter, -update, rtol=0, atol=0)

    def test_memory_batch_budget_accounts_for_dtype(self) -> None:
        """FP32 halves the element budget while preserving owner-filling semantics."""
        optimizer = object.__new__(Muon)
        parameters = [torch.empty(160 * 1024 * 1024, device="meta") for _ in range(3)]
        bf16 = optimizer._split_into_memory_safe_batches(parameters, dtype=torch.bfloat16)
        fp32 = optimizer._split_into_memory_safe_batches(parameters, dtype=torch.float32)
        self.assertEqual([len(batch) for batch in bf16], [3])
        self.assertEqual([len(batch) for batch in fp32], [1, 1, 1])
        owner_filled = optimizer._split_into_memory_safe_batches(parameters, min_batch_size=2, dtype=torch.float32)
        self.assertEqual([len(batch) for batch in owner_filled], [2, 1])


if __name__ == "__main__":
    unittest.main()
