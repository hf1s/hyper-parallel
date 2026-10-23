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

"""CPU numerical and checkpoint contracts for Muon's cross-parameter NS policy."""

import copy
from contextlib import contextmanager, ExitStack
from datetime import timedelta
import math
from pathlib import Path
import tempfile
from typing import Iterator
import unittest
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist

from hyper_parallel.components.checkpoint.dcp_checkpointer import DistributedCheckpointer
from hyper_parallel.components.optim.builders import Muon as MuonBuilder
from hyper_parallel.core.distributed_checkpoint import save as dcp_save
from hyper_parallel.core.optimizer import muon as muon_module
from hyper_parallel.core.optimizer.muon import Muon, NSInputTransform, zeropower_via_newtonschulz5
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer


def _inputs(shapes: list[tuple[int, ...]]) -> list[torch.Tensor]:
    """Make small, non-identical CPU matrices without changing the global RNG."""
    generator = torch.Generator(device="cpu").manual_seed(71)
    return [torch.randn(shape, generator=generator) for shape in shapes]


def _optimizer(inputs: list[torch.Tensor], **options: object) -> Muon:
    """Construct the real optimizer over plain CPU parameters."""
    parameters = [torch.nn.Parameter(torch.zeros_like(value)) for value in inputs]
    for index, parameter in enumerate(parameters):
        parameter.model_name = f"matrix_{index}"
    return Muon(parameters, ns_variant="legacy", ns_steps=3, **options)


def _run_ns(optimizer: Muon, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    """Exercise the ordinary update dispatch and its original in-place writeback."""
    parameters = optimizer.param_groups[0]["params"]
    working = {parameter: value.clone() for parameter, value in zip(parameters, inputs)}
    with torch.no_grad():
        updates = optimizer._compute_batched_ns_updates(  # pylint: disable=protected-access
            parameters, working, optimizer.param_groups[0], no_shard=True,
        )
    for parameter in parameters:
        if updates[parameter].data_ptr() != working[parameter].data_ptr():
            raise AssertionError("NS update must retain the original input storage")
    return [updates[parameter] for parameter in parameters]


def _chain(flatten: bool, policy: bool) -> tuple[torch.nn.Module, ChainedOptimizer, Muon]:
    """Build real FQN state without initializing a process group."""
    model = torch.nn.Linear(5, 3)
    muon = Muon([model.weight], ns_batch_across_params=policy)
    muon.state[model.weight]["momentum_buffer"] = torch.ones_like(model.weight)
    optimizers = {"muon": muon}
    if flatten:
        adam = torch.optim.AdamW([model.bias], foreach=False)
        model.bias.grad = torch.ones_like(model.bias)
        adam.step()
        adam.zero_grad()
        optimizers = {"adamw": adam, "muon": muon}
    return model, ChainedOptimizer(model, optimizers, flatten=flatten), muon


def _remove_policy(state: dict, flatten: bool) -> None:
    """Represent an old checkpoint while retaining every other field."""
    if flatten:
        state.pop("param_groups.weight.ns_batch_across_params")
    else:
        for group in state["param_groups"]:
            group.pop("ns_batch_across_params", None)


@contextmanager
def _single_rank_process_group() -> Iterator[None]:
    """Provide DCP's real CPU rank/barrier context without reusing an existing group."""
    if dist.is_initialized():
        raise RuntimeError("Checkpoint tests require an uninitialized process group")
    with tempfile.TemporaryDirectory() as rendezvous:
        try:
            dist.init_process_group(
                backend="gloo", init_method=(Path(rendezvous) / "store").as_uri(),
                rank=0, world_size=1, timeout=timedelta(seconds=30),
            )
            yield
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


class TestMuonNSBatching(unittest.TestCase):
    """Check numeric calls, logical matrix boundaries, and callback behavior."""

    def test_disabled_matches_independent_matrix_ns(self) -> None:
        """Both routes must compute the original NS on each original 2D tensor."""
        inputs = _inputs([(3, 5), (3, 5)])
        expected = [zeropower_via_newtonschulz5(value, 3, "legacy") * (math.sqrt(5) * 0.2)
                    for value in inputs]
        for transformed in (False, True):
            with self.subTest(transformed=transformed):
                options = {"ns_batch_across_params": False}
                if transformed:
                    options["ns_transform_fn"] = lambda _name, _value: None
                optimizer = _optimizer(inputs, **options)
                with patch.object(muon_module, "zeropower_via_newtonschulz5",
                                  wraps=zeropower_via_newtonschulz5) as ns:
                    actual = _run_ns(optimizer, inputs)
                self.assertEqual([tuple(call.args[0].shape) for call in ns.call_args_list], [(3, 5), (3, 5)])
                for result, reference in zip(actual, expected):
                    torch.testing.assert_close(result, reference, rtol=0, atol=0)

    def test_default_and_true_keep_cross_parameter_call(self) -> None:
        """The default retains the original merged-call numeric result on both routes."""
        inputs = _inputs([(3, 5), (3, 5)])
        expected = zeropower_via_newtonschulz5(torch.stack(inputs), 3, "legacy") * (math.sqrt(5) * 0.2)
        for transformed in (False, True):
            for policy in ("default", "true", "missing"):
                with self.subTest(transformed=transformed, policy=policy):
                    options = {"ns_batch_across_params": True} if policy == "true" else {}
                    optimizer = _optimizer(inputs, **options)
                    if policy == "missing":
                        optimizer.param_groups[0].pop("ns_batch_across_params")
                    if transformed:
                        optimizer.ns_transform_fn = lambda _name, _value: None
                    with patch.object(muon_module, "zeropower_via_newtonschulz5",
                                      wraps=zeropower_via_newtonschulz5) as ns:
                        actual = _run_ns(optimizer, inputs)
                    ns.assert_called_once()
                    self.assertEqual(tuple(ns.call_args.args[0].shape), (2, 3, 5))
                    for result, reference in zip(actual, expected):
                        torch.testing.assert_close(result, reference, rtol=0, atol=0)

    def test_intrinsic_expert_conv_and_higher_rank_batches_preserved(self) -> None:
        """Disabling cross-parameter batches preserves each tensor's existing shape rules."""
        shapes = [(2, 3, 5), (3, 1, 5), (2, 2, 3, 5)]
        inputs = _inputs(shapes)
        expected = [
            zeropower_via_newtonschulz5(inputs[0], 3, "legacy"),
            zeropower_via_newtonschulz5(inputs[1].squeeze(1), 3, "legacy").unsqueeze(1),
            zeropower_via_newtonschulz5(inputs[2].reshape(4, 3, 5), 3, "legacy").reshape(shapes[2]),
        ]
        for transformed in (False, True):
            with self.subTest(transformed=transformed):
                optimizer = _optimizer(inputs, ns_batch_across_params=False)
                if transformed:
                    optimizer.ns_transform_fn = lambda _name, _value: None
                with patch.object(muon_module, "zeropower_via_newtonschulz5",
                                  wraps=zeropower_via_newtonschulz5) as ns:
                    actual = _run_ns(optimizer, inputs)
                self.assertEqual([tuple(call.args[0].shape) for call in ns.call_args_list],
                                 [(2, 3, 5), (3, 5), (4, 3, 5)])
                for result, reference in zip(actual, expected):
                    torch.testing.assert_close(result, reference * (math.sqrt(5) * 0.2), rtol=0, atol=0)

    def test_same_parameter_pieces_and_callbacks_preserved(self) -> None:
        """Native views and reversible transforms keep their own pieces together."""
        inputs = _inputs([(6, 5), (6, 5)])
        expected = [zeropower_via_newtonschulz5(value.reshape(2, 3, 5), 3, "legacy").reshape(6, 5)
                    * (math.sqrt(5) * 0.2) for value in inputs]
        for transformed in (False, True):
            with self.subTest(transformed=transformed):
                prepared_names = []
                restored_names = []

                def reshape(name: str, value: torch.Tensor) -> list[torch.Tensor]:
                    prepared_names.append(name)
                    return list(value.chunk(2, dim=0))

                def transform(name: str, value: torch.Tensor) -> NSInputTransform:
                    prepared_names.append(name)

                    def restore(updates: list[torch.Tensor], output: torch.Tensor) -> None:
                        restored_names.append(name)
                        output.copy_(torch.cat(updates, dim=0))

                    return NSInputTransform([piece.clone() for piece in value.chunk(2, dim=0)], restore)

                def custom_ns(value: torch.Tensor, *, steps: int) -> torch.Tensor:
                    return zeropower_via_newtonschulz5(value, steps, "legacy")

                custom = Mock(side_effect=custom_ns)
                options = {"ns_transform_fn": transform} if transformed else {"reshape_fn": reshape}
                optimizer = _optimizer(inputs, ns_batch_across_params=False, zeropower_fn=custom, **options)
                actual = _run_ns(optimizer, inputs)
                self.assertEqual(prepared_names, ["matrix_0", "matrix_1"])
                self.assertEqual(restored_names, ["matrix_0", "matrix_1"] if transformed else [])
                self.assertEqual([tuple(call.args[0].shape) for call in custom.call_args_list], [(2, 3, 5)] * 2)
                self.assertEqual([call.kwargs for call in custom.call_args_list], [{"steps": 3}] * 2)
                for result, reference in zip(actual, expected):
                    torch.testing.assert_close(result, reference, rtol=0, atol=0)

    def test_callback_exception_is_not_retried(self) -> None:
        """The original exception propagates without a fallback NS calculation."""
        failure = RuntimeError("custom NS failure")
        for transformed in (False, True):
            with self.subTest(transformed=transformed):
                custom = Mock(side_effect=failure)
                inputs = _inputs([(3, 5), (3, 5)])
                optimizer = _optimizer(inputs, ns_batch_across_params=False, zeropower_fn=custom)
                if transformed:
                    optimizer.ns_transform_fn = lambda _name, _value: None
                with self.assertRaises(RuntimeError) as caught:
                    _run_ns(optimizer, inputs)
                self.assertIs(caught.exception, failure)
                custom.assert_called_once()

    def test_invalid_policy_rejected_before_state_changes(self) -> None:
        """Truthiness is not a substitute for the explicit boolean policy."""
        for invalid in (None, 0, 1, "false", [], {}):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "ns_batch_across_params must be a bool"):
                    _optimizer(_inputs([(3, 5)]), ns_batch_across_params=invalid)
                parameter = torch.nn.Parameter(torch.ones(3, 5))
                with self.assertRaisesRegex(ValueError, "ns_batch_across_params must be a bool"):
                    Muon([{"params": [parameter], "ns_batch_across_params": invalid}])
        optimizer = _optimizer(_inputs([(3, 5)]))
        optimizer.param_groups[0]["ns_batch_across_params"] = 1
        closure = Mock()
        with self.assertRaisesRegex(ValueError, "ns_batch_across_params must be a bool"):
            optimizer.step(closure)
        closure.assert_not_called()
        self.assertEqual(len(optimizer.state), 0)
        self.assertNotIn("step", optimizer.param_groups[0])

    def test_builder_forwards_plain_and_prefixed_policy(self) -> None:
        """Existing builder kwargs preserve the flag rather than filtering it out."""
        for key in ("ns_batch_across_params", "muon_ns_batch_across_params"):
            with self.subTest(key=key):
                model = torch.nn.Linear(5, 3)
                builder = MuonBuilder(muon_config={key: False}, adamw_config={}, model=model)
                leaf = builder.get_optimizer().optimizers_dict["muon"]
                self.assertIs(leaf.defaults["ns_batch_across_params"], False)
                self.assertTrue(all(group["ns_batch_across_params"] is False for group in leaf.param_groups))


class TestMuonNSBatchingCheckpoint(unittest.TestCase):
    """Restore explicit policies and legacy defaults through real checkpoint paths."""

    def test_raw_checkpoint_policies_and_legacy_default(self) -> None:
        """Raw leaf state restores False and defaults missing metadata to True."""
        optimizer = _optimizer(_inputs([(3, 5)]), ns_batch_across_params=False)
        saved = copy.deepcopy(optimizer.state_dict())
        optimizer.param_groups[0]["ns_batch_across_params"] = True
        optimizer.load_state_dict(saved)
        self.assertIs(optimizer.param_groups[0]["ns_batch_across_params"], False)
        saved["param_groups"][0].pop("ns_batch_across_params")
        optimizer.load_state_dict(saved)
        self.assertIs(optimizer.param_groups[0]["ns_batch_across_params"], True)
        self.assertNotIn("ns_batch_across_params", saved["param_groups"][0])
        saved["param_groups"][0]["ns_batch_across_params"] = 0
        with self.assertRaisesRegex(ValueError, "ns_batch_across_params must be a bool"):
            optimizer.load_state_dict(saved)
        self.assertIs(optimizer.param_groups[0]["ns_batch_across_params"], True)

    def test_fqn_checkpoint_missing_policy_and_invalid_value(self) -> None:
        """Fresh FQN restores fill old fields and reject bad values before any leaf load."""
        for flatten in (False, True):
            with self.subTest(flatten=flatten):
                _, source, _ = _chain(flatten, False)
                saved = copy.deepcopy(source.state_dict())
                legacy = copy.deepcopy(saved)
                _remove_policy(legacy, flatten)
                _, chain, muon = _chain(flatten, False)
                chain.load_state_dict(legacy)
                self.assertIs(muon.param_groups[0]["ns_batch_across_params"], True)
                _, chain, muon = _chain(flatten, True)
                chain.load_state_dict(saved)
                self.assertIs(muon.param_groups[0]["ns_batch_across_params"], False)
                if flatten:
                    saved["param_groups.weight.ns_batch_across_params"] = "False"
                else:
                    saved["param_groups"][0]["ns_batch_across_params"] = "False"
                original_lrs = [leaf.param_groups[0]["lr"] for leaf in chain]
                with ExitStack() as stack:
                    leaf_loads = [stack.enter_context(patch.object(leaf, "load_state_dict",
                                                                 wraps=leaf.load_state_dict)) for leaf in chain]
                    with self.assertRaisesRegex(ValueError, "ns_batch_across_params must be a bool"):
                        chain.load_state_dict(saved)
                    for leaf_load in leaf_loads:
                        leaf_load.assert_not_called()
                self.assertEqual([leaf.param_groups[0]["lr"] for leaf in chain], original_lrs)
                self.assertIs(muon.param_groups[0]["ns_batch_across_params"], False)

    @_single_rank_process_group()
    def test_dcp_restores_false_and_defaults_legacy_missing_to_true(self) -> None:
        """Real DCP I/O must not retain a live False when an old checkpoint lacks the field."""
        for flatten in (False, True):
            for legacy in (False, True):
                for as_list in (False, True):
                    with self.subTest(flatten=flatten, legacy=legacy, as_list=as_list):
                        _, chain, muon = _chain(flatten, False)
                        saved = copy.deepcopy(chain.state_dict())
                        if legacy:
                            _remove_policy(saved, flatten)
                        muon.param_groups[0]["ns_batch_across_params"] = not legacy
                        with tempfile.TemporaryDirectory() as checkpoint:
                            payload = {"optimizer": [saved] if as_list else saved}
                            dcp_save(payload, checkpoint_id=checkpoint, no_dist=True)
                            skeleton = chain.state_dict()
                            load_state = {"optimizer": [skeleton] if as_list else skeleton}
                            DistributedCheckpointer().load(checkpoint, load_state)
                        loaded = load_state["optimizer"][0] if as_list else load_state["optimizer"]
                        chain.load_state_dict(loaded)
                        self.assertIs(muon.param_groups[0]["ns_batch_across_params"], legacy)

    @_single_rank_process_group()
    def test_dcp_weights_only_keeps_recipe_policy(self) -> None:
        """Loading only model tensors cannot change the configured optimizer policy."""
        model, chain, muon = _chain(True, False)
        saved = copy.deepcopy(model.state_dict())
        momentum = muon.state[model.weight]["momentum_buffer"].clone()
        with tempfile.TemporaryDirectory() as checkpoint:
            dcp_save({"model": saved}, checkpoint_id=checkpoint, no_dist=True)
            load_state = {"model": model.state_dict()}
            DistributedCheckpointer().load(checkpoint, load_state)
            model.load_state_dict(load_state["model"])
        self.assertIs(muon.param_groups[0]["ns_batch_across_params"], False)
        torch.testing.assert_close(muon.state[model.weight]["momentum_buffer"], momentum, rtol=0, atol=0)
        self.assertEqual(len(chain.chained_optimizers), 2)


if __name__ == "__main__":
    unittest.main()
