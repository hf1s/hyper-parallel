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
"""Complete-step preparation, bounded lookahead, and delivered checkpoint progress."""

import unittest
from contextlib import nullcontext
from itertools import product
from threading import Event, current_thread
from unittest.mock import Mock, patch

import torch

from tests.common.mark_utils import arg_mark
from tests.ut.distributed_data.test_accumulation_window import _GatedDataset, _ids, _loader
from tests.ut.distributed_data.test_batch_sampler import _TrackedDataset, _loader as _native_loader, _sampler


class TestOptimizerStepPrefetch(unittest.TestCase):
    """Check the public builder's complete-step buffering path."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_full_step_is_ready_and_only_one_successor_is_prefetched(self) -> None:
        """Feature: Complete-step double buffering.
        Description: Finish source reads, packing and staging for two steps before consuming the first.
        Expectation: Only twelve samples are read, all six batches are staged, and checkpoints retain one delivery.
        """
        dataset = _GatedDataset()
        dataset.release.set()
        staged = []
        with _loader(samples=dataset, accumulation=3, optimizer_step=True) as loader:
            original = loader._source._stage_batch

            def stage(batch: object) -> object:
                """Record completed per-micro-batch staging on the producer."""
                staged.append((_ids(batch), current_thread()))
                return original(batch)

            with patch.object(loader._source, "_stage_batch", side_effect=stage):
                self.assertEqual(_ids(next(loader)), [0, 1])
                loader.wait_for_prefetch()
                self.assertEqual(dataset.reads, list(range(12)))
                self.assertEqual([ids for ids, _ in staged], [list(range(i, i + 2)) for i in range(0, 12, 2)])
                self.assertEqual({thread for _, thread in staged}, {loader._thread})
                self.assertIsNot(loader._thread, current_thread())
                self.assertEqual(loader.last_plan.step, 0)
                state = loader.state_dict()
                self.assertEqual(state["step"], 1)
                self.assertEqual(state["dataset_reader"]["sampler"]["consumed_samples"], 0)
                self.assertEqual(state["dataset_reader"]["window"]["sampler"]["consumed_samples"], 6)
                self.assertEqual(state["optimizer_step_remaining"], 2)
                for _ in range(3):
                    loader.prefetch_plan()
                    loader.prefetch()
                self.assertEqual(dataset.reads, list(range(12)))
                self.assertEqual(_ids(next(loader)), [2, 3])
                self.assertEqual(_ids(next(loader)), [4, 5])
                self.assertEqual(dataset.reads, list(range(12)))
                self.assertIsNone(loader._source._prefetch_thread)
                self.assertIsNone(loader._source._dataset_reader._window_thread)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_slow_next_step_does_not_block_current_delivery_or_checkpoint(self) -> None:
        """Feature: Independent current and future steps.
        Description: Block the next step in reading, planning, exchange or staging.
        Expectation: Current batches and their checkpoints remain available before the future is released.
        """
        for stage in ("read", "plan", "exchange", "stage"):
            with self.subTest(stage=stage):
                self._check_blocked_stage(stage)

    def _check_blocked_stage(self, stage: str) -> None:
        dataset = _GatedDataset()
        entered, release = Event(), Event()
        loader = _loader(samples=dataset, accumulation=3, optimizer_step=True)
        source = loader._source
        owner, method = {
            "read": (source, "_prepare_local_metadata"), "plan": (source._planner, "plan_window"),
            "exchange": (source._data_plane, "exchange_prepared"), "stage": (source, "_stage_batch"),
        }[stage]
        original = getattr(owner, method)
        dataset.release.set()

        def gated(*args: object, **kwargs: object) -> object:
            """Hold the first operation after the current step has been prepared."""
            if source._step >= 3:
                entered.set()
                if not release.wait(timeout=10):
                    raise RuntimeError("Test did not release future preparation")
            return original(*args, **kwargs)

        try:
            with patch.object(owner, method, side_effect=gated):
                self.assertEqual(_ids(next(loader)), [0, 1])
                self.assertTrue(entered.wait(timeout=5))
                state = loader.state_dict()
                self.assertEqual(_ids(next(loader)), [2, 3])
                self.assertEqual(_ids(next(loader)), [4, 5])
                self.assertFalse(release.is_set())
                release.set()
                expected = [_ids(batch) for batch in loader]
        finally:
            release.set()
            loader.close()
        with _loader(samples=[{"id": i} for i in range(18)], accumulation=3, optimizer_step=True) as resumed:
            resumed.load_state_dict(state)
            self.assertEqual([_ids(batch) for batch in resumed], [[2, 3], [4, 5], *expected])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_resume_preserves_subwindows_and_partial_optimizer_boundaries(self) -> None:
        """Feature: Checkpointing complete-step speculation.
        Description: Resume online/offline loaders inside and between differently sized balancing windows.
        Expectation: Remaining batches and planning boundaries match uninterrupted execution.
        """
        for metadata, accumulation, window in product((False, True), (3, 10), (1, 3)):
            with self.subTest(metadata=metadata, accumulation=accumulation, window=window):
                options = {"metadata_mode": metadata, "accumulation": accumulation,
                           "window": window, "optimizer_step": True}
                states = []
                with _loader(**options) as loader:
                    delivered = []
                    for batch in loader:
                        delivered.append(_ids(batch))
                        if len(delivered) in (1, 3, accumulation, accumulation + 1):
                            states.append((len(delivered), loader.state_dict()))
                    loader.set_epoch(1)
                    self.assertEqual([_ids(batch) for batch in loader], delivered)
                for count, state in states:
                    with _loader(**options) as resumed:
                        resumed.load_state_dict(state)
                        self.assertEqual([_ids(batch) for batch in resumed], delivered[count:])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_step_prefetch_preserves_three_three_three_one_planning(self) -> None:
        """Feature: Independent optimizer and balancing windows.
        Description: Buffer ten-round steps while balancing at most three rounds together.
        Expectation: Planner windows remain 3+3+3+1 and the epoch tail stays within its optimizer step.
        """
        with _loader(accumulation=10, window=3, optimizer_step=True) as loader:
            counts = []
            original = loader._source._plan_native_batch

            def record(metadata: list) -> object:
                """Record native rounds presented to each planning invocation."""
                counts.append(len(metadata[0]) // 2)
                return original(metadata)

            with patch.object(loader._source, "_plan_native_batch", side_effect=record):
                batches = [_ids(batch) for batch in loader]
            self.assertEqual(batches, [list(range(i, i + 2)) for i in range(0, 44, 2)])
            self.assertEqual(counts, [3, 3, 3, 1, 3, 3, 3, 1, 2, 0])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_future_failure_is_deferred_until_current_step_finishes(self) -> None:
        """Feature: Speculative error delivery.
        Description: Fail the next optimizer step while three current batches are ready.
        Expectation: Current batches finish and the original error appears at the next step boundary.
        """
        error = ValueError("future step failed")
        dataset = _GatedDataset(error=error)
        dataset.release.set()
        with _loader(samples=dataset, accumulation=3, optimizer_step=True) as loader:
            self.assertEqual(_ids(next(loader)), [0, 1])
            with self.assertRaises(ValueError) as caught:
                loader.wait_for_prefetch()
            self.assertIs(caught.exception, error)
            self.assertEqual(_ids(next(loader)), [2, 3])
            self.assertEqual(_ids(next(loader)), [4, 5])
            state = loader.state_dict()
            self.assertEqual(state["dataset_reader"]["sampler"]["consumed_samples"], 6)
            with self.assertRaises(ValueError) as caught:
                next(loader)
            self.assertIs(caught.exception, error)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_device_binding_and_complete_step_stream_handoff(self) -> None:
        """Feature: Accelerator step ownership.
        Description: Prepare CPU-shaped test inputs under a mocked CUDA stream and event API.
        Expectation: The producer binds once, publishes complete events, and consumers record their own stream.
        """
        module = Mock()
        module.stream.side_effect = lambda _: nullcontext()
        with patch.object(torch, "cuda", module), patch(
                "hyper_parallel.distributed_data.api._create_device_prefetcher", return_value=None,
        ), patch("hyper_parallel.distributed_data.optimizer_step._record_stream") as record, _loader(
                accumulation=3, samples=[{"id": i} for i in range(12)], optimizer_step=True, device="cuda:3",
        ) as loader:
            self.assertEqual(_ids(next(loader)), [0, 1])
            loader.wait_for_prefetch()
            module.set_device.assert_called_once_with(torch.device("cuda:3"))
            self.assertEqual(module.Event.return_value.record.call_count, 2)
            self.assertEqual(module.Event.return_value.synchronize.call_count, 2)
            module.current_stream.return_value.wait_event.assert_called_once_with(module.Event.return_value)
            record.assert_called_once_with(({"id": 0}, {"id": 1}), module.current_stream.return_value)
            module.synchronize.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_device_initialization_failure_reaches_consumer(self) -> None:
        """Feature: Failed producer initialization.
        Description: Fail device binding before the background thread reads any source data.
        Expectation: next() raises the original error and close() drains without hanging.
        """
        error = RuntimeError("device binding failed")
        with patch("hyper_parallel.distributed_data.api._create_device_prefetcher", return_value=None), patch.object(
                torch.cuda, "set_device", side_effect=error,
        ), _loader(accumulation=3, optimizer_step=True, device="cuda:3") as loader:
            with self.assertRaises(RuntimeError) as caught:
                next(loader)
            self.assertIs(caught.exception, error)
            self.assertEqual(loader.state_dict()["step"], 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_initial_sampler_offset_stops_at_its_optimizer_boundary(self) -> None:
        """Feature: Initial partial optimizer steps.
        Description: Start one round into an optimizer step and set the same epoch before iterating.
        Expectation: The first buffer contains only two rounds and future preparation respects the next boundary.
        """
        dataset = _TrackedDataset(size=18)
        sampler = _sampler(total_samples=18, global_batch_size=6, consumed_samples=2)
        with _native_loader(dataset, sampler=sampler, balance_window_size=3) as loader:
            loader.set_epoch(0)
            self.assertEqual(_ids(next(loader)), [2, 3])
            self.assertEqual(len(loader._current.batches), 1)
            loader.wait_for_prefetch()
            self.assertEqual(dataset.reads, list(range(2, 12)))
            self.assertEqual(loader.state_dict()["optimizer_step_remaining"], 1)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_same_epoch_after_prefetch_preserves_undelivered_inputs(self) -> None:
        """Feature: Idempotent initial epoch selection.
        Description: Prefetch a full step before setting the current epoch again.
        Expectation: The prepared step and delivered cursor remain unchanged.
        """
        dataset = _TrackedDataset(size=18)
        sampler = _sampler(total_samples=18, global_batch_size=6, consumed_samples=2)
        with _native_loader(dataset, sampler=sampler, balance_window_size=3) as loader:
            loader.prefetch()
            loader.wait_for_prefetch()
            state = loader.state_dict()
            future = loader._future
            loader.set_epoch(0)
            self.assertEqual(loader.state_dict(), state)
            self.assertIs(loader._future, future)
            self.assertEqual(dataset.reads, list(range(2, 6)))
            self.assertEqual([_ids(batch) for batch in loader], [list(range(i, i + 2)) for i in range(2, 18, 2)])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_legacy_checkpoint_uses_saved_sampler_and_window_progress(self) -> None:
        """Feature: Native checkpoints predating optimizer-step prefetch.
        Description: Restore a partial step into a fresh sampler with a different initial cursor.
        Expectation: Both metadata modes stop their first buffer at the saved optimizer boundary.
        """
        for metadata_mode, window in product((False, True), (1, 3)):
            with self.subTest(metadata_mode=metadata_mode, window=window):
                dataset = _TrackedDataset(size=18)
                sampler = _sampler(total_samples=18, global_batch_size=6, consumed_samples=2)
                options = {"metadata_mode": metadata_mode, "balance_window_size": window}
                with _native_loader(dataset, sampler=sampler, **options) as loader:
                    self.assertEqual(_ids(next(loader)), [2, 3])
                    state = loader.state_dict()
                    del state["optimizer_step_remaining"]
                    expected = [_ids(batch) for batch in loader]
                sampler = _sampler(total_samples=18, global_batch_size=6)
                with _native_loader(dataset, sampler=sampler, **options) as resumed:
                    resumed.load_state_dict(state)
                    self.assertEqual(_ids(next(resumed)), expected[0])
                    self.assertEqual(len(resumed._current.batches), 0)
                    self.assertEqual(resumed.state_dict()["optimizer_step_remaining"], 3)
                    self.assertEqual([_ids(batch) for batch in resumed], expected[1:])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_workers_and_epoch_reset_preserve_step_replay(self) -> None:
        """Feature: Reusable source workers on the step producer.
        Description: Read two epochs and restore a partial step with persistent or transient DataLoader workers.
        Expectation: Output order and checkpoints match while the producer exits at end of iteration.
        """
        for persistent in (False, True):
            with self.subTest(persistent=persistent):
                options = {"samples": [{"id": i} for i in range(18)], "accumulation": 3,
                           "optimizer_step": True, "workers": 1, "persistent_workers": persistent}
                with _loader(**options) as loader:
                    self.assertEqual(_ids(next(loader)), [0, 1])
                    state = loader.state_dict()
                    expected = [_ids(batch) for batch in loader]
                    self.assertIsNone(loader._thread)
                    loader.set_epoch(1)
                    self.assertEqual([_ids(batch) for batch in loader], [[0, 1], *expected])
                with _loader(**options) as resumed:
                    resumed.load_state_dict(state)
                    self.assertEqual([_ids(batch) for batch in resumed], expected)
