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
"""Prepare complete optimizer steps while training consumes the previous step."""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Iterator, Mapping
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Any

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed_data.device_prefetch import _record_stream
from hyper_parallel.distributed_data.distributed_dataloader import DistributedDataLoader
from hyper_parallel.distributed_data.schema import DistributedPackingPlan


@dataclass
class _PreparedMicroBatch:
    """Retain one device batch and its delivered-progress snapshot."""

    batch: Any
    plan: DistributedPackingPlan | None
    checkpoint: dict[str, Any]


@dataclass
class _PreparedOptimizerStep:
    """Publish an optimizer step after every input copy and broadcast finishes."""

    batches: deque[_PreparedMicroBatch]
    exhausted: bool
    ready_event: Any = None


class OptimizerStepDataLoader(Iterator[Any]):
    """Double-buffer complete optimizer steps and yield one micro-batch at a time.

    The source is owned exclusively by one producer. Per-delivery snapshots
    prevent its speculative progress from entering training checkpoints.
    Data collectives use the source's dedicated groups, including model-peer
    input broadcasts, so foreground training never enters those groups.
    """

    collective_source = True

    def __init__(
            self, source: DistributedDataLoader, *, accumulation_steps: int,
            first_step_rounds: int, device: Any,
    ) -> None:
        """Store a fresh native source and its optimizer-step boundaries.

        Args:
            source: Unstarted distributed loader exclusively owned by this wrapper.
            accumulation_steps: Micro-batches per complete optimizer step.
            first_step_rounds: Remaining rounds before the first optimizer boundary.
            device: Rank-local training device, including model-consumer ranks.
        """
        if isinstance(accumulation_steps, bool) or not isinstance(accumulation_steps, int) or accumulation_steps < 1:
            raise ValueError("accumulation_steps must be a positive integer.")
        if (
                isinstance(first_step_rounds, bool) or not isinstance(first_step_rounds, int)
                or not 1 <= first_step_rounds <= accumulation_steps
        ):
            raise ValueError("first_step_rounds must be within the accumulation step count.")
        self._source = source
        self._accumulation_steps = accumulation_steps
        self._first_step_rounds: int | None = first_step_rounds
        self._device = torch.device(device)
        self._current: _PreparedOptimizerStep | None = None
        self._future: Future[_PreparedOptimizerStep] | None = None
        self._requests: Queue[Future[_PreparedOptimizerStep] | None] = Queue(maxsize=1)
        self._thread: Thread | None = None
        self._closed = False
        self._last_plan: DistributedPackingPlan | None = None
        self._delivered_state = source.state_dict()
        self._delivered_state["optimizer_step_remaining"] = first_step_rounds

    def __iter__(self) -> OptimizerStepDataLoader:
        """Return the rank-local iterator."""
        return self

    def __next__(self) -> Any:
        """Activate a ready step and start its successor before handing off data."""
        if self._closed or self._delivered_state["stopped"]:
            raise StopIteration
        if self._current is None or not self._current.batches:
            if self._current is not None and self._current.exhausted:
                self._delivered_state["stopped"] = True
                self.close()
                raise StopIteration
            self.prefetch()
            self._current = self._future.result()
            self._future = None
            if not self._current.batches:
                self._delivered_state["stopped"] = True
                self.close()
                raise StopIteration
            if not self._current.exhausted:
                self.prefetch()
        prepared = self._current.batches.popleft()
        if self._current.ready_event is not None:
            module = getattr(torch, self._device.type)
            stream = module.current_stream(self._device)
            stream.wait_event(self._current.ready_event)
            _record_stream(prepared.batch, stream)
        self._delivered_state = prepared.checkpoint
        self._last_plan = prepared.plan
        return prepared.batch

    @property
    def last_plan(self) -> DistributedPackingPlan | None:
        """Return the last delivered plan, excluding speculative production."""
        return self._last_plan

    def prefetch_plan(self) -> None:
        """Let existing Trainer hooks request the next complete step."""
        self.prefetch()

    def prefetch(self) -> None:
        """Submit at most one future step to the persistent producer."""
        if self._closed or self._future is not None or self._delivered_state["stopped"]:
            return
        if self._current is not None and self._current.exhausted:
            return
        self._future = Future()
        if self._thread is None:
            self._thread = Thread(target=self._run, name="hp-optimizer-step-prefetch", daemon=True)
            self._thread.start()
        self._requests.put(self._future)

    def _run(self) -> None:
        stream = None
        try:
            while True:
                future = self._requests.get()
                if future is None:
                    return
                try:
                    if stream is None and self._device.type in ("npu", "cuda"):
                        module = getattr(torch, self._device.type)
                        module.set_device(self._device)
                        stream = module.Stream(device=self._device)
                    context = getattr(torch, self._device.type).stream(stream) if stream is not None else nullcontext()
                    with context:
                        step = self._prepare_step(stream)
                except BaseException as exc:
                    future.set_exception(exc)
                else:
                    future.set_result(step)
        finally:
            if stream is not None:
                stream.synchronize()

    def _prepare_step(self, stream: Any) -> _PreparedOptimizerStep:
        rounds = self._first_step_rounds
        if rounds is None:
            rounds = self._source.optimizer_step_remaining()
        self._first_step_rounds = self._accumulation_steps
        batches = deque()
        exhausted = False
        for index in range(rounds):
            try:
                batch = self._source.prepare_batch()
            except StopIteration:
                exhausted = True
                break
            state = self._source.state_dict()
            state["optimizer_step_remaining"] = rounds - index - 1 or self._accumulation_steps
            batches.append(_PreparedMicroBatch(batch, self._source.last_plan, state))
        step = _PreparedOptimizerStep(batches, exhausted)
        if stream is not None:
            step.ready_event = getattr(torch, self._device.type).Event()
            step.ready_event.record(stream)
            # Publish only after all H2D and input broadcasts in this step finish.
            # Training on the other stream continues while this producer waits.
            step.ready_event.synchronize()
        return step

    def state_dict(self) -> dict[str, Any]:
        """Snapshot delivered progress without waiting for the next step."""
        return copy.deepcopy(self._delivered_state)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore delivered progress before starting the producer.

        Args:
            state: Rank-local checkpoint produced at a micro-batch boundary.
        """
        if self._thread is not None or self._current is not None:
            raise ValueError("load_state_dict must run before optimizer-step iteration starts.")
        remaining = state.get("optimizer_step_remaining")
        if remaining is not None and (
                isinstance(remaining, bool) or not isinstance(remaining, int)
                or not 1 <= remaining <= self._accumulation_steps
        ):
            raise ValueError("optimizer_step_remaining must be within the accumulation step count.")
        self._source.load_state_dict(state)
        self._delivered_state = copy.deepcopy(dict(state))
        self._delivered_state["optimizer_step_remaining"] = remaining
        self._first_step_rounds = remaining

    def wait_for_prefetch(self) -> None:
        """Drain complete-step communication before checkpoint teardown or shutdown."""
        if self._future is not None:
            self._future.result()

    def close(self) -> None:
        """Drain the producer before releasing buffered inputs and source resources."""
        if self._thread is not None:
            if self._thread.is_alive():
                self._requests.put(None)
            self._thread.join()
            self._thread = None
        self._source.close()
        self._current = None
        self._future = None
        self._closed = True

    def set_epoch(self, epoch: int) -> None:
        """Reset an exhausted iterator and its native source for another epoch.

        Args:
            epoch: Non-negative Dataset epoch.
        """
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer.")
        if self._delivered_state["step"] != 0 and not self._delivered_state["stopped"]:
            raise ValueError("set_epoch requires a fresh or exhausted optimizer-step loader.")
        if epoch == self._delivered_state["epoch"] and not self._delivered_state["stopped"]:
            return
        remaining = self._delivered_state["optimizer_step_remaining"] if epoch == self._delivered_state["epoch"] else (
            self._accumulation_steps
        )
        self.close()
        self._source.set_epoch(epoch)
        self._requests = Queue(maxsize=1)
        self._closed = False
        self._last_plan = None
        self._delivered_state = self._source.state_dict()
        self._first_step_rounds = remaining
        self._delivered_state["optimizer_step_remaining"] = self._first_step_rounds

    def __enter__(self) -> OptimizerStepDataLoader:
        """Support deterministic producer cleanup."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Drain prefetch before process-group teardown."""
        self.close()


__all__ = ["OptimizerStepDataLoader"]
