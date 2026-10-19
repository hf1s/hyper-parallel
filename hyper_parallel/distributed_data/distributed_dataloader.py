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
"""Dataset Reader, Planner, and Data Constructor orchestration."""

from __future__ import annotations

import copy
import logging
import math
from collections import deque
from collections.abc import Iterator, Mapping
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass, field
from queue import Queue
from threading import Thread
from typing import Any, Union

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed_data.balance_logging import format_balance_stats, logger as balance_logger
from hyper_parallel.distributed_data.data_constructor import PackingDataConstructor
from hyper_parallel.distributed_data.batch_sampler import BatchSamplerReader
from hyper_parallel.distributed_data.device_prefetch import DevicePrefetchedStep, DeviceStepPrefetcher, _record_stream
from hyper_parallel.distributed_data.planner import DynamicPackingPlanner
from hyper_parallel.distributed_data.schema import (
    DistributedPackingPlan,
    SampleKey,
    SampleMetadata,
)
from hyper_parallel.distributed_data.metadata import PlannedSampleLoader
from hyper_parallel.distributed_data.topology import DataTopology
from hyper_parallel.distributed_data.transport import (
    DataPlaneTransport,
    ModelParallelTransport,
    PreparedPayloadExchange,
)


_PlanControl = Union[DistributedPackingPlan, tuple[DistributedPackingPlan, ...], None]


def _validate_loader_components(
        *,
        topology: DataTopology,
        dataset_reader_ranks: tuple[int, ...],
        dataset_reader: Any | None,
        metadata_reader: BatchSamplerReader | None,
        direct_sample_loader: PlannedSampleLoader | None,
        metadata_mode: bool,
) -> None:
    if not isinstance(metadata_mode, bool):
        raise ValueError("metadata_mode must be boolean.")
    if metadata_mode and dataset_reader is not None:
        raise ValueError("Metadata mode must not configure an online Dataset Reader.")
    if not metadata_mode and (metadata_reader is not None or direct_sample_loader is not None):
        raise ValueError("Online mode must not configure metadata loading components.")
    is_reader = topology.global_rank in dataset_reader_ranks
    planning_reader = metadata_reader if metadata_mode else dataset_reader
    if is_reader != (planning_reader is not None):
        raise ValueError("Dataset Reader ownership does not match dataset_reader_ranks.")
    if not metadata_mode and planning_reader is not None and not callable(
            getattr(planning_reader, "prepare_next_step", None)
    ):
        raise ValueError("Online Dataset Reader must provide prepare_next_step().")
    _validate_metadata_loader_owner(
        topology,
        metadata_mode=metadata_mode,
        direct_sample_loader=direct_sample_loader,
    )


def _validate_metadata_loader_owner(
        topology: DataTopology,
        *,
        metadata_mode: bool,
        direct_sample_loader: PlannedSampleLoader | None,
) -> None:
    if not metadata_mode:
        return
    if topology.is_constructor != (direct_sample_loader is not None):
        raise ValueError("Every metadata Data Constructor must own one plan-aware sample loader.")


@dataclass
class _PreparedWindow:
    """Retain one balancing window and an optional complete-step checkpoint."""

    batches: deque[Any]
    exhausted: bool
    checkpoint: dict[str, Any] | None
    ready_event: Any = None
    balance_logs: list[str] = field(default_factory=list)


class DistributedDataLoader(Iterator[Any]):
    """Prepare balancing windows and deliver one local micro-batch at a time.

    A single producer owns reads, planning, data collectives and device staging
    across epochs, until the loader is explicitly closed.
    Buffer the current window and at most one future window within step boundaries.
    Checkpoints advance at complete optimizer-step delivery boundaries.
    """

    collective_source = True

    def __init__(
            self,
            *,
            topology: DataTopology,
            dataset_reader_ranks: tuple[int, ...],
            dataset_reader: Any | None,
            metadata_reader: BatchSamplerReader | None,
            direct_sample_loader: PlannedSampleLoader | None,
            metadata_mode: bool,
            planner: DynamicPackingPlanner,
            data_constructor: PackingDataConstructor,
            data_plane: DataPlaneTransport,
            model_transport: ModelParallelTransport,
            device_prefetch: DeviceStepPrefetcher | None = None,
            initial_epoch: int = 0,
            broadcast_plan: bool = True,
            log_balance: bool = False,
            accumulation_steps: int = 1,
            balance_window_size: int = 1,
            device: Any = "cpu",
    ) -> None:
        """Store the fully validated runtime components."""
        _validate_loader_components(
            topology=topology,
            dataset_reader_ranks=dataset_reader_ranks,
            dataset_reader=dataset_reader,
            metadata_reader=metadata_reader,
            direct_sample_loader=direct_sample_loader,
            metadata_mode=metadata_mode,
        )
        self._topology = topology
        self._dataset_reader_ranks = frozenset(dataset_reader_ranks)
        self._dataset_reader = dataset_reader
        self._metadata_reader = metadata_reader
        self._direct_sample_loader = direct_sample_loader
        self._metadata_mode = metadata_mode
        self._broadcast_plan = broadcast_plan
        self._log_balance = log_balance
        self._planner = planner
        self._data_constructor = data_constructor
        self._data_plane = data_plane
        self._model_transport = model_transport
        self._device_prefetch = device_prefetch
        self._epoch = initial_epoch
        self._step = 0
        self._stopped = False
        self._pending_local_keys: set[SampleKey] = set()
        self._window_plans: deque[DistributedPackingPlan] = deque()
        for name, value in (("accumulation_steps", accumulation_steps), ("balance_window_size", balance_window_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self._accumulation_steps = accumulation_steps
        self._balance_window_size = balance_window_size
        self._device = torch.device(device)
        self._current: _PreparedWindow | None = None
        self._future: Future[_PreparedWindow] | None = None
        self._requests: Queue[Future[_PreparedWindow] | None] = Queue(maxsize=1)
        self._prefetch_thread: Thread | None = None
        self._closed = False
        self._balance_logs: list[str] = []
        self._delivered_state = self._snapshot_state()

    def __iter__(self) -> DistributedDataLoader:
        """Return the rank-local iterator."""
        return self

    def __next__(self) -> Any:
        """Order training after a prepared window and start its successor before delivery."""
        if self._closed or self._delivered_state["stopped"]:
            raise StopIteration
        if self._current is None or not self._current.batches:
            if self._current is not None and self._current.exhausted:
                self._delivered_state["stopped"] = True
                raise StopIteration
            self.prefetch()
            self._current = self._future.result()
            self._future = None
            if not self._current.batches:
                self._delivered_state["stopped"] = True
                raise StopIteration
            if not self._current.exhausted:
                self.prefetch()
            for message in self._current.balance_logs:
                balance_logger.info("%s", message)
            self._current.balance_logs.clear()
        batch = self._current.batches.popleft()
        if self._current.ready_event is not None:
            module = getattr(torch, self._device.type)
            stream = module.current_stream(self._device)
            stream.wait_event(self._current.ready_event)
            _record_stream(batch, stream)
        if not self._current.batches and self._current.checkpoint is not None:
            self._delivered_state = self._current.checkpoint
        return batch

    def prefetch(self) -> None:
        """Submit at most one future balancing window to the persistent producer."""
        if self._closed or self._future is not None or self._delivered_state["stopped"]:
            return
        if self._current is not None and self._current.exhausted:
            return
        self._future = Future()
        if self._prefetch_thread is None:
            self._prefetch_thread = Thread(
                target=self._run_prefetch, name=f"hp-data-prefetch-rank-{self._topology.global_rank}", daemon=True,
            )
            self._prefetch_thread.start()
        self._requests.put(self._future)

    def _run_prefetch(self) -> None:
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
                    prefetch_stream_context = (
                        getattr(torch, self._device.type).stream(stream) if stream is not None else nullcontext()
                    )
                    with prefetch_stream_context:
                        window = self._prepare_window(stream)
                except BaseException as exc:
                    future.set_exception(exc)
                else:
                    future.set_result(window)
                    del window
                # An idle producer must not retain a discarded epoch's inputs.
                del future
        finally:
            if stream is not None:
                stream.synchronize()

    def _prepare_window(self, stream: Any) -> _PreparedWindow:
        micro_batch_num = min(
            self._balance_window_size, self._accumulation_steps - self._step % self._accumulation_steps,
        )
        self._balance_logs = []
        batches = deque()
        exhausted = False
        for _ in range(micro_batch_num):
            try:
                batch = self._prepare_batch()
            except StopIteration:
                exhausted = True
                break
            batches.append(batch)
        # Intermediate windows must replay from the optimizer step's beginning.
        checkpoint = self._snapshot_state() if not exhausted and self._step % self._accumulation_steps == 0 else None
        window = _PreparedWindow(batches, exhausted, checkpoint, balance_logs=self._balance_logs)
        if stream is not None:
            window.ready_event = getattr(torch, self._device.type).Event()
            window.ready_event.record(stream)
        return window

    def _prepare_batch(self) -> Any:
        """Construct one round on the owning window producer without source lookahead.

        The producer serializes calls on its device-bound thread.
        Keeping all data collectives on that thread prevents training hooks
        from racing ahead of an unfinished data transaction.
        """
        if self._stopped:
            raise StopIteration
        batch = self._prepare_next_batch()
        if isinstance(batch, DevicePrefetchedStep):
            batch = batch.take_batch()
        return self._consume_batch(self._broadcast_batch(batch))

    def _stage_batch(self, batch: Any) -> Any:
        return self._device_prefetch(batch) if batch is not None and self._device_prefetch is not None else batch

    def _prepare_next_batch(self) -> Any:
        """Select, balance, route, and construct samples without model-group broadcast."""
        if self._data_plane.is_member:
            return self._stage_batch(self._produce_on_data_plane())
        return None

    def _broadcast_batch(self, constructed_batch: Any) -> Any:
        """Broadcast the constructed batch to model-parallel peers on the producer thread."""
        return self._model_transport.broadcast(constructed_batch)

    def _consume_batch(self, received: Any) -> Any:
        """Validate the batch, commit Reader progress, and return its training data."""
        if received is None:
            self._stopped = True
            self._pending_local_keys.clear()
            raise StopIteration
        # Dataset Reader buffers are committed only after construction and broadcast
        # have both succeeded, leaving checkpoint boundaries unambiguous.
        planning_reader = self._planning_reader()
        if planning_reader is not None:
            planning_reader.commit(self._pending_local_keys, prefetch_window=False)
        self._pending_local_keys.clear()
        self._stopped = False
        self._step += 1
        return received

    def wait_for_prefetch(self) -> None:
        """Drain buffered window communication before checkpoint teardown or shutdown."""
        window = self._future.result() if self._future is not None else self._current
        if window is not None and window.ready_event is not None:
            # The latest event also covers earlier windows on the same producer stream.
            window.ready_event.synchronize()

    def close(self) -> None:
        """Drain the producer before releasing buffered inputs and reader resources."""
        if self._prefetch_thread is not None:
            if self._prefetch_thread.is_alive():
                self._requests.put(None)
            self._prefetch_thread.join()
            self._prefetch_thread = None
        try:
            if self._device_prefetch is not None:
                self._device_prefetch.close()
        finally:
            reader = self._planning_reader()
            if reader is not None:
                reader.close()
            self._current = None
            self._future = None
            self._balance_logs = []
            self._closed = True

    def __enter__(self) -> "DistributedDataLoader":
        """Support deterministic cleanup on early loop exit."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Release prefetch resources before process-group teardown."""
        self.close()

    def state_dict(self) -> dict[str, Any]:
        """Snapshot the last fully delivered optimizer step without waiting.

        Save training checkpoints only after the corresponding optimizer update.
        An incomplete step leaves this cursor at that step's beginning.
        """
        return copy.deepcopy(self._delivered_state)

    def _snapshot_state(self) -> dict[str, Any]:
        """Capture producer progress at an optimizer-step boundary."""
        state = {
            "global_rank": self._topology.global_rank,
            "epoch": self._epoch,
            "step": self._step,
            "stopped": self._stopped,
            "dataset_reader": self._dataset_reader.state_dict() if self._dataset_reader is not None else None,
            "metadata_reader": self._metadata_reader.state_dict() if self._metadata_reader is not None else None,
            "direct_sample_loader": (
                self._direct_sample_loader.state_dict() if self._direct_sample_loader is not None else None
            ),
        }
        try:
            copied_state = copy.deepcopy(state)
        except Exception as exc:
            raise ValueError(f"Distributed DataLoader state is not checkpointable: {exc}") from exc
        return copied_state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore a fixed-topology rank-local checkpoint.

        Args:
            state_dict: State produced on this same global rank.
        """
        self._validate_restore_boundary()
        try:
            state = copy.deepcopy(dict(state_dict))
        except Exception as exc:
            raise ValueError(f"Distributed DataLoader state is not copyable: {exc}") from exc
        if state.get("global_rank") != self._topology.global_rank:
            raise ValueError(
                f"Distributed DataLoader checkpoint global_rank={state.get('global_rank')!r} "
                f"does not match {self._topology.global_rank!r}."
            )
        epoch, step, stopped = self._validate_checkpoint_values(state)
        reader_state, metadata_reader_state, direct_sample_state = self._validate_component_states(state, epoch)
        if (
                step % self._accumulation_steps or state.get("window_plans")
                or state.get("optimizer_step_remaining", self._accumulation_steps) != self._accumulation_steps
        ):
            raise ValueError("Distributed DataLoader checkpoint must align to a complete optimizer step.")
        self._restore_component_states(reader_state, metadata_reader_state, direct_sample_state)
        self._epoch = epoch
        self._step = step
        self._stopped = stopped

        self._delivered_state = self._snapshot_state()

    def _validate_restore_boundary(self) -> None:
        if self._prefetch_thread is not None or self._current is not None or self._step != 0:
            raise ValueError("load_state_dict must run before distributed iteration starts.")

    @staticmethod
    def _validate_checkpoint_values(state: Mapping[str, Any]) -> tuple[int, int, bool]:
        epoch = state.get("epoch")
        step = state.get("step")
        stopped = state.get("stopped")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("Distributed DataLoader checkpoint epoch must be non-negative.")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("Distributed DataLoader checkpoint step must be non-negative.")
        if not isinstance(stopped, bool):
            raise ValueError("Distributed DataLoader checkpoint stopped must be boolean.")
        return epoch, step, stopped

    def _validate_component_states(
            self,
            state: Mapping[str, Any],
            epoch: int,
    ) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None, Mapping[str, Any] | None]:
        reader_state = state.get("dataset_reader")
        # Accept the old field name when restoring checkpoints written before the rename.
        metadata_reader_state = state.get("metadata_reader", state.get("sidecar_reader"))
        direct_sample_state = state.get("direct_sample_loader")
        if (self._dataset_reader is None) != (reader_state is None):
            raise ValueError("Distributed DataLoader checkpoint Dataset Reader ownership changed.")
        if (self._metadata_reader is None) != (metadata_reader_state is None):
            raise ValueError("Distributed DataLoader checkpoint metadata ownership changed.")
        if (self._direct_sample_loader is None) != (direct_sample_state is None):
            raise ValueError("Distributed DataLoader checkpoint direct-reader ownership changed.")
        for component_name, component_state in (
                ("Dataset Reader", reader_state),
                ("metadata reader", metadata_reader_state),
                ("direct reader", direct_sample_state),
        ):
            if component_state is None:
                continue
            if not isinstance(component_state, Mapping):
                raise ValueError(f"Distributed DataLoader checkpoint {component_name} state must be a mapping.")
            if component_state.get("epoch") != epoch:
                raise ValueError(
                    f"Distributed DataLoader checkpoint {component_name} epoch "
                    f"{component_state.get('epoch')!r} does not match loader epoch {epoch}."
                )
        return reader_state, metadata_reader_state, direct_sample_state

    def _restore_component_states(
            self,
            reader_state: Mapping[str, Any] | None,
            metadata_reader_state: Mapping[str, Any] | None,
            direct_sample_state: Mapping[str, Any] | None,
    ) -> None:
        if self._dataset_reader is not None:
            self._dataset_reader.load_state_dict(reader_state)
        if self._metadata_reader is not None:
            self._metadata_reader.load_state_dict(metadata_reader_state)
        if self._direct_sample_loader is not None:
            self._direct_sample_loader.load_state_dict(direct_sample_state)

    def set_epoch(self, epoch: int) -> None:
        """Reset the native source while retaining the producer thread and stream.

        Args:
            epoch: Non-negative Dataset epoch.
        """
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer.")
        if self._closed:
            raise ValueError("set_epoch cannot restart a closed loader.")
        if (self._current is not None or self._delivered_state["step"] != 0) and not self._delivered_state["stopped"]:
            raise ValueError("set_epoch requires a fresh or exhausted optimizer-step loader.")
        if epoch == self._delivered_state["epoch"] and not self._delivered_state["stopped"]:
            return
        self.wait_for_prefetch()
        if self._device_prefetch is not None:
            self._device_prefetch.close()
        if self._dataset_reader is not None:
            self._dataset_reader.set_epoch(epoch)
        if self._metadata_reader is not None:
            self._metadata_reader.set_epoch(epoch)
        if self._direct_sample_loader is not None:
            self._direct_sample_loader.set_epoch(epoch)
        self._epoch = epoch
        self._step = 0
        self._stopped = False
        self._pending_local_keys.clear()
        self._window_plans.clear()
        self._current = None
        self._future = None
        self._balance_logs = []
        self._delivered_state = self._snapshot_state()

    def _produce_on_data_plane(self) -> Any:
        plan = self._next_plan_control()
        if plan is None:
            return None

        plan = self._set_pending_plan(plan)
        local_selected_keys = self._pending_local_keys
        if self._metadata_mode:
            return self._produce_metadata_batch(plan)

        prepared_exchange = self._prepare_outgoing(plan, local_selected_keys)
        received_payloads = self._data_plane.exchange_prepared(prepared_exchange)
        return self._construct_received_payloads(plan, received_payloads)

    def _set_pending_plan(
            self, plan: _PlanControl,
    ) -> DistributedPackingPlan | None:
        if isinstance(plan, tuple):
            self._window_plans = deque(plan)
        if self._window_plans:
            plan = self._window_plans.popleft()
        self._pending_local_keys = set() if plan is None else {
            key for key in plan.selected_keys if key.reader_rank == self._topology.global_rank
        }
        return plan

    def _produce_metadata_batch(self, plan: DistributedPackingPlan) -> Any:
        """Directly read constructor-assigned shared indices without payload A2A."""
        received_payloads: dict[SampleKey, Any] = {}
        if self._topology.is_constructor:
            received_payloads = self._direct_sample_loader.fetch_samples(
                plan.local_sample_keys(self._topology.data_rank)
            )
        return self._construct_received_payloads(plan, received_payloads)

    def _construct_received_payloads(
            self,
            plan: DistributedPackingPlan,
            received_payloads: dict[SampleKey, Any],
    ) -> Any:
        """Construct one local batch and return it on the Data Constructor rank."""
        if not self._topology.is_constructor:
            return None
        local_batch_plan = plan.local_batch_for(self._topology.data_rank)
        local_batch = self._data_constructor.construct(local_batch_plan, received_payloads)
        return self._require_batch(local_batch)

    @staticmethod
    def _require_batch(local_batch: Any) -> Any:
        """Reject ``None`` because it is reserved for distributed end-of-stream."""
        if local_batch is None:
            raise ValueError("The Data Constructor must return a non-None batch.")
        return local_batch

    def _next_plan_control(self) -> _PlanControl:
        """Reuse the current window or plan a new one; None means end of stream."""
        local_metadata = self._prepare_local_metadata()
        if self._window_plans:
            return self._window_plans[0]
        if not self._broadcast_plan:
            return self._build_plan_control(self._data_plane.all_gather_object(local_metadata))
        gathered_metadata = self._data_plane.gather_object_to_planner(local_metadata)
        plan = None
        if self._topology.global_rank == self._data_plane.planner_rank:
            plan = self._build_plan_control(gathered_metadata)
        return self._data_plane.broadcast_from_planner(plan)

    def _prepare_local_metadata(self) -> tuple[SampleMetadata, ...]:
        """Prepare the local sampler window and return its planning metadata."""
        if self._topology.global_rank not in self._dataset_reader_ranks:
            return ()
        planning_reader = self._planning_reader()
        planning_reader.prepare_next_step()
        return planning_reader.metadata()

    def _build_plan_control(
            self, gathered_metadata: tuple[tuple[SampleMetadata, ...], ...],
    ) -> _PlanControl:
        """Return the native window's plans; None means end of stream."""
        # Collective order identifies the source rank even when its metadata is empty.
        reader_metadata = [
            metadata for rank, metadata in zip(self._data_plane.ranks, gathered_metadata)
            if rank in self._dataset_reader_ranks
        ]
        return self._plan_native_batch(reader_metadata)

    def _plan_native_batch(
            self, reader_metadata: list[tuple[SampleMetadata, ...]],
    ) -> _PlanControl:
        """Plan the native sampler window, returning one plan per training round."""
        if all(not metadata for metadata in reader_metadata):
            return None
        if any(not metadata for metadata in reader_metadata):
            raise ValueError("Native BatchSampler ranks exhausted at different forward/backward rounds.")
        counts = {len({item.key.global_sample_position for item in metadata}) for metadata in reader_metadata}
        if len(counts) != 1 or next(iter(counts)) % self._planner.local_batch_size:
            raise ValueError("Native BatchSampler ranks must provide equal complete micro-batch windows.")
        samples = tuple(sorted(
            (item for metadata in reader_metadata for item in metadata),
            key=lambda item: (item.key.global_sample_position, item.key.sequence_index),
        ))
        original_bins = {}
        for item in samples:
            original_bins.setdefault(item.key.global_sample_position, []).append(item.key)
        plan = self._planner.plan if next(iter(counts)) == self._planner.local_batch_size else self._planner.plan_window
        reference_bins = tuple(tuple(keys) for keys in original_bins.values())
        plans = plan(samples, reference_bins=reference_bins, step=self._step)
        self._capture_balance_log(samples, reference_bins, plans)
        return plans

    def _capture_balance_log(
            self, samples: tuple[SampleMetadata, ...], reference_bins: tuple[tuple[SampleKey, ...], ...],
            plans: DistributedPackingPlan | tuple[DistributedPackingPlan, ...],
    ) -> None:
        """Freeze scalar window statistics before the producer advances to another plan."""
        if (
            not self._log_balance
            or self._topology.global_rank != self._data_plane.planner_rank
            or not balance_logger.isEnabledFor(logging.INFO)
        ):
            return
        plans = plans if isinstance(plans, tuple) else (plans,)
        metadata = {item.key: item for item in samples}
        costs = self._planner.last_sample_costs

        def describe(keys: tuple[SampleKey, ...]) -> dict[str, Any]:
            """Use evaluated costs without invoking the cost model again.

            Args:
                keys: Sample occurrences assigned to one pack.
            """
            return {
                "seq_len": sum(metadata[key].pack_tokens for key in keys),
                "samples": len(keys), "cost": math.fsum(costs[key].llm for key in keys),
            }

        local_size = self._planner.local_batch_size
        bin_count = self._planner.distributed_bin_count
        rounds = []
        for offset, plan in enumerate(plans):
            original = reference_bins[offset * bin_count:(offset + 1) * bin_count]
            rounds.append({
                "micro_batch": plan.step % self._accumulation_steps + 1,
                "bins_before": tuple(
                    tuple(describe(keys) for keys in original[start:start + local_size])
                    for start in range(0, bin_count, local_size)
                ),
                "bins_after": tuple(
                    tuple(describe(packing_bin.sample_keys) for packing_bin in local_batch)
                    for local_batch in plan.local_batches
                ),
            })
        stats = {
            **self._planner.last_balance_decision,
            "objective": self._planner.objective, "epoch": self._epoch,
            "planner_rank": self._data_plane.planner_rank,
            "constructor_ranks": self._topology.constructor_ranks,
            "group_ranks": tuple(range(self._topology.data_parallel_size)), "rounds": rounds,
        }
        self._balance_logs.append(format_balance_stats(stats, self._step // self._accumulation_steps + 1))

    def _prepare_outgoing(
            self,
            plan: DistributedPackingPlan,
            local_selected_keys: set[SampleKey],
    ) -> PreparedPayloadExchange:
        outgoing: dict[int, list[tuple[SampleKey, Any]]] = {}
        if local_selected_keys:
            payloads = self._dataset_reader.selected_payloads(local_selected_keys)
            target_by_key = {
                key: self._topology.constructor_ranks[data_rank]
                for data_rank, local_batch in enumerate(plan.local_batches)
                for packing_bin in local_batch
                for key in packing_bin.sample_keys
            }
            for key, payload in payloads:
                outgoing.setdefault(target_by_key[key], []).append((key, payload))
        return self._data_plane.prepare_exchange(outgoing)

    def _planning_reader(self) -> Any | None:
        """Return this rank's online or metadata-only Dataset Reader."""
        if self._dataset_reader is not None:
            return self._dataset_reader
        return self._metadata_reader



__all__ = ["DistributedDataLoader"]
