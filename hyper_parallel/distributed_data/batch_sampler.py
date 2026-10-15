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
"""Transactional Dataset reads driven by HP's native DP BatchSampler."""

from __future__ import annotations

import copy
import operator
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, replace
from queue import Queue
from threading import Thread
from typing import Any

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed_data.schema import SampleKey, SampleMetadata
from hyper_parallel.distributed_data.metadata import PlannedSampleLoader


def validate_native_sampler(sampler: Any, *, data_rank: int, dp_size: int, local_batch_size: int) -> None:
    """Validate the native HP sampler interface, DP layout, and cursor alignment.

    Args:
        sampler: Native HP batch sampler to validate.
        data_rank: Expected DP coordinate for this rank.
        dp_size: Expected number of DP replicas.
        local_batch_size: Expected samples per local forward/backward round.

    Raises:
        ValueError: If the sampler interface, DP settings, or cursor is incompatible.
    """
    for name in ("__iter__", "state_dict", "load_state_dict", "set_epoch", "enable_source_batch_resume"):
        if not callable(getattr(sampler, name, None)):
            raise ValueError(f"batch_sampler must implement HP's native sampler contract: missing {name}.")
    expected = {"dp_rank": data_rank, "dp_world_size": dp_size, "micro_batch_size": local_batch_size,
                "drop_last": True}
    for name, value in expected.items():
        if getattr(sampler, name, None) != value:
            raise ValueError(f"batch_sampler.{name} must equal {value!r}.")
    if sampler.consumed_samples != sampler.total_samples and sampler.consumed_samples % (dp_size * local_batch_size):
        raise ValueError("batch_sampler.consumed_samples must align to a synchronized forward/backward round.")


def preserve_sample(samples: Sequence[Any], seq_len: int) -> Any:
    """Pass one native Dataset output unchanged to the original collator.

    Args:
        samples: Source samples for one native sequence bin.
        seq_len: Sequence capacity, unused when preserving a Dataset output.

    Returns:
        The sole source sample, unchanged.
    """
    del seq_len
    if len(samples) != 1:
        raise ValueError("Native BatchSampler mode must keep one Dataset output per sequence bin.")
    return samples[0]


@dataclass
class _ReaderWindow:
    """Keep one window's data and cursor independent of speculative sampling."""

    metadata: tuple[SampleMetadata, ...]
    samples: dict[SampleKey, Any]
    sampler_state: dict[str, Any]
    rounds: int
    delivered_rounds: int = 0


class BatchSamplerReader:
    """Read a bounded window of native batches within one optimizer step.

    Only the DP Constructor owns this reader. Payload loading is deferred until
    after planning when metadata is available; otherwise it precedes metadata
    extraction. The sampler's speculative cursor is never a checkpoint cursor.
    """

    def __init__(
            self,
            sampler: Any,
            *,
            reader_rank: int,
            metadata: Sequence[SampleMetadata] | None,
            metadata_fn: Callable[[Any], SampleMetadata] | None,
            sample_loader: PlannedSampleLoader | None,
            split_fn: Callable[[Any], Sequence[Any]] | None = None,
            balance_window_size: int = 1,
            device: Any = None,
    ) -> None:
        """Store the validated native sampler and metadata/payload sources."""
        if metadata is None and (sample_loader is None or not callable(metadata_fn)):
            raise ValueError("Online BatchSampler Reader requires payload loading and metadata_fn.")
        if split_fn is not None and (not callable(split_fn) or metadata is not None):
            raise ValueError("split_fn requires online metadata and must be callable.")
        self._sampler = sampler
        self._reader_rank = reader_rank
        self._metadata = metadata
        self._metadata_fn = metadata_fn
        self._sample_loader = sample_loader
        self._split_fn = split_fn
        self._balance_window_size = balance_window_size
        self._device = torch.device(device) if device is not None else None
        self._window: _ReaderWindow | None = None
        self._next_window: Future[_ReaderWindow | None] | None = None
        self._window_thread: Thread | None = None
        self._window_requests: Queue[Future[_ReaderWindow | None] | None] = Queue(maxsize=1)
        self._sampler.enable_source_batch_resume()
        self._committed_state = copy.deepcopy(sampler.state_dict())
        self._iterator = None
        self._exhausted = False

    @property
    def exhausted(self) -> bool:
        """Return whether the native sampler has exhausted its epoch."""
        return self._exhausted

    @property
    def buffer_size(self) -> int:
        """Return the number of pending native sample occurrences."""
        return len(self.metadata())

    @property
    def batch_position(self) -> int:
        """Return the native global cursor before this pending round."""
        return self._committed_state["consumed_samples"]

    def prepare_next_step(self) -> None:
        """Activate the next window only after the current window is delivered."""
        if self._window is not None or self._exhausted:
            return
        if self._next_window is None:
            self._window = self._read_window()
        else:
            self._window = self._next_window.result()
            self._next_window = None
        self._exhausted = self._window is None

    def _read_window(self) -> _ReaderWindow | None:
        # Only one read runs at a time. It must not mutate the active window
        # while that window's samples are being routed and committed.
        try:
            if self._iterator is None:
                self._iterator = iter(self._sampler)
            keys, rounds = self._read_window_keys()
            if not keys:
                return None
            samples = {}
            if self._metadata is None:
                samples = self._sample_loader.fetch_samples(keys)
                if self._split_fn is not None:
                    samples = self._split_payloads(samples)
                    keys = tuple(samples)
                metadata = [self._metadata_fn(samples[key]) for key in keys]
            else:
                metadata = [self._metadata[key.dataset_index] for key in keys]
            if any(not isinstance(item, SampleMetadata) for item in metadata):
                raise ValueError("BatchSampler metadata must contain SampleMetadata entries.")
            return _ReaderWindow(
                metadata=tuple(replace(item, key=key) for key, item in zip(keys, metadata)),
                samples=samples,
                sampler_state=copy.deepcopy(self._sampler.state_dict()),
                rounds=rounds,
            )
        except StopIteration as exc:
            # A user callback's StopIteration must not masquerade as dataset EOF.
            raise RuntimeError(f"Native BatchSampler callback raised StopIteration: {exc}") from exc

    def _read_window_keys(self) -> tuple[tuple[SampleKey, ...], int]:
        round_size = self._sampler.global_micro_batch_size
        accumulation_steps = self._sampler.global_batch_size // round_size
        round_in_step = self._sampler.consumed_samples // round_size % accumulation_steps
        window_size = min(self._balance_window_size, accumulation_steps - round_in_step)
        keys = []
        rounds = 0
        for _ in range(window_size):
            position = self._sampler.consumed_samples
            try:
                indices = next(self._iterator)
            except StopIteration:
                break
            keys.extend(self._sample_keys(indices, position))
            rounds += 1
        return tuple(keys), rounds

    def _split_payloads(self, samples: Mapping[SampleKey, Any]) -> dict[SampleKey, Any]:
        payloads = {}
        for key, sample in samples.items():
            sequences = tuple(self._split_fn(sample))
            if not sequences:
                raise ValueError("split_fn must return at least one independent sequence per Dataset output.")
            for index, sequence in enumerate(sequences):
                payloads[replace(key, sequence_index=index)] = sequence
        return payloads

    def _prefetch_next_window(self) -> None:
        if self._balance_window_size == 1 or self._next_window is not None:
            return
        self._next_window = Future()
        if self._window_thread is None:
            # DataLoader may fork workers here. A plain thread avoids inheriting
            # ThreadPoolExecutor shutdown hooks that try to join the forked thread.
            self._window_thread = Thread(
                target=self._run_window_prefetch,
                name=f"hp-window-reader-rank-{self._reader_rank}", daemon=True,
            )
            self._window_thread.start()
        self._window_requests.put(self._next_window)

    def _run_window_prefetch(self) -> None:
        device_bound = False
        while True:
            future = self._window_requests.get()
            if future is None:
                return
            try:
                if not device_bound:
                    # New DataLoader pin-memory workers inherit this thread's device.
                    device_module = getattr(torch, self._device.type, None) if self._device is not None else None
                    if device_module is not None and hasattr(device_module, "set_device"):
                        device_module.set_device(self._device)
                    device_bound = True
                window = self._read_window()
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(window)

    def close(self) -> None:
        """Drain speculative CPU reads before resetting the sampler or exiting."""
        if self._window_thread is not None:
            self._window_requests.put(None)
            self._window_thread.join()
            self._window_thread = None
        self._next_window = None

    def _sample_keys(self, indices: Sequence[int], batch_position: int) -> tuple[SampleKey, ...]:
        if len(indices) != self._sampler.micro_batch_size:
            raise ValueError("Native BatchSampler must yield exactly local_batch_size indices on every DP rank.")
        # A native sampler exposes local indices, not its internal permutation
        # positions. Stable DP-major occurrence slots preserve duplicates.
        start = batch_position + self._sampler.dp_rank * self._sampler.micro_batch_size
        keys = []
        for offset, index in enumerate(indices):
            if isinstance(index, bool):
                raise ValueError("Native BatchSampler indices must be non-negative integers.")
            keys.append(SampleKey(self._reader_rank, operator.index(index), start + offset))
        return tuple(keys)

    def metadata(self) -> tuple[SampleMetadata, ...]:
        """Return metadata for the undelivered occurrences in the frozen window."""
        return () if self._window is None else self._window.metadata

    def selected_payloads(self, selected_keys: set[SampleKey]) -> tuple[tuple[SampleKey, Any], ...]:
        """Return already-read payloads without rereading their Dataset indices.

        Args:
            selected_keys: Sample occurrences selected from this Reader's buffer.

        Returns:
            Selected key-payload pairs in buffered sample order.
        """
        return tuple(
            (item.key, self._window.samples[item.key]) for item in self.metadata() if item.key in selected_keys
        )

    def commit(self, selected_keys: set[SampleKey]) -> None:
        """Retire delivered occurrences and commit the cursor at the window end.

        Args:
            selected_keys: Source-owned occurrences delivered in this distributed round.
        """
        window = self._window
        remaining = {item.key for item in self.metadata()}
        if window is None or not selected_keys <= remaining:
            raise ValueError("Native BatchSampler commit must consume the complete pending local batch.")
        last_round = window.delivered_rounds + 1 == window.rounds
        if last_round and selected_keys != remaining:
            raise ValueError("Native BatchSampler commit must consume the complete pending local batch.")
        window.metadata = tuple(item for item in window.metadata if item.key not in selected_keys)
        for key in selected_keys:
            window.samples.pop(key, None)
        window.delivered_rounds += 1
        if last_round:
            self._committed_state = window.sampler_state
            self._window = None
        self._prefetch_next_window()

    @property
    def window_partially_delivered(self) -> bool:
        """Return whether a checkpoint must retain the remaining window plan."""
        return self._window is not None and self._window.delivered_rounds > 0

    def state_dict(self) -> dict[str, Any]:
        """Snapshot the delivered native cursor, excluding speculative prefetch."""
        state = {
            "epoch": self._committed_state["epoch"],
            "sampler": self._committed_state,
            "sample_loader": self._sample_loader.state_dict() if self._sample_loader is not None else None,
        }
        if self.window_partially_delivered:
            state["window"] = {
                "sampler": self._window.sampler_state,
                "metadata": self._window.metadata,
                "rounds": self._window.rounds,
                "delivered_rounds": self._window.delivered_rounds,
            }
        return copy.deepcopy(state)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the original native cursor before the next sampler yield.

        Args:
            state: Reader checkpoint containing the committed native sampler state.

        Raises:
            ValueError: If the checkpoint epoch or payload reader is incompatible.
        """
        sampler_state = state["sampler"]
        if sampler_state.get("epoch") != state.get("epoch"):
            raise ValueError("Native BatchSampler checkpoint epoch is inconsistent.")
        if (state.get("sample_loader") is None) != (self._sample_loader is None):
            raise ValueError("Native BatchSampler checkpoint payload reader ownership changed.")
        self.close()
        self._sampler.load_state_dict(sampler_state)
        if self._sample_loader is not None:
            self._sample_loader.load_state_dict(state["sample_loader"])
        self._committed_state = copy.deepcopy(self._sampler.state_dict())
        self._reset_pending()
        if "window" in state:
            self._restore_window(state["window"])

    def _restore_window(self, window: Mapping[str, Any]) -> None:
        self._sampler.load_state_dict(window["sampler"])
        metadata = tuple(window["metadata"])
        samples = {}
        if self._metadata is None:
            remaining_keys = {item.key for item in metadata}
            source_keys = tuple(dict.fromkeys(replace(item.key, sequence_index=0) for item in metadata))
            samples = self._sample_loader.fetch_samples(source_keys)
            if self._split_fn is not None:
                samples = self._split_payloads(samples)
            samples = {key: sample for key, sample in samples.items() if key in remaining_keys}
        self._window = _ReaderWindow(
            metadata=metadata, samples=samples, sampler_state=copy.deepcopy(window["sampler"]),
            rounds=window["rounds"], delivered_rounds=window["delivered_rounds"],
        )

    def set_epoch(self, epoch: int) -> None:
        """Delegate epoch changes to the native sampler without changing its policy.

        Args:
            epoch: Epoch to apply to the sampler and sample loader.
        """
        self.close()
        self._sampler.load_state_dict(self._committed_state)
        self._sampler.set_epoch(epoch)
        if self._sample_loader is not None:
            self._sample_loader.set_epoch(epoch)
        self._committed_state = copy.deepcopy(self._sampler.state_dict())
        self._reset_pending()

    def _reset_pending(self) -> None:
        self._iterator = None
        self._window = None
        self._exhausted = False
