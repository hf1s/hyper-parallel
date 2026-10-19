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
"""DP2/TP2 native BatchSampler ownership, metadata, and checkpoint coverage."""

import logging
from collections import Counter
from contextlib import ExitStack, nullcontext
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from hyper_parallel.data.parallel import build_dataset_batch_sampler
from hyper_parallel.distributed_data import (
    DistributedDatasetConfig, SampleMetadata, WorkloadCost, build_distributed_dataloader,
    collate_indexed_text_sequences, pack_indexed_text_samples, split_packed_text_sample,
)
from hyper_parallel.distributed_data.balance_logging import logger as balance_logger


class _Dataset:
    """Track native index reads independently from metadata lookup."""

    def __init__(self) -> None:
        """Initialize the payload read log."""
        self.reads = []

    def __len__(self) -> int:
        """Return a size with an incomplete distributed tail."""
        return 18

    def __getitem__(self, index: int) -> dict:
        """Record a read and return original model fields."""
        self.reads.append(index)
        return {"id": index, "labels": [index + 1], "position_ids": [7], "loss_mask": [1]}


def _metadata(sample: dict) -> SampleMetadata:
    return SampleMetadata(1, cost=WorkloadCost(llm=9 if sample["id"] % 4 < 2 else 1), sample_id=sample["id"])


def _sampler(sampler_type: str, data_sharding: bool) -> object:
    return build_dataset_batch_sampler(
        total_samples=18, micro_batch_size=2, global_batch_size=8,
        dp_rank=dist.get_rank() // 2, dp_world_size=2,
        sampler_type=sampler_type, data_sharding=data_sharding, seed=17,
        index_mapping=[index // 2 for index in range(18)] if sampler_type == "single" else None,
    )


def _gather(value: object) -> list:
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def _check_round_payloads(batch: tuple[dict, ...], expected_local_indices: list[int]) -> None:
    """Check selected occurrences and whole-sample fields across DP and TP peers."""
    all_batches = _gather(batch)
    expected_indices = _gather(expected_local_indices)
    expected = Counter(expected_indices[0] + expected_indices[2])
    actual = Counter(sample["id"] for target in (0, 2) for sample in all_batches[target])
    assert actual == expected, f"Native round membership changed: actual={actual}, expected={expected}"
    assert all_batches[0] == all_batches[1] and all_batches[2] == all_batches[3], (
        f"TP peers received different batches: batches={all_batches}"
    )
    for sample in batch:
        assert sample["labels"] == [sample["id"] + 1] and sample["position_ids"] == [7], (
            f"Original native fields changed: sample={sample}"
        )


def _check_dataset_reads(dataset: _Dataset, reference: list[list[int]]) -> None:
    """Compare physical reads across ranks with all selected native occurrences."""
    all_reads = _gather(dataset.reads)
    actual_reads = Counter(index for rank_reads in all_reads for index in rank_reads)
    expected_reads = Counter(index for refs in _gather(reference)[::2] for row in refs for index in row)
    assert actual_reads == expected_reads, (
        f"Read coverage differs: actual={actual_reads}, expected={expected_reads}"
    )


def _run_case(
        mesh: object, *, metadata_mode: bool, sampler_type: str, data_sharding: bool, broadcast_plan: bool,
) -> list:
    rank = dist.get_rank()
    dataset = _Dataset()
    reference = list(_sampler(sampler_type, data_sharding))
    source_sampler = _sampler(sampler_type, data_sharding)
    config = DistributedDatasetConfig(
        seq_len=8, local_batch_size=2, communication_backend="gloo", metadata_mode=metadata_mode,
        broadcast_plan=broadcast_plan,
    )
    metadata_options = (
        {"metadata": [_metadata({"id": index}) for index in range(len(dataset))]}
        if metadata_mode else {"metadata_fn": _metadata}
    )
    no_a2a = patch.object(dist, "all_to_all_single", side_effect=AssertionError("metadata used A2A"))
    with (no_a2a if metadata_mode else nullcontext()), ExitStack() as resources:
        loader = resources.enter_context(build_distributed_dataloader(
            dataset if rank % 2 == 0 else None, mesh, config, batch_sampler=source_sampler, **metadata_options,
            cost_model=lambda metadata: metadata.cost,
        ))
        saved_state = None
        delivered = []
        for round_idx, expected_local_indices in enumerate(reference):
            batch = next(loader)
            delivered.append(batch)
            _check_round_payloads(batch, expected_local_indices)
            if rank % 2:
                assert source_sampler.consumed_samples == 0 and not dataset.reads, (
                    f"TP peer advanced sampler or read payloads: cursor={source_sampler.consumed_samples}, "
                    f"reads={dataset.reads}"
                )
            if round_idx == 1:
                saved_state = loader.state_dict()
                if rank % 2 == 0:
                    owner = "metadata_reader" if metadata_mode else "dataset_reader"
                    saved_cursor = saved_state[owner]["sampler"]["consumed_samples"]
                    assert saved_cursor == 8, f"Saved speculative sampler cursor: actual={saved_cursor}, expected=8"

        assert not list(loader), f"Native sampler should exhaust after {len(reference)} rounds"
        _check_dataset_reads(dataset, reference)
        resumed = resources.enter_context(build_distributed_dataloader(
            dataset if rank % 2 == 0 else None, mesh, config,
            batch_sampler=_sampler(sampler_type, data_sharding), **metadata_options,
            cost_model=lambda metadata: metadata.cost,
        ))
        resumed.load_state_dict(saved_state)
        remaining = list(resumed)
        assert remaining == delivered[2:], f"Native replay differs: actual={remaining}, expected={delivered[2:]}"
    return delivered


def _run_cost_balance(mesh: object, broadcast_plan: bool) -> None:
    """Redistribute unequal native samples while keeping the selected four outputs."""
    sampler = build_dataset_batch_sampler(
        total_samples=18, micro_batch_size=2, global_batch_size=8,
        dp_rank=dist.get_rank() // 2, dp_world_size=2,
    )
    loader = build_distributed_dataloader(
        _Dataset(), mesh, DistributedDatasetConfig(
            seq_len=8, local_batch_size=2, communication_backend="gloo", broadcast_plan=broadcast_plan,
        ),
        batch_sampler=sampler, metadata_fn=_metadata,
        cost_model=lambda metadata: metadata.cost,
    )
    batches = _gather(next(loader))
    actual_costs = [sum(_metadata(sample).cost.llm for sample in batches[rank]) for rank in (0, 2)]
    assert actual_costs == [10, 10], f"Expected costs=[10, 10] instead of native [18, 2], got costs={actual_costs}"
    actual_indices = sorted(sample["id"] for rank in (0, 2) for sample in batches[rank])
    assert actual_indices == [0, 1, 2, 3], f"Cost balancing changed the native round: indices={actual_indices}"
    loader.close()


def test_native_batch_sampler_dp2_tp2_gloo() -> None:
    """Check native DP slicing and collective replay under both data paths."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        for metadata_mode in (False, True):
            for sampler_type, data_sharding in (("single", False), ("cyclic", False), ("cyclic", True)):
                options = {
                    "metadata_mode": metadata_mode, "sampler_type": sampler_type, "data_sharding": data_sharding,
                }
                baseline = _run_case(mesh, **options, broadcast_plan=True)
                independent = _run_case(mesh, **options, broadcast_plan=False)
                assert independent == baseline, (
                    f"Broadcast mode changed batches: actual={independent}, expected={baseline}"
                )
        for broadcast_plan in (True, False):
            _run_cost_balance(mesh, broadcast_plan)
            _run_sequence_repacking(mesh, broadcast_plan)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _window_loader(
        mesh: object, *, metadata_mode: bool, repack: bool, broadcast_plan: bool, accumulation: int,
) -> object:
    records = [
        {"id": index, "parts": [{"id": index, "part": part} for part in range(2)]}
        for index in range(44)
    ]
    sampler = build_dataset_batch_sampler(
        total_samples=len(records), micro_batch_size=1, global_batch_size=2 * accumulation,
        dp_rank=dist.get_rank() // 2, dp_world_size=2,
    )

    def metadata(sample: dict) -> SampleMetadata:
        """Assign uneven native-record costs to exercise cross-round placement.

        Args:
            sample: A native record or one independently repacked sequence.
        """
        tokens = 1 if "part" in sample else 2
        return SampleMetadata(tokens, cost=WorkloadCost(llm=tokens * (9 if sample["id"] % 2 == 0 else 1)))

    options = {"metadata": [metadata(sample) for sample in records]} if metadata_mode else {"metadata_fn": metadata}
    if repack:
        options.update(split_fn=lambda sample: sample["parts"], pack_fn=lambda samples, _: tuple(samples))
    return build_distributed_dataloader(
        records, mesh,
        DistributedDatasetConfig(
            seq_len=4, local_batch_size=1, communication_backend="gloo", balance_window_size=3,
            metadata_mode=metadata_mode, broadcast_plan=broadcast_plan,
        ),
        batch_sampler=sampler, cost_model=lambda item: item.cost, device="cpu", **options,
    )


def _window_output(batch: tuple, repack: bool) -> list[tuple[int, int]]:
    samples = [sample for packed in batch for sample in packed] if repack else batch
    return [(sample["id"], sample.get("part", -1)) for sample in samples]


def _check_window_membership(outputs: list, repack: bool, accumulation: int) -> None:
    assert outputs[0] == outputs[1] and outputs[2] == outputs[3], f"TP batches differ: outputs={outputs}"
    start = 0
    while start < len(outputs[0]):
        rounds = min(3, accumulation - start % accumulation, len(outputs[0]) - start)
        stop = start + rounds
        actual = Counter(item for rank in (0, 2) for batch in outputs[rank][start:stop] for item in batch)
        expected = Counter(
            (index, part) for index in range(start * 2, stop * 2) for part in (range(2) if repack else (-1,))
        )
        assert actual == expected, f"Window membership differs: actual={actual}, expected={expected}"
        start = stop


def test_accumulation_window_dp2_tp2_gloo() -> None:
    """Verify real payload redistribution, TP broadcast, and incomplete-step replay."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        for accumulation in (3, 10):
            for metadata_mode, repack in ((False, False), (True, False), (False, True)):
                for broadcast_plan in (False, True):
                    options = {
                        "metadata_mode": metadata_mode, "repack": repack,
                        "broadcast_plan": broadcast_plan, "accumulation": accumulation,
                    }
                    with _window_loader(mesh, **options) as loader:
                        first = _window_output(next(loader), repack)
                        state = loader.state_dict()
                        remaining = [_window_output(batch, repack) for batch in loader]
                    outputs = _gather([first] + remaining)
                    _check_window_membership(outputs, repack, accumulation)
                    with _window_loader(mesh, **options) as resumed:
                        resumed.load_state_dict(state)
                        actual = [_window_output(batch, repack) for batch in resumed]
                        expected = [first, *remaining]
                        assert actual == expected, f"Step replay differs: actual={actual}, expected={expected}"
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _check_repacked_tokens(records: list[dict], gathered: list[dict]) -> None:
    """Check token and target conservation across the two DP representatives."""
    expected = Counter(
        (token, label) for record in records
        for token, label in zip(record["input_ids"].tolist(), record["labels"].tolist())
    )
    actual = Counter(
        (token, label) for rank in (0, 2)
        for token, label in zip(gathered[rank]["input_ids"][0], gathered[rank]["labels"][0])
        if token != 0
    )
    assert actual == expected, f"Repacking changed selected tokens/labels: actual={actual}, expected={expected}"


def _run_sequence_repacking(mesh: object, broadcast_plan: bool) -> None:
    """Balance independent sequences with TP replication and per-rank cursor replay."""
    records = []
    for index in range(6):
        lengths = [6, 6] if index % 2 == 0 else [2, 2, 2, 2]
        ids = torch.arange(index * 20 + 1, index * 20 + sum(lengths) + 1)
        labels = ids + 1
        labels[::3] = -100
        records.append({
            "input_ids": ids, "labels": labels,
            "cu_seq_lens": torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()]),
        })

    def build_loader() -> object:
        """Build a fresh reader and sampler for checkpoint replay."""
        return build_distributed_dataloader(
            records if dist.get_rank() % 2 == 0 else None, mesh,
            DistributedDatasetConfig(
                seq_len=12, local_batch_size=1, communication_backend="gloo", broadcast_plan=broadcast_plan,
            ),
            batch_sampler=build_dataset_batch_sampler(
                total_samples=6, micro_batch_size=1, global_batch_size=2,
                dp_rank=dist.get_rank() // 2, dp_world_size=2,
            ),
            split_fn=split_packed_text_sample, pack_fn=pack_indexed_text_samples,
            collate_fn=collate_indexed_text_sequences,
            metadata_fn=lambda sample: SampleMetadata(sample["input_ids"].numel()),
            cost_model=lambda item: WorkloadCost(llm=item.pack_tokens ** 2),
        )

    delivered = []
    with build_loader() as loader:
        for step, batch in enumerate(loader):
            serializable = {name: value.tolist() for name, value in batch.items()}
            delivered.append(serializable)
            gathered = _gather(serializable)
            assert gathered[0] == gathered[1] and gathered[2] == gathered[3], (
                f"TP batch mismatch after sequence repacking: batches={gathered}"
            )
            _check_repacked_tokens(records[step * 2:step * 2 + 2], gathered)
            costs = [
                sum((end - start) ** 2
                    for start, end in zip(gathered[rank]["cu_seq_lens"], gathered[rank]["cu_seq_lens"][1:])
                    if gathered[rank]["input_ids"][0][start] != 0)
                for rank in (0, 2)
            ]
            assert costs == [44, 44], f"Expected repacked costs [44, 44], got {costs}"
            if step == 0:
                saved_state = loader.state_dict()
                if dist.get_rank() % 2 == 0:
                    cursor = saved_state["dataset_reader"]["sampler"]["consumed_samples"]
                    assert cursor == 2, f"Prefetched record progress was saved: cursor={cursor}"
    with build_loader() as resumed:
        resumed.load_state_dict(saved_state)
        remaining = [{name: value.tolist() for name, value in batch.items()} for batch in resumed]
    assert remaining == delivered[1:], f"Repacked checkpoint replay differs: actual={remaining}"


def _check_full_pack_documents(records: list[dict], gathered: list[dict]) -> None:
    """Compare each received attention segment with its complete original document."""
    for rank in (0, 2):
        value = gathered[rank]
        offsets = value["cu_seq_lens"]
        assert offsets[0] == 0 and offsets[-1] == 10, f"Invalid repaired boundaries: offsets={offsets}"
        for start, end in zip(offsets, offsets[1:]):
            document = value["input_ids"][0][start:end]
            original = records[(document[0] - 1) // 10]
            original_start = (document[0] - 1) % 10
            original_offsets = original["cu_seq_lens"].tolist()
            source_index = original_offsets.index(original_start)
            expected_end = original_offsets[source_index + 1]
            expected = original["input_ids"][original_start:expected_end].tolist()
            assert document == expected, f"Document boundary changed: actual={document}, expected={expected}"


def _full_pack_records() -> tuple[list[dict], dict[int, int]]:
    """Create full records whose token lengths lead greedy LPT into a dead end."""
    records, costs = [], {}
    for index in range(14):
        lengths = (6, 2, 2) if index % 2 == 0 else (5, 3, 2)
        weights = (90, 70, 50) if index % 2 == 0 else (100, 80, 60)
        ids = torch.arange(index * 10 + 1, index * 10 + 11)
        labels = ids + 1
        labels[::3] = -100
        offsets = [0, *torch.tensor(lengths).cumsum(0).tolist()]
        records.append({"input_ids": ids, "labels": labels, "cu_seq_lens": torch.tensor(offsets)})
        costs.update({int(ids[start]): weight for start, weight in zip(offsets, weights)})
    return records, costs


def _full_pack_statistics(gathered: list[dict], costs: dict[int, int]) -> tuple[Counter, Counter, int]:
    """Count delivered token/document occurrences and the slowest DP pack's cost."""
    tokens, documents = Counter(), Counter()
    rank_costs = []
    for rank in (0, 2):
        value = gathered[rank]
        tokens.update(zip(value["input_ids"][0], value["labels"][0]))
        starts = [value["input_ids"][0][start] for start in value["cu_seq_lens"][:-1]]
        documents.update(starts)
        rank_costs.append(sum(costs[token] for token in starts))
    return tokens, documents, max(rank_costs)


def _run_full_pack_repair(mesh: object, accumulation: int, broadcast_plan: bool) -> None:
    """Exercise repaired full packs, repeated records, window tails, and checkpoint replay."""
    records, costs = _full_pack_records()
    mapping = [0, 1] * 7

    def build_loader() -> object:
        """Create a fresh sampler and source reader for the same occurrence stream."""
        return build_distributed_dataloader(
            records, mesh, DistributedDatasetConfig(
                seq_len=10, local_batch_size=1, communication_backend="gloo", balance_window_size=3,
                broadcast_plan=broadcast_plan,
            ),
            batch_sampler=build_dataset_batch_sampler(
                total_samples=len(records), micro_batch_size=1, global_batch_size=2 * accumulation,
                dp_rank=dist.get_rank() // 2, dp_world_size=2, index_mapping=mapping,
            ),
            device="cpu", split_fn=split_packed_text_sample, pack_fn=pack_indexed_text_samples,
            collate_fn=collate_indexed_text_sequences,
            metadata_fn=lambda sample: SampleMetadata(
                sample["input_ids"].numel(), cost=WorkloadCost(llm=costs[int(sample["input_ids"][0])]),
            ),
            cost_model=lambda item: item.cost,
        )

    delivered, window_tokens, window_documents = [], Counter(), Counter()
    window_objective = 0
    saved = None
    with build_loader() as loader:
        for step, batch in enumerate(loader):
            output = {name: value.tolist() for name, value in batch.items()}
            delivered.append(output)
            gathered = _gather(output)
            assert gathered[0] == gathered[1] and gathered[2] == gathered[3], (
                f"Repaired TP batches differ: batches={gathered}"
            )
            _check_full_pack_documents(records, gathered)
            tokens, documents, objective = _full_pack_statistics(gathered, costs)
            window_tokens.update(tokens)
            window_documents.update(documents)
            window_objective += objective
            if step == accumulation - 1:
                saved = loader.state_dict()
            if (step + 1) % accumulation == 0 or step == 6:
                start = step // accumulation * accumulation
                assert window_objective < (step + 1 - start) * 240, (
                    f"Full-pack repair did not improve the delivered window: cost={window_objective}"
                )
                window_objective = 0
                expected_documents = Counter(
                    int(records[mapping[position]]["input_ids"][offset])
                    for position in range(start * 2, (step + 1) * 2)
                    for offset in records[mapping[position]]["cu_seq_lens"][:-1].tolist()
                )
                expected_tokens = Counter(
                    (token, label) for position in range(start * 2, (step + 1) * 2)
                    for token, label in zip(records[mapping[position]]["input_ids"].tolist(),
                                            records[mapping[position]]["labels"].tolist())
                )
                assert window_documents == expected_documents, (
                    f"Document occurrences changed: actual={window_documents}, expected={expected_documents}"
                )
                assert window_tokens == expected_tokens, (
                    f"Tokens or labels changed: actual={window_tokens}, expected={expected_tokens}"
                )
                window_tokens.clear()
                window_documents.clear()
    assert saved is not None, f"No checkpoint was captured: delivered={len(delivered)}"
    with build_loader() as restored:
        restored.load_state_dict(saved)
        actual = [{name: value.tolist() for name, value in batch.items()} for batch in restored]
    assert actual == delivered[accumulation:], f"Repaired replay differs: {actual} vs {delivered[accumulation:]}"


def test_full_pack_repair_dp2_tp2_gloo() -> None:
    """Verify repairs with and without gradient accumulation under both plan modes."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        for accumulation in (1, 3):
            for broadcast_plan in (False, True):
                _run_full_pack_repair(mesh, accumulation, broadcast_plan)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _gather_balance_logs(caplog: pytest.LogCaptureFixture, planner_rank: int, expected_count: int) -> list[str]:
    """Check that only the planner emits the expected number of window reports."""
    messages = _gather([record.getMessage() for record in caplog.records if record.name == balance_logger.name])
    counts = [len(records) for records in messages]
    expected_counts = [expected_count if rank == planner_rank else 0 for rank in range(4)]
    assert counts == expected_counts, f"Duplicate or missing logs: actual={counts}, expected={expected_counts}"
    return messages[planner_rank]


def _run_balance_logging_case(
        mesh: object, caplog: pytest.LogCaptureFixture, *, metadata_mode: bool, broadcast_plan: bool,
        planner_rank: int, min_gain: float,
) -> None:
    """Verify one writer, modeled window gains and deferred speculative output."""
    dataset = _Dataset()

    def metadata(sample: dict) -> SampleMetadata:
        """Assign alternating costs to the two native DP sources.

        Args:
            sample: Original indexed source sample.
        """
        return SampleMetadata(1, cost=WorkloadCost(llm=9 if sample["id"] % 2 == 0 else 1))

    options = {"metadata": [metadata(dataset[index]) for index in range(len(dataset))]} if metadata_mode else {
        "metadata_fn": metadata,
    }
    caplog.clear()
    with build_distributed_dataloader(
            dataset, mesh, DistributedDatasetConfig(
                seq_len=8, local_batch_size=1, communication_backend="gloo", balance_window_size=3,
                metadata_mode=metadata_mode, broadcast_plan=broadcast_plan,
                planner_rank=planner_rank, min_balance_gain=min_gain, log_balance=True,
            ),
            batch_sampler=build_dataset_batch_sampler(
                total_samples=len(dataset), micro_batch_size=1, global_batch_size=10,
                dp_rank=dist.get_rank() // 2, dp_world_size=2,
            ),
            cost_model=lambda item: item.cost, **options,
    ) as loader:
        loader.prefetch()
        loader.wait_for_prefetch()
        assert not caplog.records, f"Unconsumed step emitted logs: {caplog.records}"
        first = next(loader)
        loader.wait_for_prefetch()
        first_window = _gather_balance_logs(caplog, planner_rank, expected_count=1)[0]
        assert "iteration 1 :" in first_window and "micro_batches=1-3" in first_window, (
            f"Incorrect first window identity: {first_window}"
        )
        assert f"planner_rank={planner_rank}" in first_window, f"Incorrect planner rank: {first_window}"
        assert "constructor_ranks=(0, 2)" in first_window, f"Incorrect DP rank mapping: {first_window}"
        assert "original=27, candidate=19," in first_window, f"Incorrect window costs: {first_window}"
        accepted = min_gain == 0
        assert f"accepted={accepted}" in first_window, f"Incorrect acceptance decision: {first_window}"
        assert f"accepted_objective={19 if accepted else 27}" in first_window, (
            f"Incorrect accepted objective: {first_window}"
        )
        batches = _gather(first)
        costs = [metadata(batches[rank][0]).cost.llm for rank in (0, 2)]
        expected_costs = [9, 9] if accepted else [9, 1]
        assert costs == expected_costs, f"Logged assignment differs: actual={costs}, expected={expected_costs}"
        for _ in range(3):
            next(loader)
        second_window = _gather_balance_logs(caplog, planner_rank, expected_count=2)[1]
        assert "iteration 1 :" in second_window and "micro_batches=4-5" in second_window, (
            f"Incorrect second window identity: {second_window}"
        )


def test_native_balance_logging_dp2_tp2_gloo(caplog: pytest.LogCaptureFixture) -> None:
    """Check INFO reports on the chosen planner under both metadata and planning paths.

    Args:
        caplog: Per-process log capture, without replacing the logging callbacks.
    """
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        with caplog.at_level(logging.INFO, logger=balance_logger.name):
            for metadata_mode, broadcast_plan, planner_rank, min_gain in (
                    (False, True, 0, 0.0), (False, False, 2, 0.0),
                    (True, True, 2, 0.9), (True, False, 0, 0.0),
            ):
                _run_balance_logging_case(
                    mesh, caplog, metadata_mode=metadata_mode, broadcast_plan=broadcast_plan,
                    planner_rank=planner_rank, min_gain=min_gain,
                )
        dist.barrier()
    finally:
        dist.destroy_process_group()
