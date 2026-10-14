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

from collections import Counter
from contextlib import nullcontext
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from hyper_parallel.data.parallel import build_dataset_batch_sampler
from hyper_parallel.distributed_data import (
    DistributedDatasetConfig, SampleMetadata, WorkloadCost, build_distributed_dataloader,
    collate_indexed_text_sequences, pack_indexed_text_samples, split_packed_text_sample,
)


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
    with no_a2a if metadata_mode else nullcontext():
        loader = build_distributed_dataloader(
            dataset if rank % 2 == 0 else None, mesh, config, batch_sampler=source_sampler, **metadata_options,
            cost_model=lambda metadata: metadata.cost,
        )
        saved_state = None
        delivered = []
        for round_idx, expected_local_indices in enumerate(reference):
            batch = next(loader)
            delivered.append(batch)
            _check_round_payloads(batch, expected_local_indices)
            plans = _gather(loader.last_plan)
            assert plans[0] == plans[2], f"Data-plane plans differ: plans={plans}"
            if rank % 2:
                assert source_sampler.consumed_samples == 0 and not dataset.reads, (
                    f"TP peer advanced sampler or read payloads: cursor={source_sampler.consumed_samples}, "
                    f"reads={dataset.reads}"
                )
            if round_idx == 0:
                saved_state = loader.state_dict()
                if rank % 2 == 0:
                    owner = "metadata_reader" if metadata_mode else "dataset_reader"
                    saved_cursor = saved_state[owner]["sampler"]["consumed_samples"]
                    assert saved_cursor == 4, f"Saved speculative sampler cursor: actual={saved_cursor}, expected=4"

        assert not list(loader), f"Native sampler should exhaust after {len(reference)} rounds"
        _check_dataset_reads(dataset, reference)
        resumed = build_distributed_dataloader(
            dataset if rank % 2 == 0 else None, mesh, config,
            batch_sampler=_sampler(sampler_type, data_sharding), **metadata_options,
            cost_model=lambda metadata: metadata.cost,
        )
        resumed.load_state_dict(saved_state)
        remaining = list(resumed)
        assert remaining == delivered[1:], f"Native replay differs: actual={remaining}, expected={delivered[1:]}"
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
    next(loader)
    plan = _gather(loader.last_plan)[0]
    actual_costs = [cost.llm for cost in plan.rank_costs]
    assert actual_costs == [10, 10], f"Expected costs=[10, 10] instead of native [18, 2], got costs={actual_costs}"
    actual_indices = sorted(key.dataset_index for key in plan.selected_keys)
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
            plans = _gather(loader.last_plan)
            assert plans[0] == plans[2], f"Repacked data-plane plans differ: plans={plans}"
            plan = plans[0]
            costs = [cost.llm for cost in plan.rank_costs]
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
