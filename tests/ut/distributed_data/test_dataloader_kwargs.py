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
"""Tests for user-supplied PyTorch DataLoader execution options."""

import inspect
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.parallel import build_dataset_batch_sampler
from hyper_parallel.distributed_data import (
    DistributedDatasetConfig,
    SampleMetadata,
    build_distributed_dataloader,
)
from hyper_parallel.distributed_data.api import _normalize_dataloader_kwargs
from hyper_parallel.distributed_data.dataset_reader import DatasetReader
from hyper_parallel.distributed_data.metadata import PlannedSampleLoader
from tests.common.mark_utils import arg_mark


class _StandaloneMesh:
    mesh_shape = (1,)
    mesh_dim_names = ("dp",)
    rank_list = (0,)


def _metadata_fn(sample: dict[str, int]) -> SampleMetadata:
    """Derive deterministic packing metadata for one sample."""
    return SampleMetadata(pack_tokens=sample["tokens"], sample_id=sample["id"])


def _worker_init_fn(worker_id: int) -> None:
    """Provide a module-level, spawn-compatible worker initializer."""
    del worker_id


class TestDataLoaderKwargs(unittest.TestCase):
    """Verify DataLoader execution options without ceding sample ownership."""

    @staticmethod
    def _config(**kwargs: object) -> DistributedDatasetConfig:
        """Build the smallest valid distributed-data configuration."""
        return DistributedDatasetConfig(seq_len=8, local_batch_size=1, **kwargs)

    @staticmethod
    def _samples() -> list[dict[str, int]]:
        """Return a mapping-style, metadata-compatible test Dataset."""
        return [{"id": 0, "tokens": 4}, {"id": 1, "tokens": 4}]

    @staticmethod
    def _execution_options() -> dict[str, object]:
        """Return valid non-default worker options for forwarding tests."""
        return {
            "num_workers": 2,
            "pin_memory": True,
            "prefetch_factor": 3,
            "persistent_workers": True,
            "timeout": 7,
            "worker_init_fn": _worker_init_fn,
            "multiprocessing_context": "spawn",
            "pin_memory_device": "npu",
            "in_order": True,
        }

    def test_public_builder_exposes_keyword_only_dataloader_kwargs(self) -> None:
        """The public API should accept optional DataLoader kwargs by keyword."""
        parameter = inspect.signature(build_distributed_dataloader).parameters["dataloader_kwargs"]

        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(parameter.default)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_dataloader_kwargs_override_config_without_mutating_either_input(self) -> None:
        """Feature: DataLoader execution overrides.
        Description: Normalize explicit options over config defaults.
        Expectation: Explicit kwargs take precedence without mutating either input.
        """
        config = self._config(
            num_workers=1,
            pin_memory=False,
            prefetch_factor=2,
            persistent_workers=False,
        )
        supplied = self._execution_options()
        original = supplied.copy()

        normalized = _normalize_dataloader_kwargs(config, supplied)

        for name, expected in original.items():
            self.assertIs(normalized[name], expected)
        self.assertEqual(supplied, original)
        self.assertEqual(config.num_workers, 1)
        self.assertFalse(config.pin_memory)
        self.assertEqual(config.prefetch_factor, 2)
        self.assertFalse(config.persistent_workers)

    def test_online_source_forwards_execution_options_to_native_dataloader(self) -> None:
        """Online Dataset Readers should pass normalized options to PyTorch."""
        supplied = self._execution_options()
        with patch("hyper_parallel.distributed_data.metadata.DataLoader") as dataloader_type:
            build_distributed_dataloader(
                self._samples(),
                _StandaloneMesh(),
                self._config(),
                metadata_fn=_metadata_fn,
                device="cpu", cost_model=lambda metadata: metadata.cost,
                dataloader_kwargs=supplied,
                batch_sampler=build_dataset_batch_sampler(
                    total_samples=2, micro_batch_size=1, global_batch_size=1, dp_world_size=1, dp_rank=0,
                ),
            )

        forwarded = dataloader_type.call_args.kwargs
        for name, expected in supplied.items():
            self.assertIs(forwarded[name], expected)
        self.assertIsNone(forwarded["batch_size"])
        self.assertIn("sampler", forwarded)
        self.assertIn("collate_fn", forwarded)
        self.assertIn("generator", forwarded)

    def test_metadata_reader_forwards_execution_options_to_native_dataloader(self) -> None:
        """Plan-aware metadata reads should use the same PyTorch worker options."""
        samples = self._samples()
        metadata = [_metadata_fn(sample) for sample in samples]
        supplied = self._execution_options()
        with patch("hyper_parallel.distributed_data.metadata.DataLoader") as dataloader_type:
            build_distributed_dataloader(
                samples,
                _StandaloneMesh(),
                self._config(metadata_mode=True),
                metadata=metadata,
                device="cpu", cost_model=lambda metadata: metadata.cost,
                dataloader_kwargs=supplied,
                batch_sampler=build_dataset_batch_sampler(
                    total_samples=2, micro_batch_size=1, global_batch_size=1, dp_world_size=1, dp_rank=0,
                ),
            )

        forwarded = dataloader_type.call_args.kwargs
        for name, expected in supplied.items():
            self.assertIs(forwarded[name], expected)
        self.assertIsNone(forwarded["batch_size"])
        self.assertIn("sampler", forwarded)
        self.assertIn("collate_fn", forwarded)
        self.assertIn("generator", forwarded)

    def test_rejects_framework_managed_and_unknown_options(self) -> None:
        """Users must not replace sample routing internals or arbitrary options."""
        cases = (
            ({"batch_size": 2}, "cannot override distributed sampling"),
            ({"sampler": object()}, "cannot override distributed sampling"),
            ({"collate_fn": list}, "cannot override distributed sampling"),
            ({"generator": torch.Generator()}, "cannot override distributed sampling"),
            ({"unknown_option": True}, "contains unsupported options"),
        )
        for supplied, expected_error in cases:
            with self.subTest(supplied=tuple(supplied)):
                with self.assertRaisesRegex(ValueError, expected_error):
                    _normalize_dataloader_kwargs(self._config(), supplied)

    def test_rejects_invalid_worker_option_combinations(self) -> None:
        """Invalid combinations should fail before native DataLoader iteration."""
        cases = (
            ({"num_workers": 0, "prefetch_factor": 2}, "prefetch_factor requires num_workers"),
            ({"num_workers": 0, "persistent_workers": True}, "persistent_workers=True requires"),
            ({"num_workers": 0, "timeout": 1}, "timeout must be zero"),
            ({"num_workers": 0, "multiprocessing_context": "spawn"}, "multiprocessing_context requires"),
            ({"num_workers": 1, "in_order": False}, "in_order must remain True"),
            ({"num_workers": 1, "worker_init_fn": object()}, "worker_init_fn must be callable"),
            ({"pin_memory_device": object()}, "pin_memory_device must be a string"),
        )
        for supplied, expected_error in cases:
            with self.subTest(supplied=tuple(supplied)):
                with self.assertRaisesRegex(ValueError, expected_error):
                    _normalize_dataloader_kwargs(self._config(), supplied)

    def test_worker_validation_is_shared_by_config_overrides_and_direct_readers(self) -> None:
        """Every entry point rejects the same invalid worker settings before creating workers."""
        cases = (
            {"num_workers": True}, {"num_workers": -1}, {"pin_memory": 1}, {"persistent_workers": 1},
            {"prefetch_factor": True}, {"prefetch_factor": 0}, {"prefetch_factor": 2},
            {"persistent_workers": True},
        )
        for invalid in cases:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self._config(**invalid)
                with self.assertRaises(ValueError):
                    _normalize_dataloader_kwargs(self._config(), invalid)
                options = {"num_workers": 0, "pin_memory": False, "prefetch_factor": None, "persistent_workers": False}
                options.update(invalid)
                with self.assertRaises(ValueError):
                    PlannedSampleLoader(self._samples(), seed=17, **options)
                with self.assertRaises(ValueError):
                    DatasetReader(
                        self._samples(), _metadata_fn, reader_rank=0, reader_idx=0, reader_count=1,
                        seq_len=8, shuffle=False, seed=17, **options,
                    )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_accepts_string_and_object_multiprocessing_contexts(self) -> None:
        """Feature: Native DataLoader execution options.
        Description: Validate both supported multiprocessing context representations.
        Expectation: The context and worker initializer reach the loader unchanged.
        """
        config = self._config(num_workers=1)
        for context in ("spawn", torch.multiprocessing.get_context("spawn")):
            with self.subTest(context=context):
                options = _normalize_dataloader_kwargs(
                    config, {"multiprocessing_context": context, "worker_init_fn": _worker_init_fn},
                )
                self.assertIs(options["multiprocessing_context"], context)
                self.assertIs(options["worker_init_fn"], _worker_init_fn)

    @staticmethod
    def _checkpoint_loader(config: DistributedDatasetConfig, **kwargs: object):
        """Build a deterministic loader with enough samples to resume multiple steps."""
        samples = [{"id": index, "tokens": 4} for index in range(6)]
        metadata_options = (
            {"metadata": [_metadata_fn(sample) for sample in samples]}
            if config.metadata_mode else {"metadata_fn": _metadata_fn}
        )
        return build_distributed_dataloader(
            samples, _StandaloneMesh(), config, **metadata_options,
            batch_sampler=build_dataset_batch_sampler(
                total_samples=len(samples), micro_batch_size=1, global_batch_size=1, dp_world_size=1, dp_rank=0,
            ),
            device="cpu", cost_model=lambda metadata: metadata.cost, **kwargs,
        )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_checkpoint_restore_allows_different_worker_options(self) -> None:
        """Feature: DataLoader checkpoint replay.
        Description: Resume online and metadata loaders with new config or kwargs worker settings.
        Expectation: Changing worker count and prefetch settings preserves remaining samples.
        """
        worker_options = {"num_workers": 1, "prefetch_factor": 2, "persistent_workers": True}
        extra_options = {"multiprocessing_context": "spawn", "worker_init_fn": _worker_init_fn, "timeout": 30}
        for metadata_mode in (False, True):
            config = self._config(metadata_mode=metadata_mode)
            with self._checkpoint_loader(config) as loader:
                next(loader)
                state = loader.state_dict()
                expected = list(loader)
            self.assertTrue(expected, "Checkpoint must have remaining samples to replay.")
            for from_config in (False, True):
                with self.subTest(metadata_mode=metadata_mode, from_config=from_config):
                    resumed_config = replace(config, **worker_options) if from_config else config
                    options = extra_options if from_config else {**worker_options, **extra_options}
                    with self._checkpoint_loader(resumed_config, dataloader_kwargs=options) as resumed:
                        resumed.load_state_dict(state)
                        self.assertEqual(list(resumed), expected, "Worker settings must not change checkpoint replay.")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_checkpoint_restore_allows_changed_packing_config(self) -> None:
        """Feature: DataLoader checkpoint compatibility.
        Description: Resume both metadata modes with changed packing settings and optional legacy fingerprints.
        Expectation: Compatible reader state restores the remaining samples without configuration hashing.
        """
        for metadata_mode in (False, True):
            config = self._config(metadata_mode=metadata_mode)
            with self._checkpoint_loader(config) as loader:
                next(loader)
                state = loader.state_dict()
                expected = list(loader)
            for removed_field in ("config_fingerprint", "topology_fingerprint", "last_plan_id"):
                self.assertNotIn(removed_field, state)
            self.assertTrue(expected, "Checkpoint must have remaining samples to replay.")
            changed = replace(config, seq_len=16, min_balance_gain=0.5)
            for legacy in (False, True):
                saved = dict(
                    state, config_fingerprint="legacy-config", topology_fingerprint="legacy-layout",
                    last_plan_id="legacy-plan",
                ) if legacy else state
                with self.subTest(metadata_mode=metadata_mode, legacy=legacy), self._checkpoint_loader(
                        changed,
                ) as resumed:
                    resumed.load_state_dict(saved)
                    self.assertEqual(list(resumed), expected, "Compatible packing settings must preserve replay.")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_checkpoint_restore_rejects_changed_reader_seed(self) -> None:
        """Feature: Deterministic reader checkpoint replay.
        Description: Restore online and metadata readers with a different worker seed.
        Expectation: Reader seed validation remains active without a loader configuration fingerprint.
        """
        for metadata_mode in (False, True):
            config = self._config(metadata_mode=metadata_mode)
            with self._checkpoint_loader(config) as loader:
                next(loader)
                state = loader.state_dict()
            changed = replace(config, seed=config.seed + 1)
            with self.subTest(metadata_mode=metadata_mode), self._checkpoint_loader(changed) as resumed:
                with self.assertRaisesRegex(ValueError, "checkpoint seed"):
                    resumed.load_state_dict(state)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_checkpoint_restore_rejects_another_rank(self) -> None:
        """Feature: Rank-local checkpoint ownership.
        Description: Restore a checkpoint saved by a different global rank in both loading modes.
        Expectation: Rank ownership is checked before restoring any reader progress.
        """
        for metadata_mode in (False, True):
            config = self._config(metadata_mode=metadata_mode)
            with self._checkpoint_loader(config) as loader:
                next(loader)
                state = loader.state_dict()
            state["global_rank"] = 1
            with self.subTest(metadata_mode=metadata_mode), self._checkpoint_loader(config) as resumed:
                with self.assertRaisesRegex(ValueError, "checkpoint global_rank"):
                    resumed.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
