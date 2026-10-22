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
"""Unit tests for the trainer checkpoint callback."""

import os
import tempfile
import traceback
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from hyper_parallel.data.batching import FixedBatchDataLoader
from hyper_parallel.data.parallel.batch_sampler import build_dataset_batch_sampler
from hyper_parallel.data.parallel.dataloader_parallel import DataLoaderParallelContext
from hyper_parallel.trainer.callbacks.checkpoint_callback import CheckpointerCallback
from hyper_parallel.trainer.state import TrainerState
from tests.common.mark_utils import arg_mark
from tests.common.port_utils import allocate_port


def _checkpoint_config(restore_from: str) -> SimpleNamespace:
    """Build the checkpoint configuration used by callback unit tests."""
    return SimpleNamespace(
        save_ckpt=False,
        checkpoint_dir=restore_from,
        save_steps=0,
        save_epochs=0,
        is_async=False,
        is_peft=False,
        save_optimizer=False,
        save_train_state=False,
        save_extra_state_per_rank=False,
        restore_from=restore_from,
        restore_optimizer=False,
        restore_train_state=False,
        restore_dataloader_state=True,
    )


def _loader(dp_rank: int, dp_world_size: int) -> FixedBatchDataLoader:
    """Build an eight-sample loader whose optimizer batch holds two samples."""
    sampler = build_dataset_batch_sampler(total_samples=8, micro_batch_size=1, global_batch_size=2,
                                          dp_rank=dp_rank, dp_world_size=dp_world_size,
                                          sampler_type="single", seed=0)
    return FixedBatchDataLoader(list(range(8)), batch_sampler=sampler, collate_fn=list)


def _layout_callback(loader: FixedBatchDataLoader, tp_world_size: int) -> CheckpointerCallback:
    """Bind a restoring callback to one layout of the two-process world, where TP rank 0 reads input."""
    tp_rank = dist.get_rank() % tp_world_size
    parallel_context = DataLoaderParallelContext(
        build_on_rank=lambda: tp_rank == 0,
        distributed_enabled=True,
        tp_rank=tp_rank,
        tp_world_size=tp_world_size,
        tp_group=dist.group.WORLD,
    )
    callback = CheckpointerCallback.__new__(CheckpointerCallback)
    callback._restore_dataloader_state = True  # pylint: disable=protected-access
    callback.trainer = SimpleNamespace(
        state=TrainerState(),
        data_iterator=None,
        train_dataloader=loader,
        train_steps=4,
        lr_scheduler=None,
        get_batch=SimpleNamespace(parallel_context=parallel_context),
    )
    return callback


def _tp_resume_worker(rank: int, port: int, errors: Any) -> None:
    """Save from TP2, then restore the per-rank states with and without TP."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    try:
        dist.init_process_group("gloo", rank=rank, world_size=2)
        tp_loader = _loader(dp_rank=0, dp_world_size=1)
        unread_state = tp_loader.state_dict()
        if rank == 0:
            batches = iter(tp_loader)
            next(batches)
            next(batches)
        saved = _layout_callback(tp_loader, tp_world_size=2)._collect_extra_state(  # pylint: disable=protected-access
            TrainerState(global_step=1)
        )
        position = saved["train_dataloader"]["_index_sampler_state"]["consumed_samples"]
        assert position == 2, f"rank {rank} saved position {position}, expected the TP reader's position 2"

        # Without TP, rank 1 reads input itself and must continue after the saved optimizer batch.
        loader = _loader(dp_rank=rank, dp_world_size=2)
        _layout_callback(loader, tp_world_size=1)._apply_extra_state(saved)  # pylint: disable=protected-access
        batch = next(iter(loader))
        assert batch == [2 + rank], f"rank {rank} resumed at {batch}, expected [{2 + rank}]"

        # Older TP checkpoints kept the non-reading rank's unread position.
        legacy = dict(saved, train_dataloader=saved["train_dataloader"] if rank == 0 else unread_state)
        try:
            _layout_callback(_loader(dp_rank=rank, dp_world_size=2), tp_world_size=1)._apply_extra_state(  # pylint: disable=protected-access
                legacy
            )
        except ValueError as error:
            assert "dataloader positions differ" in str(error), str(error)
        else:
            raise AssertionError(f"rank {rank} restored inconsistent dataloader positions without TP")
        # TP2 still resumes them because only TP rank 0 reads input.
        _layout_callback(_loader(dp_rank=0, dp_world_size=1), tp_world_size=2)._apply_extra_state(  # pylint: disable=protected-access
            legacy
        )
    except Exception:  # pylint: disable=broad-exception-caught
        errors.put(f"rank {rank}:\n{traceback.format_exc()}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestCheckpointerCallback(unittest.TestCase):
    """Tests for checkpoint restore policy and trainer state handling."""

    @patch("hyper_parallel.trainer.callbacks.checkpoint_callback.empty_cache")
    @patch("hyper_parallel.trainer.callbacks.checkpoint_callback.validate_model_init_dtype")
    @patch("hyper_parallel.trainer.callbacks.checkpoint_callback.build_checkpointer")
    def test_weights_only_restore_initializes_fresh_start_position(
            self,
            mock_build_checkpointer: MagicMock,
            mock_validate_model_init_dtype: MagicMock,
            mock_empty_cache: MagicMock,
    ) -> None:
        """Weights-only restore should retain fresh progress and define its start position."""
        with tempfile.TemporaryDirectory() as restore_path:
            checkpointer = MagicMock()
            mock_build_checkpointer.return_value = checkpointer
            trainer = SimpleNamespace(
                config=SimpleNamespace(
                    checkpoint=_checkpoint_config(restore_path),
                    model_init_dtype="float32",
                ),
                mesh=MagicMock(),
                model=torch.nn.Linear(4, 4),
                optimizer=None,
                lr_scheduler=None,
                train_dataloader=[object()] * 4,
                train_steps=4,
                state=TrainerState(),
            )
            callback = CheckpointerCallback(trainer)

            callback._load_checkpoint()

        self.assertEqual(trainer.state.global_step, 0)
        self.assertEqual(trainer.state.epoch, 0)
        self.assertEqual(trainer.start_epoch, 0)
        self.assertEqual(trainer.start_step, 0)
        checkpointer.load.assert_called_once()
        mock_validate_model_init_dtype.assert_called_once_with(
            trainer.model,
            "float32",
        )
        mock_empty_cache.assert_called_once_with()

    @patch("hyper_parallel.trainer.callbacks.checkpoint_callback.set_device_rng_state")
    def test_partial_epoch_restore_uses_step_derived_position(
            self,
            mock_set_device_rng_state: MagicMock,
    ) -> None:
        """Resume a mid-epoch checkpoint even if its epoch field was advanced."""
        callback = CheckpointerCallback.__new__(CheckpointerCallback)
        callback.trainer = SimpleNamespace(
            state=TrainerState(),
            train_dataloader=[object()] * 4,
            train_steps=4,
            lr_scheduler=None,
        )

        callback._apply_extra_state({
            "global_step": 2,
            "epoch": 1,
            "lr_scheduler": None,
            "train_dataloader": {},
            "rng_state": {},
        })

        self.assertEqual(callback.trainer.state.global_step, 2)
        self.assertEqual(callback.trainer.state.epoch, 0)
        self.assertEqual(callback.trainer.start_epoch, 0)
        self.assertEqual(callback.trainer.start_step, 2)
        mock_set_device_rng_state.assert_called_once_with(None)

    @patch("hyper_parallel.trainer.callbacks.checkpoint_callback.set_device_rng_state")
    def test_train_state_restore_can_replay_from_configured_data_start(
            self,
            mock_set_device_rng_state: MagicMock,
    ) -> None:
        """Warm-start restore keeps progress and RNG without loading the cursor."""
        dataloader = MagicMock()
        callback = CheckpointerCallback.__new__(CheckpointerCallback)
        callback._restore_dataloader_state = False
        callback.trainer = SimpleNamespace(
            state=TrainerState(),
            train_dataloader=dataloader,
            train_steps=4,
            lr_scheduler=None,
        )

        callback._apply_extra_state({
            "global_step": 1,
            "epoch": 0,
            "lr_scheduler": None,
            "train_dataloader": {"cursor": 7},
            "rng_state": {},
        })

        self.assertEqual(callback.trainer.state.global_step, 1)
        dataloader.load_state_dict.assert_not_called()
        mock_set_device_rng_state.assert_called_once_with(None)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_tp_checkpoint_resumes_reader_position_in_other_layouts(self) -> None:
        """Feature: Dataloader resume across TP layouts.

        Description: Two gloo ranks save from TP2, where only TP rank 0 reads and consumes one optimizer batch,
            then restore the per-rank states without TP and with TP. An older state whose non-reading TP rank kept
            its unread position is restored as well.
        Expectation: Both TP ranks persist the reader's position, so the resume without TP continues with the next
            samples; the older state is rejected without TP and still resumes TP2.
        """
        context = mp.get_context("spawn")
        errors = context.Queue()
        port = allocate_port()
        workers = [context.Process(target=_tp_resume_worker, args=(rank, port, errors)) for rank in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(120)
        failures = []
        while not errors.empty():
            failures.append(errors.get())
        for rank, worker in enumerate(workers):
            if worker.is_alive():
                worker.terminate()
                failures.append(f"rank {rank} timed out")
            elif worker.exitcode != 0 and not failures:
                failures.append(f"rank {rank} exited with code {worker.exitcode}")
        self.assertFalse(failures, "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
