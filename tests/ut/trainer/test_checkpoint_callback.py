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

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.trainer.callbacks.checkpoint_callback import CheckpointerCallback
from hyper_parallel.trainer.callbacks.environ_meter_callback import EnvironMeterCallback
from hyper_parallel.trainer.state import TrainerState


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


class TestCheckpointerCallback(unittest.TestCase):
    """Tests for checkpoint restore policy and trainer state handling."""

    @staticmethod
    def _meter_trainer(directory: str, per_rank: bool) -> SimpleNamespace:
        """Build a CPU trainer with the production checkpoint and meter callbacks."""
        config = _checkpoint_config(directory)
        config.save_ckpt = True
        config.save_steps = 1
        config.save_train_state = True
        config.save_extra_state_per_rank = per_rank
        config.restore_train_state = True
        trainer = SimpleNamespace(
            config=SimpleNamespace(checkpoint=config, model_init_dtype="float32"),
            mesh=SimpleNamespace(dp_cp_mesh=None, cp_size=1),
            model=torch.nn.Linear(4, 4),
            optimizer=None,
            lr_scheduler=None,
            train_dataloader=[object()] * 10,
            train_steps=10,
            state=TrainerState(),
        )
        trainer.environ_meter_callback = EnvironMeterCallback(trainer)
        return trainer

    @staticmethod
    def _meter_step(trainer: SimpleNamespace, tokens: int) -> None:
        """Consume one batch through the real meter lifecycle."""
        meter = trainer.environ_meter_callback
        meter.on_step_begin(trainer.state)
        meter.on_micro_step_begin(trainer.state, {
            "input_ids": torch.zeros(1, tokens, dtype=torch.long), "token_count": torch.tensor(tokens),
        })
        trainer.state.global_step += 1
        meter.on_step_end(trainer.state, loss=1.0, loss_dict={}, grad_norm=0.5)

    @patch("hyper_parallel.trainer.callbacks.environ_meter_callback.get_device_type", return_value="cpu")
    @patch("hyper_parallel.trainer.callbacks.checkpoint_callback.get_device_rng_state", return_value=None)
    @patch("hyper_parallel.trainer.callbacks.checkpoint_callback.set_device_rng_state")
    @patch("hyper_parallel.trainer.callbacks.checkpoint_callback.empty_cache")
    @patch("hyper_parallel.components.checkpoint.dcp_checkpointer._empty_cache")
    @patch("hyper_parallel.core.distributed_checkpoint.api.dist.get_rank", return_value=0)
    @patch("hyper_parallel.core.distributed_checkpoint.api.dist.get_world_size", return_value=1)
    @patch("hyper_parallel.core.distributed_checkpoint.api.dist.barrier")
    @patch.dict("hyper_parallel.core.distributed_checkpoint.standard_planner.StandardSavePlanner.cached_save_result",
                {}, clear=True)
    def test_cumulative_metrics_resume_from_both_checkpoint_layouts(self, *_mocks: MagicMock) -> None:
        """Real CPU checkpoint round trips preserve counters and weights-only starts fresh."""
        for per_rank in (False, True):
            with self.subTest(per_rank=per_rank), tempfile.TemporaryDirectory() as directory:
                source = self._meter_trainer(directory, per_rank)
                self._meter_step(source, 3)
                self._meter_step(source, 5)
                writer = CheckpointerCallback(source)
                writer.on_step_end(source.state)
                for restore_train_state in (True, False):
                    with self.subTest(restore_train_state=restore_train_state):
                        restored = self._meter_trainer(directory, per_rank)
                        restored.config.checkpoint.restore_from = str(Path(directory) / "global_step_2")
                        restored.config.checkpoint.restore_train_state = restore_train_state
                        reader = CheckpointerCallback(restored)
                        reader.on_train_begin(restored.state)
                        self._meter_step(restored, 7)
                        metrics = restored.step_env_metrics
                        expected_tokens, expected_samples = (15.0, 3.0) if restore_train_state else (7.0, 1.0)
                        self.assertEqual(metrics["data/consumed_tokens"], expected_tokens)
                        self.assertEqual(metrics["data/consumed_samples"], expected_samples)
                        self.assertEqual(metrics["data/step_tokens"], 7.0)
                        self.assertEqual(metrics["data/step_samples"], 1.0)

    @patch("hyper_parallel.trainer.callbacks.checkpoint_callback.set_device_rng_state")
    def test_legacy_training_state_without_metrics_keeps_fresh_counters(self, _mock_rng: MagicMock) -> None:
        """Old checkpoints cannot recover historical token counts from the global step."""
        trainer = self._meter_trainer("unused", False)
        callback = CheckpointerCallback.__new__(CheckpointerCallback)
        callback.trainer = trainer
        callback._apply_extra_state({"global_step": 2, "epoch": 0})
        self.assertEqual(trainer.state.global_step, 2)
        self.assertEqual(trainer.environ_meter_callback.state_dict(), {
            "consumed_tokens": 0, "consumed_samples": 0,
        })

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


if __name__ == "__main__":
    unittest.main()
