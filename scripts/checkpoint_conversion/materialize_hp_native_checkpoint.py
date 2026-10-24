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
"""Materialize a native HP DCP checkpoint from MF-converted offline state.

This is a standalone adapter. It does not modify HyperParallel source code.
It builds the configured HP model/optimizer/dataloader, injects the converted
MF optimizer states and cursor, then uses the existing HP CheckpointerCallback
to write a native ``global_step_<N>`` checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from hyper_parallel.components.checkpoint.dcp_checkpointer import STEP_PREFIX
from hyper_parallel.trainer.callbacks.checkpoint_callback import CheckpointerCallback
from hyper_parallel.trainer.config.parser import parse_training_args
if __package__:
    from .manifest_input import load_conversion_manifest, map_manifest_slots
    from .optimizer_state import restore_optimizer_state
    from .resume_state import validate_materialization_config, validate_resume_state
else:
    from manifest_input import load_conversion_manifest, map_manifest_slots
    from optimizer_state import restore_optimizer_state
    from resume_state import validate_materialization_config, validate_resume_state


def set_nested_cursor(value: Any, consumed_samples: int, global_batch_size: int) -> bool:
    """Set every sampler cursor in a nested dataloader state tree.

    Args:
        value: Dataloader state tree to update in place.
        consumed_samples: Number of samples already consumed.
        global_batch_size: Batch size associated with the cursor.

    Returns:
        Whether at least one cursor was updated.
    """

    changed = False
    if isinstance(value, dict):
        if "consumed_samples" in value:
            value["consumed_samples"] = consumed_samples
            value["global_batch_size"] = global_batch_size
            value["epoch"] = 0
            changed = True
        for child in value.values():
            changed = set_nested_cursor(child, consumed_samples, global_batch_size) or changed
    elif isinstance(value, list):
        for child in value:
            changed = set_nested_cursor(child, consumed_samples, global_batch_size) or changed
    return changed


def load_optimizer_npz(
        trainer: Any, state_dir: Path, step: int, *, optimizer_manifest: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Inject converted Adam/Muon states into HP optimizer objects.

    Args:
        trainer: Constructed HP text trainer.
        state_dir: Directory containing ``optimizer.npz``.
        step: Global step represented by the source checkpoint.
        optimizer_manifest: Verified manifest for an existing named-slot NPZ input.

    Returns:
        Number of assigned states and the destination-derived required state count.
    """

    with np.load(state_dir / "model.npz", allow_pickle=False) as model_archive:
        with np.load(state_dir / "optimizer.npz", allow_pickle=False) as optimizer_archive:
            arrays = (map_manifest_slots(optimizer_archive, optimizer_manifest)
                      if optimizer_manifest is not None else optimizer_archive)
            restored = restore_optimizer_state(
                trainer.base.model, trainer.base.optimizer, model_archive, arrays, step,
            )
    return restored, restored



def main() -> None:
    """Materialize one native HP DCP checkpoint from offline state."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--optimizer-key-format", choices=("family-prefix", "manifest"), default="family-prefix")
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=1)
    args = parser.parse_args()

    if args.optimizer_key_format == "manifest" and args.resume_state is None:
        raise ValueError(
            "Manifest input requires explicit --resume-state; batch/data/scaling metadata cannot be inferred"
        )
    if args.optimizer_key_format != "manifest" and args.resume_state is not None:
        raise ValueError("--resume-state applies only to manifest input; family-prefix input uses its own state.json")
    metadata_path = args.resume_state or args.state_dir / "state.json"
    state = json.loads(metadata_path.read_text(encoding="utf8"))
    validate_resume_state(state, args.step)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This offline adapter must run with WORLD_SIZE=1")
    if any(size != 1 for size in (args.tp_size, args.ep_size, args.cp_size, args.pp_size)):
        raise ValueError("This offline adapter supports only a single device")
    checkpoint_path = args.output / f"{STEP_PREFIX}{args.step}"
    if checkpoint_path.exists():
        raise ValueError(f"Refusing to replace an existing checkpoint: {checkpoint_path}")
    optimizer_manifest = (load_conversion_manifest(args.state_dir, args.step)
                          if args.optimizer_key_format == "manifest" else None)

    overrides = [
        "--checkpoint.restore_from=null",
        "--checkpoint.save_ckpt=true",
        "--checkpoint.save_optimizer=true",
        "--checkpoint.save_train_state=true",
        f"--checkpoint.checkpoint_dir={args.output}",
        f"--training.train_iters={max(1, args.step * 2)}",
        f"--accelerator.tp_size={args.tp_size}",
        f"--accelerator.ep_size={args.ep_size}",
        f"--accelerator.cp_size={args.cp_size}",
        f"--accelerator.pp_size={args.pp_size}",
        f"--accelerator.sequence_parallel={'true' if args.tp_size > 1 else 'false'}",
    ]
    config = parse_training_args([str(args.config), *overrides])
    validate_materialization_config(config.to_dict(), args.state_dir)
    # The CLI needs the optional dataset/training stack; state validation does not.
    from hyper_parallel.trainer.text_trainer import TextTrainer  # pylint: disable=import-outside-toplevel

    trainer = TextTrainer(config)
    try:
        assigned, expected = load_optimizer_npz(trainer, args.state_dir, args.step,
                                               optimizer_manifest=optimizer_manifest)

        # Constant-LR continuation has no numerical scheduler value to recover,
        # but advance its state counter so the native checkpoint records step N.
        for _ in range(args.step):
            trainer.base.lr_scheduler.step()

        dataloader_state = trainer.base.train_dataloader.state_dict()
        if not set_nested_cursor(dataloader_state, state["consumed_samples"], 1):
            raise ValueError("could not locate consumed_samples in HP dataloader state")
        trainer.base.train_dataloader.load_state_dict(dataloader_state)

        trainer.base.state.global_step = args.step
        trainer.base.state.epoch = 0
        # Trainer has no standalone public materialization method; reuse its
        # checkpoint callback to preserve the native payload and save protocol.
        callback = next(
            callback for callback in trainer.base._callbacks
            if isinstance(callback, CheckpointerCallback)
        )
        callback._save_checkpoint(trainer.base.state, force_sync=True)
        if not (checkpoint_path / ".metadata").is_file():
            raise RuntimeError(f"Native checkpoint did not complete: {checkpoint_path}")
        print(f"native HP checkpoint written: {checkpoint_path}")
        print(f"optimizer states assigned: {assigned}/{expected}")
    finally:
        trainer.base.destroy_distributed()


if __name__ == "__main__":
    main()
