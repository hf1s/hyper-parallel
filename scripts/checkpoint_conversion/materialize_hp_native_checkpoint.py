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
to write a native ``step-<N>`` checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hyper_parallel.components.checkpoint.dcp_checkpointer import initialize_optimizer_state
from hyper_parallel.trainer.callbacks.checkpoint_callback import CheckpointerCallback
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.text_trainer import TextTrainer


def leaf_optimizers(optimizer: Any) -> list[Any]:
    """Return leaf optimizer instances under HP wrappers.

    Args:
        optimizer: Top-level Hyper optimizer or optimizer wrapper.

    Returns:
        Leaf optimizer instances that own parameter state.
    """

    if hasattr(optimizer, "optimizers_dict"):
        return list(optimizer.optimizers_dict.values())
    if hasattr(optimizer, "chained_optimizers"):
        return list(optimizer.chained_optimizers)
    if hasattr(optimizer, "optimizer"):
        return leaf_optimizers(optimizer.optimizer)
    return [optimizer]


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


def load_optimizer_npz(trainer: Any, state_dir: Path, step: int) -> tuple[int, int]:
    """Inject converted Adam/Muon states into HP optimizer objects.

    Args:
        trainer: Constructed HP text trainer.
        state_dir: Directory containing ``optimizer.npz``.
        step: Global step represented by the source checkpoint.

    Returns:
        Number of assigned states and number of non-placeholder states found.
    """

    archive = np.load(state_dir / "optimizer.npz", allow_pickle=False)
    model_names = {id(param): name for name, param in trainer.base.model.named_parameters()}
    optimizers = leaf_optimizers(trainer.base.optimizer)
    assigned = 0
    expected = 0

    for leaf in optimizers:
        class_name = type(leaf).__name__.lower()
        is_muon = "muon" in class_name
        allowed = (
            (("muon_m", "momentum_buffer"),)
            if is_muon
            else (("adam_m", "exp_avg"), ("adam_v", "exp_avg_sq"))
        )
        for group in leaf.param_groups:
            group["step"] = step
            for param in group["params"]:
                name = model_names.get(id(param))
                if name is None:
                    raise KeyError("HP optimizer parameter is missing from model.named_parameters()")
                state = leaf.state[param]
                for prefix, field in allowed:
                    key = f"{prefix}.{name}"
                    if key not in archive.files:
                        continue
                    value = np.asarray(archive[key])
                    # MF writes one/tiny placeholders for the other optimizer family.
                    if value.size <= 1:
                        continue
                    expected += 1
                    if tuple(value.shape) != tuple(param.shape):
                        raise ValueError(
                            f"optimizer shape mismatch for {key}: "
                            f"MF={value.shape}, HP={tuple(param.shape)}"
                        )
                    state[field] = torch.as_tensor(
                        value, device=param.device, dtype=torch.float32
                    )
                    assigned += 1
                state["step"] = torch.tensor(step, device=param.device, dtype=torch.float32)

    if assigned != expected:
        raise RuntimeError(f"optimizer state assignment mismatch: assigned={assigned}, expected={expected}")
    return assigned, expected


def main() -> None:
    """Materialize one native HP DCP checkpoint from offline state."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=1)
    args = parser.parse_args()

    state = json.loads((args.state_dir / "state.json").read_text())
    if int(state["global_step"]) != args.step:
        raise ValueError(f"state step {state['global_step']} != requested step {args.step}")
    if int(state["global_batch_size"]) != 1:
        raise ValueError("this adapter currently requires global_batch_size=1")

    overrides = [
        "--checkpoint.restore_from=null",
        "--checkpoint.save_ckpt=true",
        f"--checkpoint.checkpoint_dir={args.output}",
        f"--training.train_iters={args.step * 2}",
        f"--accelerator.tp_size={args.tp_size}",
        f"--accelerator.ep_size={args.ep_size}",
        f"--accelerator.cp_size={args.cp_size}",
        f"--accelerator.pp_size={args.pp_size}",
        f"--accelerator.sequence_parallel={'true' if args.tp_size > 1 else 'false'}",
    ]
    config = parse_training_args([str(args.config), *overrides])
    trainer = TextTrainer(config)
    try:
        if not initialize_optimizer_state(trainer.base.optimizer):
            raise RuntimeError("could not materialize HP optimizer state")
        assigned, expected = load_optimizer_npz(trainer, args.state_dir, args.step)

        # Constant-LR continuation has no numerical scheduler value to recover,
        # but advance its state counter so the native checkpoint records step N.
        for _ in range(args.step):
            trainer.base.lr_scheduler.step()

        dataloader_state = trainer.base.train_dataloader.state_dict()
        if not set_nested_cursor(dataloader_state, args.step, 1):
            raise ValueError("could not locate consumed_samples in HP dataloader state")
        trainer.base.train_dataloader.load_state_dict(dataloader_state)

        trainer.base.state.global_step = args.step
        trainer.base.state.epoch = 0
        callback = next(
            callback for callback in trainer.base._callbacks
            if isinstance(callback, CheckpointerCallback)
        )
        callback._save_checkpoint(trainer.base.state, force_sync=True)
        print(f"native HP checkpoint written: {args.output}/step-{args.step}")
        print(f"optimizer states assigned: {assigned}/{expected}")
    finally:
        trainer.base.destroy_distributed()


if __name__ == "__main__":
    main()
