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
"""Explicit limits of the fixed single-card, sequential continuation adapter."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def validate_resume_state(state: Mapping[str, Any], step: int) -> None:
    """Reject metadata that cannot describe the supported continuation.

    Args:
        state: Converted resume metadata.
        step: Requested number of completed optimizer updates.

    Raises:
        ValueError: If counters, batch size, ordering, or scaling are unsupported.
    """
    if (not isinstance(step, int) or isinstance(step, bool)) or step < 0:
        raise ValueError("Requested step must be a nonnegative integer")
    for name, expected in (("global_step", step), ("global_batch_size", 1), ("consumed_samples", step)):
        value = state.get(name)
        if (not isinstance(value, int) or isinstance(value, bool)) or value != expected:
            raise ValueError(f"Resume state {name} must equal integer {expected}; got {value!r}")
    if "step_num" in state:
        value = state["step_num"]
        if not isinstance(value, int) or isinstance(value, bool) or value != step:
            raise ValueError("step_num and global_step must agree")
    if state.get("data_order") != "sequential":
        raise ValueError("This adapter restores only a sequential logical data cursor")
    if type(state.get("loss_scale")) not in (int, float) or state["loss_scale"] != 1.0:
        raise ValueError("This adapter does not restore dynamic or non-unit loss scaling")


def validate_materialization_config(config: Mapping[str, Any], state_dir: Path) -> None:
    """Validate recipe constraints before constructing a Trainer.

    Args:
        config: Serialized resolved Trainer configuration.
        state_dir: Exact directory whose model and optimizer arrays will be used.

    Raises:
        ValueError: If the recipe exceeds the fixed adapter's restore contract.
    """
    training = config.get("training") or {}
    if training.get("global_batch_size") != 1 or training.get("micro_batch_size", 1) != 1:
        raise ValueError("This adapter requires global_batch_size=micro_batch_size=1")
    accelerator = config.get("accelerator") or {}
    if any(accelerator.get(key, 1) != 1 for key in ("tp_size", "ep_size", "cp_size", "pp_size")):
        raise ValueError("This adapter supports only one device and no model-parallel sharding")
    dataloader = config.get("dataloader") or {}
    if dataloader.get("_target_") != "hyper_parallel.data.batching.FixedBatchDataLoader":
        raise ValueError("This adapter requires the native FixedBatchDataLoader")
    if dataloader.get("num_workers", 0) != 0:
        raise ValueError("This adapter does not reconstruct prefetched worker state")
    if dataloader.get("sampler_type", "single") != "single":
        raise ValueError("This adapter requires the sequential single sampler")
    if dataloader.get("data_rearrange_map") is not None or dataloader.get("index_mapping") is not None:
        raise ValueError("This adapter does not reconstruct a remapped sample order")
    scheduler = config.get("lr_scheduler") or {}
    target = "hyper_parallel.components.optim.lr_scheduler.MultiLRScheduler"
    if scheduler.get("_target_") != target or scheduler.get("lr_decay_style") != "constant":
        raise ValueError("This adapter requires the native constant MultiLRScheduler")
    lr_config = scheduler.get("lr_config") or {}
    if lr_config.get("lr_warmup_steps", 0) not in (None, 0) or lr_config.get("lr_warmup_ratio", 0) != 0:
        raise ValueError("This adapter does not reconstruct a warmup schedule")
    reference = (config.get("model") or {}).get("reference_weights")
    if not reference or Path(reference).resolve() != state_dir.resolve():
        raise ValueError("model.reference_weights must point to the selected offline state directory")
    checkpoint = config.get("checkpoint") or {}
    if not checkpoint.get("save_optimizer") or not checkpoint.get("save_train_state"):
        raise ValueError("Native materialization must save optimizer and training state")
