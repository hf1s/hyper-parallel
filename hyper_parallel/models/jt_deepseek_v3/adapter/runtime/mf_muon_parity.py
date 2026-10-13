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
"""MindFormers-parity Newton-Schulz partitioning for the JT DeepSeek-V3 Muon optimizer.

HP's JT recipe orthogonalizes every fused Muon matrix as one matrix. MindFormers
splits four parameter classes into blocks first (``make_model_muon_fns`` in
``parallel_core/training_graph/base_models/gpt/gpt_model.py``) and runs
Newton-Schulz on every block separately (``_apply_muon_update`` in
``core/optim/muon.py``):

    HP parameter                      MindFormers Newton-Schulz blocks
    *.self_attn.linear_qkv.weight     q_a rows | kv_a latent rows | kv_a rope rows
    *.self_attn.q_b_proj.weight       per head: nope rows | rope rows
    *.self_attn.kv_b_proj.weight      per head: key nope rows | value rows
    *.mlp.experts.gate_up_proj        per expert: gate rows | up rows

MindFormers also scales every block of one parameter by ``sqrt(max(dims))`` of the
parameter's last block, because ``dim_a, dim_b`` keep the values of the final loop
iteration in ``_apply_muon_update``. Only ``q_b_proj`` has blocks of different
sizes, so its nope block receives ``sqrt(rope_rows / nope_rows)`` of HP's per-block
scale. ``linear_qkv`` fuses two MindFormers parameters (``linear_q_down_proj`` and
``linear_kv_down_proj``), so the scale rule applies to each of them separately.

Still different from MindFormers: Newton-Schulz kernel arithmetic, and HP rounding
``scale * NS`` to bf16 before the FP32 parameter update.

Select it in the training YAML with
``optimizer._target_: hyper_parallel.models.jt_deepseek_v3.adapter.runtime.mf_muon_parity.build_optimizer``.
"""

# This adapter uses the Torch runtime, like the existing model and Trainer modules.
# pylint: disable=forbidden-backend-import

import math
import os
from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import torch

_LINEAR_QKV = "self_attn.linear_qkv.weight"
_Q_B_PROJ = "self_attn.q_b_proj.weight"
_KV_B_PROJ = "self_attn.kv_b_proj.weight"
_GATE_UP_PROJ = "mlp.experts.gate_up_proj"


class _Block(NamedTuple):
    """One MindFormers Newton-Schulz block inside an HP parameter.

    ``select`` returns the block as a view of the parameter-shaped tensor. Per-head
    blocks come back as ``[heads, rows, cols]`` and ``flatten`` turns them into the
    ``[heads * rows, cols]`` matrix MindFormers orthogonalizes.
    """

    select: Callable[[torch.Tensor], torch.Tensor]
    flatten: bool


def _rows(start: int, stop: int) -> _Block:
    """Rows ``[start, stop)`` of the matrix dimension; expert batches stay 3-D."""
    return _Block(lambda tensor: tensor[..., start:stop, :], flatten=False)


def _head_rows(period: int, start: int, stop: int) -> _Block:
    """Rows ``[start, stop)`` of every ``period``-row head of a 2-D projection."""

    def select(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(tensor.shape[0] // period, period, tensor.shape[1])[:, start:stop, :]

    return _Block(select, flatten=True)


def _hp_logical_dims(shape: Tuple[int, ...]) -> Tuple[int, int]:
    """Matrix dims HP's ``compute_muon_slice_scale`` uses for one Newton-Schulz block."""
    if len(shape) == 3 and shape[1] == 1:
        return shape[0], shape[2]
    return shape[-2], shape[-1]


def _require(condition: bool, param_fqn: str, shape: Tuple[int, ...], expected: str) -> None:
    """Reject a layout that would silently split the wrong rows."""
    if not condition:
        raise ValueError(f"mf_muon_parity: {param_fqn} has shape {shape}, expected {expected}")


def _block_groups(param_fqn: str, shape: Tuple[int, ...], config: Any) -> Optional[List[List[_Block]]]:
    """Return MindFormers' Newton-Schulz blocks, grouped per MindFormers parameter.

    Each inner list shares one Muon scale: ``sqrt(max(dims))`` of its last block.
    ``None`` keeps HP's whole-matrix Newton-Schulz, which matches MindFormers for
    every other Muon parameter.
    """
    if param_fqn.endswith(_LINEAR_QKV):
        q_rows, kv_rows, rope_rows = config.q_lora_rank, config.kv_lora_rank, config.qk_rope_head_dim
        _require(len(shape) == 2 and shape[0] == q_rows + kv_rows + rope_rows, param_fqn, shape,
                 f"[{q_rows} + {kv_rows} + {rope_rows}, hidden]")
        return [[_rows(0, q_rows)],
                [_rows(q_rows, q_rows + kv_rows), _rows(q_rows + kv_rows, q_rows + kv_rows + rope_rows)]]
    if param_fqn.endswith(_Q_B_PROJ):
        nope, rope = config.qk_nope_head_dim, config.qk_rope_head_dim
        period = nope + rope
        _require(len(shape) == 2 and shape[0] % period == 0, param_fqn, shape,
                 f"[heads * {period}, q_lora_rank]")
        return [[_head_rows(period, 0, nope), _head_rows(period, nope, period)]]
    if param_fqn.endswith(_KV_B_PROJ):
        nope, value = config.qk_nope_head_dim, config.v_head_dim
        period = nope + value
        _require(len(shape) == 2 and shape[0] % period == 0, param_fqn, shape,
                 f"[heads * {period}, kv_lora_rank]")
        return [[_head_rows(period, 0, nope), _head_rows(period, nope, period)]]
    if param_fqn.endswith(_GATE_UP_PROJ):
        intermediate = config.moe_intermediate_size
        _require(len(shape) == 3 and shape[1] == 2 * intermediate, param_fqn, shape,
                 f"[experts, 2 * {intermediate}, hidden]")
        return [[_rows(0, intermediate), _rows(intermediate, 2 * intermediate)]]
    return None


def _ns_tensors_and_factors(groups: List[List[_Block]], tensor: torch.Tensor, matched_adamw_rms: float,
                            last_block_scale: bool) -> Tuple[List[torch.Tensor], List[float]]:
    """Build the Newton-Schulz inputs and the factors that turn HP's scales into MindFormers'.

    HP multiplies each block by ``sqrt(max(own dims)) * rms``. MindFormers multiplies
    every block of a parameter by ``sqrt(max(last block dims)) * rms``.
    """
    tensors, factors = [], []
    for group in groups:
        group_tensors = []
        for block in group:
            selected = block.select(tensor)
            group_tensors.append(selected.reshape(-1, selected.shape[-1]) if block.flatten else selected)
        last = max(_hp_logical_dims(tuple(group_tensors[-1].shape)))
        for group_tensor in group_tensors:
            own = max(_hp_logical_dims(tuple(group_tensor.shape)))
            factors.append(math.sqrt(last / own) if last_block_scale and matched_adamw_rms else 1.0)
        tensors.extend(group_tensors)
    return tensors, factors


def make_mf_ns_transform_fn(config: Any, *, matched_adamw_rms: float, last_block_scale: bool = True,
                            transform_cls: Any = None) -> Callable[[str, torch.Tensor], Any]:
    """Build an HP ``ns_transform_fn`` reproducing MindFormers' Newton-Schulz blocks.

    Args:
        config: JT model config providing the MLA and MoE dimensions.
        matched_adamw_rms: Muon ``matched_adamw_rms``; zero disables the scale correction.
        last_block_scale: Reproduce MindFormers' last-block Muon scale.
        transform_cls: ``NSInputTransform`` class; defaults to HP's.

    Returns:
        ``ns_transform_fn(param_fqn, working_input)`` for ``hyper_parallel`` Muon.
    """
    if transform_cls is None:
        from hyper_parallel.core.optimizer.muon import NSInputTransform as transform_cls  # pylint: disable=C0415

    def ns_transform_fn(param_fqn: str, working_input: torch.Tensor) -> Any:
        groups = _block_groups(param_fqn, tuple(working_input.shape), config)
        if groups is None:
            return None
        blocks = [block for group in groups for block in group]
        tensors, factors = _ns_tensors_and_factors(groups, working_input, matched_adamw_rms, last_block_scale)

        def restore(updates: List[torch.Tensor], output: torch.Tensor) -> None:
            """Write each scaled block update back into its rows of the parameter update."""
            for block, update, factor in zip(blocks, updates, factors):
                target = block.select(output)
                if factor != 1.0:
                    update = update.float() * factor
                target.copy_(update.reshape(target.shape))

        return transform_cls(tensors=tensors, restore=restore)

    return ns_transform_fn


def _report_partition(model: torch.nn.Module, matched_adamw_rms: float, last_block_scale: bool) -> None:
    """Print, on rank 0, every parameter whose Newton-Schulz partition now follows MindFormers."""
    if os.environ.get("RANK", "0") != "0":
        return
    lines = []
    for name, parameter in model.named_parameters():
        groups = _block_groups(name, tuple(parameter.shape), model.config)
        if groups is None:
            continue
        tensors, factors = _ns_tensors_and_factors(
            groups, torch.empty(tuple(parameter.shape), device="meta"), matched_adamw_rms, last_block_scale)
        blocks = ", ".join(f"{tuple(tensor.shape)}x{factor:.4f}" for tensor, factor in zip(tensors, factors))
        lines.append(f"  {name}: {blocks}")
    print(f"[mf_muon_parity] MindFormers Newton-Schulz blocks (last-block scale: {last_block_scale}) "
          f"for {len(lines)} parameters; block shape x scale factor:\n" + "\n".join(lines), flush=True)


def build_optimizer(*, model: torch.nn.Module, qk_clip_threshold: float, mf_last_block_scale: bool = True,
                    **kwargs: Any) -> Any:
    """Build the JT Muon/AdamW optimizer with MindFormers' Newton-Schulz partitioning.

    Drop-in for ``hyper_parallel.models.jt_deepseek_v3.adapter.runtime.jt_optimizer.build_optimizer``.

    Args:
        model: JT model whose final FSDP parameter layouts are prepared.
        qk_clip_threshold: Positive clipping threshold for QK projections.
        mf_last_block_scale: Reproduce MindFormers' last-block Muon scale; ``False``
            keeps only the block partition.
        **kwargs: Remaining JT optimizer arguments, including ``muon_config``.

    Returns:
        The Muon builder returned by the JT optimizer factory.
    """
    from hyper_parallel.models.jt_deepseek_v3.adapter.runtime.jt_optimizer import (  # pylint: disable=C0415
        build_optimizer as build_jt_optimizer,
    )

    muon_config = dict(kwargs.pop("muon_config"))
    for key in ("ns_transform_fn", "reshape_fn", "muon_ns_transform_fn", "muon_reshape_fn"):
        if muon_config.get(key) is not None:
            raise ValueError(f"mf_muon_parity owns the Newton-Schulz partition; remove muon_config.{key}")
    matched_adamw_rms = muon_config.get("matched_adamw_rms", muon_config.get("muon_matched_adamw_rms", 0.2))
    muon_config["ns_transform_fn"] = make_mf_ns_transform_fn(
        model.config, matched_adamw_rms=matched_adamw_rms, last_block_scale=mf_last_block_scale)
    _report_partition(model, matched_adamw_rms, mf_last_block_scale)
    return build_jt_optimizer(model=model, qk_clip_threshold=qk_clip_threshold, muon_config=muon_config, **kwargs)
