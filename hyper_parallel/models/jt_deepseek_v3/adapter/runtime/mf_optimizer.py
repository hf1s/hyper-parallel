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
"""JT DeepSeek-V3 Muon aligned with MindFormers: ``MFMuon`` and ``build_mf_optimizer``.

Selected by pointing the recipe's optimizer ``_target_`` at ``build_mf_optimizer``; the
default ``jt_optimizer.build_optimizer`` path is unchanged. Two update details follow
MindFormers r1.8.0 ``core/optim/muon.py::_apply_muon_update``:

1. Newton-Schulz granularity. MindFormers' ``muon_split_fn`` orthogonalizes every logical
   matrix of a fused parameter on its own. ``make_mf_ns_transform_fn`` cuts the fused HP
   parameters into the same blocks, and each block is scaled by its own
   ``sqrt(max(rows, cols))``.
2. Scale precision. ``Muon`` multiplies the BF16 Newton-Schulz output in place, rounding
   the scaled update to BF16. MindFormers multiplies the BF16 output out of place by the
   FP32 scalar ``lr * sqrt(max(rows, cols)) * matched_adamw_rms`` and subtracts that FP32
   product from the FP32 parameter. ``MFMuon`` does the same.

Momentum, the Newton-Schulz iteration, communication and the AdamW leaf stay native.
"""

# This adapter uses the Torch runtime, like the existing model and Trainer modules.
# pylint: disable=forbidden-backend-import

import math
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from hyper_parallel.components.optim.parameter_groups import get_adamw_param_groups, split_muon_adamw_params
from hyper_parallel.core.optimizer import _build_configured_optimizer, _filter_optimizer_config
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.dtensor_compat import detect_dtensor_backend, to_local_if_dtensor
from hyper_parallel.core.optimizer.muon import Muon, NSInputTransform
from hyper_parallel.core.optimizer.muon_shard import chunk_update_by_layout
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.models.jt_deepseek_v3.adapter.runtime.jt_optimizer import _after_update

_LINEAR_QKV = ".self_attn.linear_qkv.weight"
_Q_B_PROJ = ".self_attn.q_b_proj.weight"
_KV_B_PROJ = ".self_attn.kv_b_proj.weight"
_EXPERT_GATE_UP = ".mlp.experts.gate_up_proj"
_DENSE_FC1 = ".mlp.linear_fc1.weight"
_SHARED_FC1 = ".mlp.shared_experts.linear_fc1.weight"

_Block = Tuple[int, int, int]


def _block_rows(tensor: torch.Tensor, block: _Block) -> torch.Tensor:
    """View rows ``[start, stop)`` of every ``period``-row group as ``[..., groups, rows, cols]``."""
    period, start, stop = block
    return tensor.unflatten(-2, (-1, period))[..., start:stop, :]


def _mf_blocks(param_fqn: str, shape: Tuple[int, ...], config: Any) -> Optional[Tuple[_Block, ...]]:
    """Return MindFormers' Newton-Schulz blocks of one HP parameter as ``(period, start, stop)``.

    MindFormers keeps every other Muon parameter whole (``None``). In ``muon_split_fn`` terms:
    ``linear_qkv`` fuses ``linear_q_down_proj`` (whole) and ``linear_kv_down_proj`` (latent | rope
    rows); ``q_b_proj`` and ``kv_b_proj`` are the per-head ``periodic`` splits; the expert
    ``gate_up_proj`` is ``reshape_concat``; a fused SwiGLU ``linear_fc1`` holds the gate and up
    rows that MindFormers' interleaved ``linear_fc1`` separates with ``alt_pair_periodic``.
    A shape that does not fit ``config`` also returns ``None``: the parameter keeps Muon's
    whole-matrix Newton-Schulz instead of being cut at the wrong rows.
    """
    rows = shape[-2] if len(shape) >= 2 else 0
    if param_fqn.endswith(_LINEAR_QKV):
        q_rows, kv_rows = config.q_lora_rank, config.kv_lora_rank
        total = q_rows + kv_rows + config.qk_rope_head_dim
        fits = len(shape) == 2 and rows == total
        blocks = (total, 0, q_rows), (total, q_rows, q_rows + kv_rows), (total, q_rows + kv_rows, total)
    elif param_fqn.endswith((_Q_B_PROJ, _KV_B_PROJ)):
        nope = config.qk_nope_head_dim
        period = nope + (config.qk_rope_head_dim if param_fqn.endswith(_Q_B_PROJ) else config.v_head_dim)
        fits = len(shape) == 2 and rows % period == 0
        blocks = (period, 0, nope), (period, nope, period)
    elif param_fqn.endswith(_EXPERT_GATE_UP):
        intermediate = config.moe_intermediate_size
        fits = len(shape) == 3 and rows == 2 * intermediate
        blocks = (rows, 0, intermediate), (rows, intermediate, rows)
    elif param_fqn.endswith((_DENSE_FC1, _SHARED_FC1)):
        fits = len(shape) == 2 and rows % 2 == 0
        blocks = (rows, 0, rows // 2), (rows, rows // 2, rows)
    else:
        return None
    return blocks if fits else None


def make_mf_ns_transform_fn(config: Any) -> Callable[[str, torch.Tensor], Optional[NSInputTransform]]:
    """Build a Muon ``ns_transform_fn`` that cuts fused parameters into MindFormers' blocks.

    Args:
        config: JT model config with the MLA ranks, head dimensions and ``moe_intermediate_size``.

    Returns:
        ``ns_transform_fn(param_fqn, working_input)``; it returns ``None`` for a parameter that
        MindFormers orthogonalizes as one matrix or whose shape does not fit ``config``.
    """

    def ns_transform_fn(param_fqn: str, working_input: torch.Tensor) -> Optional[NSInputTransform]:
        """Split one parameter-shaped Newton-Schulz input into its MindFormers blocks.

        Args:
            param_fqn: Model parameter name.
            working_input: Contiguous Newton-Schulz input with the parameter's shape.

        Returns:
            The block transform, or ``None`` to keep one matrix.
        """
        blocks = _mf_blocks(param_fqn, tuple(working_input.shape), config)
        if blocks is None:
            return None

        def restore(updates: List[torch.Tensor], output: torch.Tensor) -> None:
            """Write each block update into its rows of ``output``.

            Args:
                updates: Block updates in block order.
                output: Parameter-shaped update; the blocks cover every row.
            """
            for block, update in zip(blocks, updates):
                rows = _block_rows(output, block)
                rows.copy_(update.reshape(rows.shape))

        tensors = [_block_rows(working_input, block).flatten(-3, -2) for block in blocks]
        return NSInputTransform(tensors=tensors, restore=restore)

    return ns_transform_fn


def _mf_multiplier(shape: Tuple[int, ...], group: Dict[str, Any]) -> float:
    """Return the FP32 factor MindFormers applies to one block's Newton-Schulz output.

    ``sqrt(max(dims)) * matched_adamw_rms`` and then ``lr *`` that ratio, each rounded once to
    FP32 as in ``_apply_muon_update``. With ``apply_lr_in_update`` the negated factor is the
    whole parameter step, otherwise ``lr`` is applied by the caller.
    """
    rms = group["matched_adamw_rms"]
    if rms:
        ratio = np.sqrt(np.float32(max(Muon._shape_to_core_shape(shape)))) * np.float32(rms)
    else:
        ratio = np.float32(1.0 if group["zero_rms_scale_mode"] == "use_lr" else 0.0)
    if group["apply_lr_in_update"]:
        ratio = -(np.float32(group["lr"]) * ratio)
    return float(ratio)


class MFMuon(Muon):
    """Muon whose scaled update stays FP32, as in MindFormers.

    The update of every Newton-Schulz block is ``-(lr * sqrt(max(dims)) * rms) * output`` with
    an FP32 factor and an FP32 result, applied to the parameter with ``alpha=1``. ``lr`` is
    therefore always folded into the update: ``apply_lr_in_update`` is True whatever the
    configuration says, because that is how this class applies lr.
    """

    def __init__(self, params: Any, **kwargs: Any) -> None:
        """Build native Muon state with lr folded into the update.

        Args:
            params: Parameters or parameter groups optimized by Muon.
            **kwargs: ``Muon`` options; ``ns_transform_fn`` may split parameters into blocks.
                A configured ``apply_lr_in_update`` is replaced by True.

        Raises:
            ValueError: If ``reshape_fn`` is given.
        """
        if kwargs.get("reshape_fn") is not None:
            raise ValueError("MFMuon does not accept reshape_fn; it splits Newton-Schulz inputs with ns_transform_fn")
        kwargs["apply_lr_in_update"] = True
        super().__init__(params, **kwargs)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore the native state, keeping lr folded into the update.

        ``apply_lr_in_update`` records where lr is applied, not optimizer state. A checkpoint
        written by native Muon carries ``False``; momentum, step and lr load unchanged.

        Args:
            state_dict: Optimizer state produced by ``state_dict()``.
        """
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            group["apply_lr_in_update"] = True

    def _compute_batched_ns_updates(self, p_list, ns_inputs, group, no_shard=False):
        """Send every batch to the FP32 path; ``Muon``'s native fast path scales in BF16."""
        return self._compute_batched_ns_updates_with_transform(p_list, ns_inputs, group, no_shard)

    def _transform(self, param: torch.nn.Parameter, working_input: torch.Tensor) -> Optional[NSInputTransform]:
        """Return the validated block transform of ``param``, or ``None`` for one whole matrix."""
        param_fqn = getattr(param, "model_name", None)
        if self.ns_transform_fn is None or param_fqn is None:
            return None
        transform = self.ns_transform_fn(param_fqn, working_input)
        if transform is not None:
            self._validate_ns_transform(transform, working_input)
        return transform

    def _compute_batched_ns_updates_with_transform(self, p_list, ns_inputs, group, no_shard=False):
        """Run batched Newton-Schulz per block and return FP32 updates.

        Same batching as ``Muon._compute_batched_ns_updates_with_transform``. Each BF16 block
        output is multiplied out of place by its FP32 factor, and split parameters are restored
        into a new FP32 tensor instead of the BF16 Newton-Schulz input buffer.
        """
        if not p_list:
            return {}
        shapes, transforms, block_updates = {}, {}, []
        blocks_by_shape = defaultdict(list)
        for param in p_list:
            shapes[param] = tuple(to_local_if_dtensor(param.data).shape) if no_shard else tuple(param.shape)
            working_input = ns_inputs[param].view(shapes[param]).contiguous()
            transforms[param] = self._transform(param, working_input)
            tensors = [working_input] if transforms[param] is None else transforms[param].tensors
            block_updates.append([None] * len(tensors))
            for block_index, tensor in enumerate(tensors):
                blocks_by_shape[self._shape_to_core_shape(tuple(tensor.shape))].append(
                    (tensor, len(block_updates) - 1, block_index))

        for records in blocks_by_shape.values():
            outputs = self._compute_batched_ns_outputs_for_tensors(
                [record[0] for record in records],
                group["ns_steps"],
                ns_variant=group["ns_variant"],
                ns_coefficients=group["ns_coefficients"],
                ns_epsilon=group["ns_epsilon"],
            )
            for (_, param_index, block_index), output in zip(records, outputs):
                block_updates[param_index][block_index] = output.float().mul_(
                    _mf_multiplier(tuple(output.shape), group))

        restored = {}
        for param, updates in zip(p_list, block_updates):
            if transforms[param] is None:
                restored[param] = updates[0]
            else:
                restored[param] = updates[0].new_empty(shapes[param])
                transforms[param].restore(updates, restored[param])
        return restored

    def _fused_broadcast_and_apply(self, valid_params, my_updates, param_compute_coord, group, hsdp_assign):
        """``Muon._fused_broadcast_and_apply`` with an FP32 instead of a BF16 broadcast buffer.

        The native buffer rounds every FP32 update to BF16 before the relay broadcast, even
        when no rank sends anything. Packing, 512-byte alignment, the relay order and the
        parameter update are unchanged; the broadcast carries twice the bytes.
        """
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        shard_sizes, local_coords, shard_pgs, total_shard_size = self._get_shard_info(hsdp_assign)
        device = to_local_if_dtensor(valid_params[0].data).device

        coord_groups = defaultdict(list)
        for p in valid_params:
            coord_groups[param_compute_coord[p]].append(p)

        alignment_elements = max(1, 512 // torch.float32.itemsize)
        pack_buffers = {}
        coord_param_offsets = {}
        for coord, coord_params in coord_groups.items():
            param_offsets = []
            total_padded_numel = 0
            for p in coord_params:
                actual_numel = p.numel()
                padded_numel = ((actual_numel + alignment_elements - 1) // alignment_elements) * alignment_elements
                param_offsets.append((total_padded_numel, actual_numel, padded_numel))
                total_padded_numel += padded_numel
            coord_param_offsets[coord] = param_offsets
            pack_buffer = torch.empty(total_padded_numel, dtype=torch.float32, device=device)
            if coord == local_coords:
                for p, (offset, actual_numel, padded_numel) in zip(coord_params, param_offsets):
                    pack_buffer[offset:offset + actual_numel].copy_(my_updates[p].reshape(-1))
                    if padded_numel > actual_numel:
                        pack_buffer[offset + actual_numel:offset + padded_numel].zero_()
            pack_buffers[coord] = pack_buffer

        if total_shard_size > 1:
            self._batched_relay_broadcast(pack_buffers, shard_pgs, shard_sizes, local_coords)

        all_local_params, all_update_shards = [], []
        for coord, coord_params in coord_groups.items():
            for p, (offset, actual_numel, _) in zip(coord_params, coord_param_offsets[coord]):
                full_update = pack_buffers[coord][offset:offset + actual_numel].view(tuple(p.shape))
                update_to_apply = chunk_update_by_layout(
                    full_update, p, hsdp_assign.layout_spec, self._param_shard_metadata.get(p))
                local_param = to_local_if_dtensor(p.data)
                all_local_params.append(local_param)
                all_update_shards.append(update_to_apply.view(local_param.shape))
        if not all_local_params:
            return
        if weight_decay != 0.0:
            torch._foreach_mul_(all_local_params, 1.0 - lr * weight_decay)  # pylint: disable=protected-access
        apply_alpha = 1.0 if group["apply_lr_in_update"] else -lr
        torch._foreach_add_(all_local_params, all_update_shards, alpha=apply_alpha)  # pylint: disable=protected-access


class MFOptimizerBuilder:
    """Same contract as ``components.optim.builders.Muon``, with ``MFMuon`` as the Muon leaf."""

    def __init__(
            self,
            muon_config: dict,
            adamw_config: dict,
            model: torch.nn.Module,
            extra_adamw_name_keywords: Optional[List[str]] = None,
            no_decay_params: Optional[List[str]] = None,
    ) -> None:
        """Route parameters like the framework builder and chain native AdamW with ``MFMuon``.

        Args:
            muon_config: Muon hyperparameters resolved from YAML.
            adamw_config: Fallback AdamW hyperparameters resolved from YAML.
            model: JT model; ``model.config`` provides the MindFormers block dimensions.
            extra_adamw_name_keywords: Additional names routed to AdamW.
            no_decay_params: Optional names excluded from weight decay.

        Raises:
            ValueError: If no parameter is eligible for Muon or ``muon_config`` sets its own
                ``ns_transform_fn``.
        """
        muon_params, adamw_params, _, _ = split_muon_adamw_params(
            model, extra_adamw_name_keywords=extra_adamw_name_keywords or ())
        if not muon_params:
            raise ValueError("Muon requires at least one eligible matrix parameter")
        adamw_groups, _ = get_adamw_param_groups(
            model, weight_decay=adamw_config.get("adamw_weight_decay", 1e-2), no_decay_params=no_decay_params,
            allowed_param_ids=[id(parameter) for parameter in adamw_params])
        muon_kwargs = _filter_optimizer_config("muon", Muon, muon_config)
        if muon_kwargs.get("ns_transform_fn") is not None:
            raise ValueError("build_mf_optimizer owns the Newton-Schulz split; remove muon_config.ns_transform_fn")
        muon_kwargs["ns_transform_fn"] = make_mf_ns_transform_fn(model.config)

        detect_dtensor_backend(adamw_groups, muon_params)
        optimizers = {}
        if adamw_groups:
            optimizers["adamw"] = _build_configured_optimizer(
                "adamw", AdamW, adamw_groups, _filter_optimizer_config("adamw", AdamW, adamw_config))
        optimizers["muon"] = _build_configured_optimizer("muon", MFMuon, muon_params, muon_kwargs)
        self.optimizer = ChainedOptimizer(model, optimizers=optimizers, flatten=bool(adamw_groups))

    def get_optimizer(self) -> ChainedOptimizer:
        """Return the chained AdamW and ``MFMuon`` runtime."""
        return self.optimizer


def build_mf_optimizer(*, model: torch.nn.Module, qk_clip_threshold: float, **kwargs: Any) -> MFOptimizerBuilder:
    """Build the MindFormers-aligned Muon/AdamW optimizer and register the JT post-step hook.

    Takes the same arguments as ``jt_optimizer.build_optimizer`` and registers the same QK-clip
    and expert-bias hook, so a recipe switches by changing only ``optimizer._target_``.

    Args:
        model: Model whose final FSDP parameter layouts are already prepared.
        qk_clip_threshold: Positive clipping threshold for QK projections.
        **kwargs: ``MFOptimizerBuilder`` options from the training recipe, including ``muon_config``.

    Returns:
        The builder holding the chained optimizer.

    Raises:
        ValueError: If ``qk_clip_threshold`` is not finite and positive.

    Note:
        MindFormers hard-codes legacy Newton-Schulz coefficients, 5 steps and epsilon 1e-7. The
        JT recipes keep ``ns_variant: legacy``, ``ns_steps: 5`` and ``ns_epsilon: 1.0e-07`` so that
        the Newton-Schulz polynomial agrees with MindFormers as well; nothing here requires them.
        The block split and the FP32 scale of ``MFMuon`` do not depend on these settings and
        work with any Newton-Schulz variant, step count or epsilon.
    """
    if not math.isfinite(qk_clip_threshold) or qk_clip_threshold <= 0:
        raise ValueError("qk_clip_threshold must be finite and positive")
    builder = MFOptimizerBuilder(model=model, **kwargs)
    builder.get_optimizer().chained_optimizers[-1].register_step_post_hook(
        partial(_after_update, model, qk_clip_threshold))
    return builder
