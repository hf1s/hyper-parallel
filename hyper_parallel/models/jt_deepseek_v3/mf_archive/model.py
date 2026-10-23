# Copyright 2026 Huawei Technologies Co., Ltd.
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

"""An explicit, constrained model factory for the fixed MF r1.8.0 archive."""

# This explicitly selected recipe is Torch/NPU-only.
# pylint: disable=forbidden-backend-import

import copy
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch_npu
from torch import nn

from hyper_parallel.components.checkpoint.weight_conversion import get_model_conversion_mapping
from hyper_parallel.components.losses import calculate_mtp_loss
from hyper_parallel.components.losses.mtp import iter_mtp_targets
from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager, _WrapModuleInfo
from hyper_parallel.models._transformers.model_builder import (
    _build_replacement_context, apply_model_infrastructure, instantiate_infrastructure,
)
from hyper_parallel.models.build_options import FSDP2Config, get_device_id
from hyper_parallel.models.jt_deepseek_v3 import modeling_jt_deepseek_v3 as jt
from hyper_parallel.models.jt_deepseek_v3.adapter.jt_builder import _bind_statistics_groups, _load_reference_state
from hyper_parallel.models.jt_deepseek_v3.mf_archive.layers import (
    ArchivedDecoder, ArchivedEmbedding, ArchivedHead, ArchivedNorm,
)
from hyper_parallel.models.jt_deepseek_v3.mf_archive.losses import archived_cross_entropy
from hyper_parallel.models.replacement import _apply_module_replacement_actions


def validate_profile(config: Any) -> None:
    """Reject configurations outside the archive's measured shape and semantics contract."""
    expected = {"hidden_size": 5120, "vocab_size": 152064, "num_hidden_layers": 2,
                "intermediate_size": 12288, "moe_intermediate_size": 1536,
                "num_attention_heads": 128, "q_lora_rank": 1536, "kv_lora_rank": 512,
                "qk_rope_head_dim": 64, "qk_nope_head_dim": 128, "v_head_dim": 128,
                "n_routed_experts": 8, "n_shared_experts": 2, "num_nextn_predict_layers": 1,
                "num_experts_per_tok": 2, "n_group": 1, "topk_group": 1,
                "norm_topk_prob": True, "enable_chunk_loss": True, "chunk_loss_size": 4096,
                "tie_word_embeddings": False, "attention_dropout": 0.0, "mlp_bias": False,
                "attention_bias": False, "use_pad_tokens": True, "hidden_act": "silu",
                "rms_norm_eps": 1e-6, "routed_scaling_factor": 2.827,
                "first_k_dense_replace": 1, "mlp_layer_types": ["dense", "sparse"]}
    mismatches = {key: (getattr(config, key, None), value)
                  for key, value in expected.items() if getattr(config, key, None) != value}
    if mismatches:
        raise ValueError(f"Configuration is outside the fixed archived MF profile: {mismatches}")
    if config.rope_parameters.get("rope_type") != "default" or config.rope_parameters.get("rope_theta") != 5000000:
        raise ValueError("The archived profile requires the measured unscaled RoPE configuration")


class ArchivedModel(jt.JTDeepseekV3ForCausalLM):
    """Opt-in legacy packed-MTP objective and numerical component choices.

    Unlike the current native objective, this archive allows the MTP shift to
    cross document boundaries. It is reproduced for checkpoint comparison only;
    the native document-aware objective remains the recommended default.
    """

    def __init__(self, config: Any) -> None:
        """Build the explicitly selected archive layers with native parameter names."""
        validate_profile(config)
        super().__init__(config)
        self.model.embed_tokens = ArchivedEmbedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.lm_head = ArchivedHead(config.hidden_size, config.vocab_size, bias=False)
        self.model.layers = nn.ModuleList([
            ArchivedDecoder(config, index, shared_output=index == config.num_hidden_layers - 1)
            for index in range(config.num_hidden_layers)])
        self.model.norm = ArchivedNorm(config.hidden_size, config.rms_norm_eps, output_dtype=torch.bfloat16)
        mtp_config = copy.deepcopy(config)
        mtp_config.mlp_layer_types = ["sparse"]
        layer = self.mtp.layers[0]
        layer.enorm = ArchivedNorm(config.hidden_size, config.rms_norm_eps)
        layer.hnorm = ArchivedNorm(config.hidden_size, config.rms_norm_eps, output_dtype=torch.bfloat16)
        layer.transformer_layer = ArchivedDecoder(mtp_config, 0, mtp=True)
        layer.final_layernorm = ArchivedNorm(config.hidden_size, config.rms_norm_eps)

    def _projection_loss(self, hidden, targets):
        return jt.projected_cross_entropy(
            hidden, targets, head=self.lm_head, loss_fn=archived_cross_entropy,
            vocab_size=self.vocab_size, chunk_size=self.chunk_loss_size,
            sequence_parallel_size=self.loss_sequence_parallel_size, tp_mesh=self.loss_tp_mesh)


    def compute_jt_losses(self, input_ids: torch.Tensor, labels: torch.Tensor,
                          loss_mask: torch.Tensor, *, position_ids: torch.Tensor | None = None,
                          actual_seq_len: tuple[int, ...] | None = None, sequence_start: int = 0
                          ) -> dict[str, torch.Tensor]:
        """Compute the LM, MTP and router auxiliary losses on pre-shifted labels.

        Args:
            input_ids: Token IDs, ``[1, sequence]``.
            labels: Targets already shifted by one token.
            loss_mask: Mask of the supervised targets.
            position_ids: Optional sequence positions for RoPE.
            actual_seq_len: Global cumulative ends of independent documents.
            sequence_start: Global offset of the local input interval.

        Returns:
            ``lm_loss``, ``mtp_loss`` and ``aux_loss`` as 0-d tensors.

        Raises:
            ValueError: If the inputs are not one sequence with matching shapes.
        """
        cfg = self.config
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("JT sequence loss currently requires a two-dimensional batch-one input")
        sequence_length = input_ids.shape[1]
        if labels.shape != input_ids.shape or loss_mask.shape != input_ids.shape:
            raise ValueError("Tokens, pre-shifted labels and loss mask must have identical shapes")
        ends = (sequence_length,) if actual_seq_len is None else tuple(actual_seq_len)
        if (not ends or ends[0] <= 0 or any(left >= right for left, right in zip(ends, ends[1:]))
                or sequence_start < 0 or sequence_start + sequence_length > ends[-1]):
            raise ValueError("JT cumulative document ends must increase and cover the local token interval")
        indices = torch.arange(sequence_start, sequence_start + sequence_length, device=input_ids.device)
        sequence_end_mask = None
        # Historical MF shifts across packed documents; this is deliberately opt-in.
        absolute_positions = indices.unsqueeze(0)
        if position_ids is not None and not torch.equal(position_ids, absolute_positions):
            raise ValueError("The archived recipe requires reset_position_ids=False")
        position_ids = absolute_positions
        # Preserve the archived FP32 inverse-frequency construction.
        dim = cfg.qk_rope_head_dim
        inverse = 1.0 / (cfg.rope_parameters["rope_theta"] ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        inverse = torch.from_numpy(inverse.astype(np.float32)).to(input_ids.device)
        if position_ids.shape != input_ids.shape:
            raise ValueError("JT position_ids must match input_ids")
        frequency = position_ids.to(device=input_ids.device, dtype=torch.float32).unsqueeze(-1) * inverse
        frequency = torch.cat((frequency, frequency), dim=-1)
        hidden = self.model.embed_tokens(input_ids)
        # FP32 tables; JTDeepseekV3RotaryEmbedding rotates in FP32 and rounds to the activation dtype once.
        attention_kwargs = {"position_embeddings": (frequency.cos(), frequency.sin()), "actual_seq_len": ends}
        auxiliary = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for layer in self.model.layers:
            hidden = layer(hidden, **attention_kwargs)
            if hasattr(layer.mlp, "pop_auxiliary_loss"):
                auxiliary = auxiliary + layer.mlp.pop_auxiliary_loss()
        valid = (loss_mask != 0) & (labels >= 0)
        targets = labels.masked_fill(~valid, -100)
        normalized = self.model.norm(hidden)
        lm_loss = (self._projection_loss(normalized, targets) if self.enable_chunk_loss
                   else self._token_loss(self.lm_head(normalized), labels, loss_mask))
        mtp_output = self.mtp(hidden, input_ids, embedding=self.model.embed_tokens,
                              head=None if self.enable_chunk_loss else self.lm_head,
                              decoder_kwargs=attention_kwargs, shift_fn=self.shift_mtp_inputs,
                              sequence_end_mask=sequence_end_mask)
        for layer in self.mtp.layers:
            auxiliary = auxiliary + layer.transformer_layer.mlp.pop_auxiliary_loss()
        shift_targets = partial(self.shift_mtp_inputs, pad_value=-100)
        if self.enable_chunk_loss:
            mtp_loss = torch.zeros_like(lm_loss)
            depths = len(mtp_output.prediction_hidden_states)
            targets_by_depth = iter_mtp_targets(targets, depths, shift_fn=shift_targets,
                                                sequence_end_mask=sequence_end_mask)
            for prediction, future_targets in zip(mtp_output.prediction_hidden_states, targets_by_depth):
                mtp_loss = mtp_loss + self._projection_loss(prediction, future_targets) * (cfg.mtp_loss_factor / depths)
        else:
            mtp_loss = calculate_mtp_loss(mtp_output.logits, targets, self._mtp_token_loss, vocab_size=cfg.vocab_size,
                                          loss_factor=cfg.mtp_loss_factor, shift_fn=shift_targets,
                                          sequence_end_mask=sequence_end_mask)
        return {"lm_loss": lm_loss, "mtp_loss": mtp_loss, "aux_loss": auxiliary}


class ArchivedPrecisionFSDP(FSDP2Manager):
    """Give the named archive components separate FP32 FSDP compute policies."""

    def _find_wrap_modules(self, model, metadata_by_parameter=None):
        units = super()._find_wrap_modules(model, metadata_by_parameter)
        selected = {id(unit.module) for unit in units}
        for name, module in dict(model.named_modules()).items():
            if getattr(module, "keep_compute_in_fp32", False) and id(module) not in selected:
                units.append(_WrapModuleInfo(name, module))
                selected.add(id(module))
        return units

    def _parallelize_child_units(self, wrap_modules, owner_by_parameter, metadata_by_parameter,
                                replicate_params, dense_fsdp_kwargs):
        precision = dict(dense_fsdp_kwargs)
        precision["mp_policy"] = replace(dense_fsdp_kwargs["mp_policy"], param_dtype=torch.float32,
                                          cast_forward_inputs=False, output_dtype=None)
        embedding = dict(precision)
        embedding["mp_policy"] = replace(precision["mp_policy"],
                                          output_dtype=dense_fsdp_kwargs["mp_policy"].param_dtype)
        count = 0
        for unit in sorted(wrap_modules, key=lambda item: item.fqn.count("."), reverse=True):
            options = precision if getattr(unit.module, "keep_compute_in_fp32", False) else dense_fsdp_kwargs
            if options is precision and isinstance(unit.module, nn.Embedding):
                options = embedding
            count += super()._parallelize_child_units(
                [unit], owner_by_parameter, metadata_by_parameter, replicate_params, options)
        return count


def build_archived_model(*, config: dict[str, Any], reference_weights: str | Path, archive_profile: str,
                         distributed_setup: Any, **infrastructure_options: Any) -> nn.Module:
    """Build the explicit single-card archived recipe using native state loading.

    Args:
        config: The measured architecture; unsupported shapes fail validation.
        reference_weights: Converted FP32 model.npz from the fixed MF anchor.
        archive_profile: Explicit acknowledgement of the versioned legacy recipe.
        distributed_setup: Current Trainer setup, with every parallel degree one.
        **infrastructure_options: Native infrastructure options.

    Returns:
        The model with the standard JT parameter names and checkpoint interface.
    """
    if archive_profile != "mf-r180-cann85-910b3-v1":
        raise ValueError("Select archive_profile='mf-r180-cann85-910b3-v1' explicitly")
    cann = Path("/usr/local/Ascend/cann/opp/version.info").read_text(encoding="utf-8")
    if "Version=9.2." not in cann:
        raise ValueError("This compatibility implementation requires the validated CANN 9.2.X HP runtime")
    mesh = distributed_setup.mesh_context
    if any(getattr(mesh, name) != 1 for name in ("tp_size", "cp_size", "ep_size", "pp_size", "dp_size")):
        raise ValueError("The archived MF profile supports only one device")
    if torch.npu.get_device_name() != "Ascend910B3":
        raise ValueError("The archived CANN schedules are validated only on Ascend910B3")
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.use_deterministic_algorithms(True)
    config = jt.JTDeepseekV3Config(**config)
    framework_setup = replace(distributed_setup, module_replacements=(),
                              strategy_config=distributed_setup.strategy_config or FSDP2Config())
    planner, _ = instantiate_infrastructure(distributed_setup=framework_setup)
    fsdp = ArchivedPrecisionFSDP(framework_setup.strategy_config, mesh,
                                 fp32_main_params=framework_setup.fp32_main_params)
    with torch.device("meta"):
        model = ArchivedModel(config)
        model, _ = _apply_module_replacement_actions(
            model, getattr(distributed_setup, "module_replacements", None),
            weights_mapping=get_model_conversion_mapping(model),
            context=_build_replacement_context(distributed_setup, None))
    model.to_empty(device="cpu")
    with np.load(Path(reference_weights) / "model.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    _load_reference_state(model, arrays)
    device = torch.device(mesh.device_mesh.device_type, get_device_id())
    model.to(device)
    model = apply_model_infrastructure(
        model, mesh=mesh, sharding_planner=planner, fsdp2_manager=fsdp,
        distributed_setup=framework_setup, device=device, is_meta_device=False,
        is_hf_model=True, **infrastructure_options)
    _bind_statistics_groups(model, mesh)
    return model
