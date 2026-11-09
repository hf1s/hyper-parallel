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
"""Build a depth- and expert-cropped DeepSeek-V3.2 model from configuration only."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig, PreTrainedModel

from hyper_parallel.models.deepseek_v32.modeling_deepseek_v32 import build_model

# ``DeepseekV32Config.__post_init__`` derives these per-layer lists from the
# layer count and ``first_k_dense_replace``. Copying the released 61-entry
# lists into a cropped config would keep the released dense/MoE pattern, so
# they are re-derived instead.
_LAYER_DERIVED_FIELDS = ("mlp_layer_types", "layer_types")


def build_cropped_deepseek_v32_config(
        config_path: str,
        *,
        num_hidden_layers: int,
        first_k_dense_replace: int,
        n_routed_experts: int,
        num_nextn_predict_layers: int = 0,
        local_files_only: bool = True,
) -> PretrainedConfig:
    """Return the released DeepSeek-V3.2 configuration cropped in depth and experts.

    Hidden, MLA, indexer and vocabulary dimensions stay as released. The FP8
    ``quantization_config`` describes the released checkpoint storage rather
    than this randomly initialized model, so it is dropped.

    Args:
        config_path: Hugging Face DeepSeek-V3.2 directory or repository id.
        num_hidden_layers: Decoder layers retained in the cropped model.
        first_k_dense_replace: Leading layers that keep the dense MLP; the
            remaining layers use the MoE block.
        n_routed_experts: Routed experts per MoE layer. The released group
            routing scores each of ``n_group`` groups by its top-2 experts and
            keeps ``topk_group`` groups, so every group needs two experts and
            the kept groups must hold ``num_experts_per_tok`` experts.
        num_nextn_predict_layers: MTP depth; 0 disables MTP.
        local_files_only: Disable implicit Hub downloads when true.

    Returns:
        The cropped configuration with ``use_cache`` disabled.
    """
    released = AutoConfig.from_pretrained(
        config_path,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    if getattr(released, "model_type", None) != "deepseek_v32":
        raise ValueError(
            "config_path must contain a DeepSeek-V3.2 configuration; "
            f"got model_type={getattr(released, 'model_type', None)!r}"
        )
    if num_hidden_layers <= 0 or not 0 <= first_k_dense_replace <= num_hidden_layers:
        raise ValueError(
            "need num_hidden_layers > 0 and 0 <= first_k_dense_replace <= num_hidden_layers; "
            f"got num_hidden_layers={num_hidden_layers}, first_k_dense_replace={first_k_dense_replace}"
        )
    experts_per_group, remainder = divmod(n_routed_experts, released.n_group)
    if remainder or experts_per_group < 2 or experts_per_group * released.topk_group < released.num_experts_per_tok:
        raise ValueError(
            f"n_routed_experts={n_routed_experts} cannot keep the released routing "
            f"(n_group={released.n_group}, topk_group={released.topk_group}, "
            f"num_experts_per_tok={released.num_experts_per_tok}): it must be a multiple of n_group "
            "with at least two experts per group, and topk_group groups must hold num_experts_per_tok experts"
        )

    values = released.to_dict()
    for name in (*_LAYER_DERIVED_FIELDS, "quantization_config"):
        values.pop(name, None)
    values.update(
        num_hidden_layers=num_hidden_layers,
        first_k_dense_replace=first_k_dense_replace,
        n_routed_experts=n_routed_experts,
        num_nextn_predict_layers=num_nextn_predict_layers,
        use_cache=False,
    )
    return type(released).from_dict(values)


def build_cropped_deepseek_v32(
        config_path: str,
        *,
        num_hidden_layers: int = 2,
        first_k_dense_replace: int = 1,
        n_routed_experts: int = 16,
        num_nextn_predict_layers: int = 0,
        local_files_only: bool = True,
        **build_kwargs: Any,
) -> PreTrainedModel:
    """Create a randomly initialized cropped DeepSeek-V3.2 model.

    The function calls ``from_config`` instead of ``from_pretrained``: it
    reads only the Hugging Face configuration from ``config_path`` and loads
    no checkpoint tensor.

    Args:
        config_path: Local Hugging Face DeepSeek-V3.2 directory.
        num_hidden_layers: Decoder layers retained in the cropped model.
        first_k_dense_replace: Leading dense-MLP layers.
        n_routed_experts: Routed experts per MoE layer.
        num_nextn_predict_layers: MTP depth; 0 disables MTP.
        local_files_only: Disable implicit Hub downloads when true.
        **build_kwargs: Model build options, both those set in YAML
            (``torch_dtype``, ``attn_implementation``, ``validate_placement``,
            training objectives) and those the Trainer injects
            (``distributed_setup``, ``activation_checkpoint``, ...).

    Returns:
        A parallelized, randomly initialized cropped DeepSeek-V3.2 model.
    """
    config = build_cropped_deepseek_v32_config(
        config_path,
        num_hidden_layers=num_hidden_layers,
        first_k_dense_replace=first_k_dense_replace,
        n_routed_experts=n_routed_experts,
        num_nextn_predict_layers=num_nextn_predict_layers,
        local_files_only=local_files_only,
    )
    return build_model(config=config, **build_kwargs)
