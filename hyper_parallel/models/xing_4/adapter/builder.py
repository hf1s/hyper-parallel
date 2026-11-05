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
"""Build the Xing4 model from the recipe's configuration fields."""

from __future__ import annotations

from typing import Any

from transformers import PreTrainedModel

from hyper_parallel.models import HyperAutoModelForCausalLM
from hyper_parallel.models.xing_4.transformer.configuration_xing4_0 import Xing4_0Config

__all__ = ["build_model"]

#: Training fields the model reads and the bundled inference configuration has no default for.
_REQUIRED_FIELDS = ("mhc_init_gating_factor", "mhc_layernorm_epsilon")
#: The same, for the routed layers; required only when the geometry builds an MoE.
_REQUIRED_MOE_FIELDS = ("moe_aux_loss_coeff", "moe_router_bias_update_rate",
                        "moe_router_enable_expert_bias", "use_pad_tokens")
#: Recipe spellings this adapter translates onto the bundled configuration's field names.
_FIELD_ALIASES = (("num_residual_streams", "hc_mult"),
                  ("mhc_sinkhorn_iterations", "hc_sinkhorn_iters"),
                  ("router_jitter_noise", "moe_input_jitter_eps"))


def build_model(*, config: dict[str, Any], **options: Any) -> PreTrainedModel:
    """Build and parallelize the model through the shared builder.

    Args:
        config: Model configuration with HF fields and HP training options.
        **options: Construction options the Trainer passes to the shared builder.

    Returns:
        The constructed, parallelized model.
    """
    return HyperAutoModelForCausalLM.from_config(create_config(config), **options)


def create_config(config: dict[str, Any]) -> Xing4_0Config:
    """Accept HF configuration fields and equivalent training aliases.

    ``PretrainedConfig`` keeps an unknown keyword as a plain attribute, so a recipe spelling this
    function does not translate would be stored and never read. Translation, not validation, is
    therefore the job here; a malformed value fails where it is used.

    Args:
        config: Model fields supplied by the training recipe.

    Returns:
        The bundled HF configuration carrying HP training options.

    Raises:
        ValueError: If a field arrives under two names with different values, if the recipe
            requests a routing mode this adapter does not implement, or if a training field the
            model reads has no value.
    """
    values = dict(config)
    for recipe_name, hf_name in _FIELD_ALIASES:
        if recipe_name not in values:
            continue
        value = values.pop(recipe_name)
        if values.setdefault(hf_name, value) != value:
            raise ValueError(f"Conflicting {recipe_name} and {hf_name}")
    # The router reads this unconditionally; the bundled inference configuration has no field for it.
    values.setdefault("moe_input_jitter_eps", 0.0)
    # The bundled router is the learned top-k one; these select other routing modes entirely.
    if values.get("moe_n_hash_layers", 0) or values.get("moe_router_force_expert_balance", False):
        raise ValueError("Xing4 uses the official learned top-k router")
    # get_is_hf_model selects the custom-model path from architectures[0]; without this the shared
    # builder would construct a Transformers class instead of the registered training model.
    values["architectures"] = ["Xing4ForCausalLM"]
    # The recipe spells the residual-mixing bounds as one pair; the bundled configuration splits
    # them. Its default is the reference configuration's [-30, 30], so the bundled forward clamps
    # unless the recipe nulls both bounds. The fused mHC operators take no bound and refuse a
    # configured clamp at the point they are installed.
    clamp = values.pop("hc_sinkhorn_clamp", None)
    if clamp is not None:
        bounds = tuple(clamp)
        if len(bounds) != 2 or not bounds[0] < bounds[1]:
            raise ValueError(f"hc_sinkhorn_clamp must be an increasing [low, high] pair, got {clamp!r}")
        for name, bound in zip(("mhc_h_res_clamp_min", "mhc_h_res_clamp_max"), bounds):
            if values.setdefault(name, bound) != bound:
                raise ValueError(f"Conflicting hc_sinkhorn_clamp and {name}")
    result = Xing4_0Config(**values)
    if not hasattr(result, "attention_softmax_scale"):
        result.attention_softmax_scale = None
    required = list(_REQUIRED_FIELDS)
    if result.num_nextn_predict_layers:
        required.append("mtp_loss_factor")
    if result.n_routed_experts is not None and (
            result.num_hidden_layers + result.num_nextn_predict_layers > result.first_k_dense_replace):
        required.extend(_REQUIRED_MOE_FIELDS)
    missing = [name for name in required if getattr(result, name, None) is None]
    if missing:
        raise ValueError(f"Xing4 training configuration requires: {', '.join(missing)}")
    if result.use_cache:
        raise ValueError("The training recipe must explicitly disable the KV cache")
    return result
