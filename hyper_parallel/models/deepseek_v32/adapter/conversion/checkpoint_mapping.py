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
"""Checkpoint key mapping for the DeepSeek-V3.2 training model."""

from __future__ import annotations

from typing import Any


def build_training_checkpoint_mapping(mtp_first_layer: int, mtp_depth: int) -> list[Any]:
    """Return the Transformers DeepSeek-V3.2 mapping plus the MTP layer renames.

    The released checkpoint stores MTP depth ``d`` as layer ``mtp_first_layer + d``:
    the fusion norms and projection, the shared-head norm and an ordinary decoder
    layer. Its embedding and head copies stay unmapped because the training model
    shares the trunk's.

    Args:
        mtp_first_layer: Checkpoint layer index of the first MTP depth.
        mtp_depth: Number of MTP depths; 0 adds no renames.

    Returns:
        Weight renames followed by the expert merge operations.
    """
    try:
        # Family registration stays importable without Transformers installed.
        # pylint: disable=import-outside-toplevel
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
        from transformers.core_model_loading import WeightRenaming
    except ImportError:
        return []

    mapping = []
    for depth in range(mtp_depth):
        source = rf"^model\.layers\.{mtp_first_layer + depth}\."
        target = f"mtp.layers.{depth}."
        mapping.extend([
            WeightRenaming(source + r"(enorm|hnorm|eh_proj)\.", target + r"\1."),
            WeightRenaming(source + r"shared_head\.norm\.", target + "post_norm."),
            WeightRenaming(source + r"(self_attn|mlp|input_layernorm|post_attention_layernorm)\.",
                           target + r"mtp_block.\1."),
        ])
    mapping.extend(get_checkpoint_conversion_mapping("deepseek_v32") or [])
    return mapping


def register_training_checkpoint_mapping(class_name: str, mtp_first_layer: int, mtp_depth: int) -> None:
    """Register the training model's mapping; its nested trunk then adds none.

    Args:
        class_name: Training model class name, which takes lookup priority.
        mtp_first_layer: Checkpoint layer index of the first MTP depth.
        mtp_depth: Number of MTP depths.
    """
    from transformers.conversion_mapping import (  # pylint: disable=import-outside-toplevel
        register_checkpoint_conversion_mapping,
    )

    register_checkpoint_conversion_mapping(
        class_name, build_training_checkpoint_mapping(mtp_first_layer, mtp_depth), overwrite=True,
    )
    # The root mapping already covers the nested trunk; a second, scoped copy
    # would merge its experts twice.
    register_checkpoint_conversion_mapping("DeepseekV32Model", [], overwrite=True)


__all__ = ["build_training_checkpoint_mapping", "register_training_checkpoint_mapping"]
