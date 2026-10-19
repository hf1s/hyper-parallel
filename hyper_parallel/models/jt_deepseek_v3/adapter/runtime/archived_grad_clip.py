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

"""Archived MF gradient clipping, applied as an optimizer step pre-hook.

``clip_archived_gradients`` computes the L2 norm over the archive's logical
gradients in their registration order with the archived FP32 reduction schedule,
then clips every local gradient. ``register_archived_clip`` runs it once per
optimizer step, after gradients are final and before any leaf optimizer updates
a parameter. No Trainer stage is replaced: the archived recipe sets
``training.max_grad_norm: .inf`` so the framework ``clip_grad_norm_`` only
measures the norm (its coefficient clamps to exactly 1.0), and the archived
threshold lives on the optimizer target. The default JT path keeps the
framework ``clip_grad_norm_``.
"""

# This explicitly selected recipe is Torch/NPU-only.
# pylint: disable=forbidden-backend-import

import json
import math
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.hooks import RemovableHandle

from hyper_parallel.core.utils.clip_grad import _get_local_grad, _clip_grads_with_norm_
from hyper_parallel.models.jt_deepseek_v3.archived_reductions import gradient_square_sum, sequential_sum

GRADIENT_LAYOUT_PATH = Path(__file__).resolve().with_name("archived_gradient_layout.json")


def local_gradient(parameters: dict[str, torch.nn.Parameter], name: str) -> torch.Tensor:
    """Read one live local FP32 gradient without modifying it."""
    value = _get_local_grad(parameters[name])
    if value is None:
        raise ValueError('Missing preclip gradient: ' + name)
    if hasattr(value, 'to_local'):
        value = value.to_local()
    if value.dtype != torch.float32:
        raise ValueError('Expected FP32 gradient: ' + name)
    return value.detach()


def mf_gradient(parameters: dict[str, torch.nn.Parameter], name: str) -> torch.Tensor:
    """Expose one archived logical gradient using native JT parameter storage."""
    direct = {'embedding.word_embeddings.weight': 'model.embed_tokens.weight',
              'output_layer.weight': 'lm_head.weight', 'decoder.final_layernorm.weight': 'model.norm.weight'}
    if name in direct:
        return local_gradient(parameters, direct[name])
    target = name.replace('decoder.layers.', 'model.layers.')
    target = target.replace('.self_attention.', '.self_attn.')
    target = target.replace('.pre_mlp_layernorm.', '.post_attention_layernorm.')
    for old, new in [('q_layernorm', 'q_a_layernorm'), ('kv_layernorm', 'kv_a_layernorm'),
                     ('linear_q_up_proj', 'q_b_proj'), ('linear_kv_up_proj', 'kv_b_proj'),
                     ('linear_proj', 'o_proj')]:
        target = target.replace('.' + old + '.', '.' + new + '.')
    for suffix, selection in [('linear_q_down_proj.weight', slice(0, 1536)),
                              ('linear_kv_down_proj.weight', slice(1536, 2112))]:
        if target.endswith(suffix):
            return local_gradient(parameters, target[:-len(suffix)] + 'linear_qkv.weight')[selection]
    if target.endswith('.linear_fc1.weight'):
        prefix = target[:-len('linear_fc1.weight')]
        gate = local_gradient(parameters, prefix + 'gate_proj.weight')
        up = local_gradient(parameters, prefix + 'up_proj.weight')
        return torch.stack((gate, up), dim=1).flatten(0, 1)
    if target.endswith('.experts.weight1'):
        gradient = local_gradient(parameters, target[:-len('weight1')] + 'gate_up_proj')
        return gradient.transpose(1, 2).contiguous().flatten(0, 1)
    if target.endswith('.experts.weight2'):
        gradient = local_gradient(parameters, target[:-len('weight2')] + 'down_proj')
        return gradient.transpose(1, 2).contiguous().flatten(0, 1)
    target = target.replace('.linear_fc2.weight', '.down_proj.weight').replace('.mlp.router.weight', '.mlp.gate.weight')
    return local_gradient(parameters, target)


@torch.no_grad()
def clip_archived_gradients(model: torch.nn.Module, max_norm: float, *,
                            layout_path: Path = GRADIENT_LAYOUT_PATH) -> torch.Tensor:
    """Calculate the L2 norm in the archive's logical layout and registration order.

    The versioned JSON contains only tensor names and shapes. Neither reference
    gradient values nor recorded norms enter this computation.

    Args:
        model: Single-card model whose FP32 local gradients are clipped in place.
        max_norm: Positive finite clipping threshold.
        layout_path: JSON list of ``{"name", "shape"}`` logical gradients in
            archive registration order; defaults to the JT archive layout.

    Returns:
        The FP32 pre-clip gradient norm.
    """
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("The archived recipe requires a positive finite clipping threshold")
    order = json.loads(Path(layout_path).read_text(encoding="utf-8"))
    parameters = dict(model.named_parameters())
    sums = []
    for entry in order:
        gradient = mf_gradient(parameters, entry["name"])
        if list(gradient.shape) != entry["shape"]:
            raise ValueError(f"Archived gradient layout mismatch: {entry['name']}")
        sums.append(gradient_square_sum(gradient))
    gradients = [_get_local_grad(parameter) for parameter in parameters.values()]
    if any(value is None or value.dtype != torch.float32 for value in gradients):
        raise ValueError("The archived recipe requires complete FP32 local gradients")
    total = sequential_sum(np.asarray(sums, dtype=np.float32))
    if not np.isfinite(total):
        raise ValueError("Nonfinite archived gradient norm")
    norm = torch.tensor(total, dtype=torch.float32, device=gradients[0].device).sqrt()
    _clip_grads_with_norm_(gradients, float(max_norm), norm, None)
    return norm


def _clip_before_update(model: torch.nn.Module, max_norm: float, layout_path: Path,
                        optimizer: Any, args: tuple, kwargs: dict) -> None:
    """Step pre-hook: clip with the archived norm before the first leaf optimizer updates."""
    del optimizer, args, kwargs
    clip_archived_gradients(model, max_norm, layout_path=layout_path)


def register_archived_clip(optimizer: Any, model: torch.nn.Module, max_norm: float, *,
                           layout_path: Path = GRADIENT_LAYOUT_PATH) -> RemovableHandle:
    """Clip with the archived norm once per step, before any leaf of ``optimizer`` updates a parameter.

    ``ChainedOptimizer.step`` steps its leaves in order, so a step pre-hook on the
    first leaf runs exactly once per optimizer step, after the gradients are final
    (including the mixed-precision wrapper's gradient preparation) and before any
    update. The JT QK-clip post-hook keeps running after the last leaf.

    Args:
        optimizer: ``ChainedOptimizer`` returned by the archived optimizer builder.
        model: Single-card model whose FP32 local gradients are clipped in place.
        max_norm: Positive finite archived clipping threshold.
        layout_path: Archived logical gradient layout; defaults to the JT archive layout.

    Returns:
        Handle that removes the hook.

    Raises:
        ValueError: If ``max_norm`` is not positive and finite.
    """
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("The archived recipe requires a positive finite clipping threshold")
    return optimizer.chained_optimizers[0].register_step_pre_hook(
        partial(_clip_before_update, model, float(max_norm), Path(layout_path)))
