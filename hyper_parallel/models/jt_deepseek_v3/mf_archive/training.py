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

"""Explicit archived gradient-clipping stage; all other Trainer stages stay native."""

# This explicitly selected recipe is Torch/NPU-only.
# pylint: disable=forbidden-backend-import

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hyper_parallel.core.utils.clip_grad import _get_local_grad, _clip_grads_with_norm_
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.config import TrainerConfig
from hyper_parallel.trainer.text_trainer import TextTrainer
from hyper_parallel.models.jt_deepseek_v3.mf_archive.reductions import gradient_square_sum, sequential_sum


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
def clip_archived_gradients(model: torch.nn.Module, max_norm: float) -> torch.Tensor:
    """Calculate the L2 norm in the archive's logical layout and registration order.

    The versioned JSON contains only tensor names and shapes. Neither reference
    gradient values nor recorded norms enter this computation.
    """
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("The archived recipe requires a positive finite clipping threshold")
    order = json.loads(Path(__file__).with_name("gradient_layout.json").read_text(encoding="utf-8"))
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


class ArchivedBaseTrainer(BaseTrainer):
    """Specialize only clipping while preserving the native training stage order."""

    def prepare_optimizer_step(self) -> Any:
        """Run the native diagnostics around this explicitly selected clipping recipe."""
        self.model_integration.after_backward_before_clip()
        self.validate_fsdp_runtime_before_optimizer()
        maximum = self.config.training.max_grad_norm
        norm = clip_archived_gradients(self.model, maximum)
        post_clip_norm = None
        if self.model_integration.runtime_enabled:
            value = float(norm.item())
            post_clip_norm = value * min(float(maximum) / (value + 1e-6), 1.0)
        self.model_integration.after_clip(post_clip_norm)
        return norm


def create_archived_trainer(config: TrainerConfig) -> TextTrainer:
    """Construct the normal TextTrainer with an explicitly selected clipping stage."""
    return TextTrainer(config, base_trainer_class=ArchivedBaseTrainer)
