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

"""Opt-in live-gradient replay of the measured single-card MF DVM norm schedule.

This experiment is restricted to the 46 native parameter layouts from the
2-trunk-layer, 1-MTP JT reference on 910B3. It reads no recorded gradients.
"""
import torch
from .convert_state import mapping_for
from hyper_parallel.core.utils.clip_grad import _get_local_grad

NATIVE_SCHEDULE = [{'name': 'embedding.word_embeddings.weight',
  'shape': [152064, 5120],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'decoder.layers.0.self_attention.linear_proj.weight',
  'shape': [5120, 16384],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'decoder.layers.0.self_attention.linear_q_down_proj.weight',
  'shape': [1536, 5120],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'decoder.layers.0.self_attention.linear_q_up_proj.weight',
  'shape': [24576, 1536],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'decoder.layers.0.self_attention.linear_kv_down_proj.weight',
  'shape': [576, 5120],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'decoder.layers.0.self_attention.linear_kv_up_proj.weight',
  'shape': [32768, 512],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'decoder.layers.0.mlp.linear_fc1.weight',
  'shape': [24576, 5120],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'decoder.layers.0.mlp.linear_fc2.weight',
  'shape': [5120, 12288],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'decoder.layers.1.self_attention.linear_proj.weight',
  'shape': [5120, 16384],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'decoder.layers.1.self_attention.linear_q_down_proj.weight',
  'shape': [1536, 5120],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'decoder.layers.1.self_attention.linear_q_up_proj.weight',
  'shape': [24576, 1536],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'decoder.layers.1.self_attention.linear_kv_down_proj.weight',
  'shape': [576, 5120],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'decoder.layers.1.self_attention.linear_kv_up_proj.weight',
  'shape': [32768, 512],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'decoder.layers.1.mlp.router.weight',
  'shape': [8, 5120],
  'tile': 1024,
  'simd_width': 64,
  'workers': 40},
 {'name': 'decoder.layers.1.mlp.experts.weight1',
  'shape': [40960, 3072],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'decoder.layers.1.mlp.experts.weight2',
  'shape': [12288, 5120],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'decoder.layers.1.mlp.shared_experts.linear_fc1.weight',
  'shape': [6144, 5120],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'decoder.layers.1.mlp.shared_experts.linear_fc2.weight',
  'shape': [5120, 3072],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'mtp.layers.0.eh_proj.weight',
  'shape': [5120, 10240],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'mtp.layers.0.transformer_layer.self_attention.linear_proj.weight',
  'shape': [5120, 16384],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'mtp.layers.0.transformer_layer.self_attention.linear_q_down_proj.weight',
  'shape': [1536, 5120],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'mtp.layers.0.transformer_layer.self_attention.linear_q_up_proj.weight',
  'shape': [24576, 1536],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'mtp.layers.0.transformer_layer.self_attention.linear_kv_down_proj.weight',
  'shape': [576, 5120],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'mtp.layers.0.transformer_layer.self_attention.linear_kv_up_proj.weight',
  'shape': [32768, 512],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'mtp.layers.0.transformer_layer.mlp.router.weight',
  'shape': [8, 5120],
  'tile': 1024,
  'simd_width': 64,
  'workers': 40},
 {'name': 'mtp.layers.0.transformer_layer.mlp.experts.weight1',
  'shape': [40960, 3072],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'mtp.layers.0.transformer_layer.mlp.experts.weight2',
  'shape': [12288, 5120],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'mtp.layers.0.transformer_layer.mlp.shared_experts.linear_fc1.weight',
  'shape': [6144, 5120],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'mtp.layers.0.transformer_layer.mlp.shared_experts.linear_fc2.weight',
  'shape': [5120, 3072],
  'tile': 8144,
  'simd_width': 16,
  'workers': 40},
 {'name': 'output_layer.weight', 'shape': [152064, 5120], 'tile': 8144, 'simd_width': 16, 'workers': 40},
 {'name': 'decoder.layers.0.input_layernorm.weight',
  'shape': [5120],
  'tile': 5120,
  'simd_width': 64,
  'workers': 1},
 {'name': 'decoder.layers.0.self_attention.q_layernorm.weight',
  'shape': [1536],
  'tile': 1536,
  'simd_width': 64,
  'workers': 1},
 {'name': 'decoder.layers.0.self_attention.kv_layernorm.weight',
  'shape': [512],
  'tile': 512,
  'simd_width': 64,
  'workers': 1},
 {'name': 'decoder.layers.0.pre_mlp_layernorm.weight',
  'shape': [5120],
  'tile': 5120,
  'simd_width': 64,
  'workers': 1},
 {'name': 'decoder.layers.1.input_layernorm.weight',
  'shape': [5120],
  'tile': 5120,
  'simd_width': 64,
  'workers': 1},
 {'name': 'decoder.layers.1.self_attention.q_layernorm.weight',
  'shape': [1536],
  'tile': 1536,
  'simd_width': 64,
  'workers': 1},
 {'name': 'decoder.layers.1.self_attention.kv_layernorm.weight',
  'shape': [512],
  'tile': 512,
  'simd_width': 64,
  'workers': 1},
 {'name': 'decoder.layers.1.pre_mlp_layernorm.weight',
  'shape': [5120],
  'tile': 5120,
  'simd_width': 64,
  'workers': 1},
 {'name': 'decoder.final_layernorm.weight',
  'shape': [5120],
  'tile': 5120,
  'simd_width': 64,
  'workers': 1},
 {'name': 'mtp.layers.0.enorm.weight', 'shape': [5120], 'tile': 5120, 'simd_width': 64, 'workers': 1},
 {'name': 'mtp.layers.0.hnorm.weight', 'shape': [5120], 'tile': 5120, 'simd_width': 64, 'workers': 1},
 {'name': 'mtp.layers.0.transformer_layer.input_layernorm.weight',
  'shape': [5120],
  'tile': 5120,
  'simd_width': 64,
  'workers': 1},
 {'name': 'mtp.layers.0.transformer_layer.self_attention.q_layernorm.weight',
  'shape': [1536],
  'tile': 1536,
  'simd_width': 64,
  'workers': 1},
 {'name': 'mtp.layers.0.transformer_layer.self_attention.kv_layernorm.weight',
  'shape': [512],
  'tile': 512,
  'simd_width': 64,
  'workers': 1},
 {'name': 'mtp.layers.0.transformer_layer.pre_mlp_layernorm.weight',
  'shape': [5120],
  'tile': 5120,
  'simd_width': 64,
  'workers': 1},
 {'name': 'mtp.layers.0.final_layernorm.weight',
  'shape': [5120],
  'tile': 5120,
  'simd_width': 64,
  'workers': 1}]


def sumx(values: torch.Tensor, width: int) -> torch.Tensor:
    """Replay the measured DVM row-halving and lane-pair addition tree."""
    if values.shape[-1] % width:
        raise ValueError('Unaligned DVM SumX input')
    work = values.reshape(-1, values.shape[-1] // width, width)
    while work.shape[1] > 1:
        count = work.shape[1]
        half = count // 2
        reduced = work[:, :half] + work[:, half:2 * half]
        if count % 2:
            reduced[:, 0] = reduced[:, 0] + work[:, -1]
        work = reduced
    work = work[:, 0]
    while work.shape[1] > 1:
        work = work[:, ::2] + work[:, 1::2]
    return work[:, 0]


def sequential(values: torch.Tensor) -> torch.Tensor:
    """Accumulate the last axis without a reassociated parallel reduction."""
    result = values[..., 0]
    for index in range(1, values.shape[-1]):
        result = result + values[..., index]
    return result


def square_sum(gradient: torch.Tensor, spec: dict) -> torch.Tensor:
    """Reduce a live native-layout FP32 gradient with its measured schedule."""
    if tuple(gradient.shape) != tuple(spec['shape']) or gradient.dtype != torch.float32:
        raise ValueError(f'Native norm shape/dtype mismatch for {spec["name"]}')
    flat = gradient.reshape(-1)
    tile, width, workers = spec['tile'], spec['simd_width'], spec['workers']
    partials = []
    # Bound temporary memory; each tile's arithmetic is independent.
    for start in range(0, flat.numel(), tile * 32768):
        values = flat[start:start + tile * 32768] * 1.0
        squared = values * values
        count = squared.numel() // tile
        if count:
            partials.append(sumx(squared[:count * tile].reshape(count, tile), width))
        if squared.numel() % tile:
            partials.append(sumx(squared[count * tile:].reshape(1, -1), width))
    pieces = torch.cat(partials)
    full = pieces.numel() // workers
    result = pieces[:workers].clone()
    for index in range(1, full):
        result = result + pieces[index * workers:(index + 1) * workers]
    tail = pieces.numel() % workers
    if tail:
        result[:tail] = result[:tail] + pieces[full * workers:]
    return sequential(result)


def native_recipes(named_gradients: dict) -> dict:
    """Map live JT gradients to native layouts without retaining packed copies."""
    recipes = {}
    for name, gradient in named_gradients.items():
        origins, transform = mapping_for(name)
        origins = [origin.removeprefix('model.') for origin in origins]
        if transform in ('even', 'odd'):
            recipes.setdefault(origins[0], {})[transform] = gradient
        elif transform == 'concat':
            recipes[origins[0]] = ('identity', gradient[:1536])
            recipes[origins[1]] = ('identity', gradient[1536:])
        else:
            recipes[origins[0]] = (transform, gradient)
    expected = {spec['name'] for spec in NATIVE_SCHEDULE}
    if set(recipes) != expected:
        raise ValueError(f'Native norm parameter mismatch: {set(recipes) ^ expected}')
    return recipes


def materialize(recipe, shape: list) -> torch.Tensor:
    """Undo HP-only packing while keeping mathematical gradient values intact."""
    if isinstance(recipe, dict):
        if set(recipe) != {'even', 'odd'}:
            raise ValueError('Missing gate/up projection for native norm')
        return torch.stack((recipe['even'], recipe['odd']), dim=1).reshape(shape)
    transform, gradient = recipe
    if transform == 'expert_transpose':
        return gradient.transpose(1, 2).contiguous().reshape(shape)
    if transform != 'identity':
        raise ValueError(f'Unexpected native norm transform: {transform}')
    return gradient.reshape(shape)


def norm_from_gradients(named_gradients: dict) -> torch.Tensor:
    """Compute native per-parameter square sums from the current HP backward pass."""
    recipes = native_recipes(named_gradients)
    sums = []
    for spec in NATIVE_SCHEDULE:
        gradient = materialize(recipes[spec['name']], spec['shape'])
        sums.append(square_sum(gradient, spec))
        del gradient
    return sequential(torch.stack(sums)).sqrt()


@torch.no_grad()
def clip_grad_norm_dvm(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None, *, observer=None):
    """Apply the existing DVM schedule; optionally observe its actual norm/coefficient."""
    del foreach
    if float(norm_type) != 2.0:
        raise ValueError('Measured MF clipping requires L2 norm')
    named = {}
    for name, parameter in parameters.named_parameters():
        gradient = _get_local_grad(parameter)
        if gradient is not None:
            key = getattr(parameter, 'model_name', name).replace('_checkpoint_wrapped_module.', '')
            named[key] = gradient
    total = norm_from_gradients(named)
    if error_if_nonfinite and not bool(torch.isfinite(total)):
        raise RuntimeError('Nonfinite MF DVM norm')
    coefficient = (total.new_tensor(float(max_norm)) / (total + total.new_tensor(1.0e-6))).clamp(max=1.0)
    for gradient in named.values():
        gradient.mul_(coefficient)
    if observer is not None:
        observer(total, coefficient)
    return total
