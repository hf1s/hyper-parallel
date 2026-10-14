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
"""Recompute the kv projection with alternate GEMM shapes and compare bitwise with MF.

The variants run once, inside a read-only pre-hook of the attention's ``linear_qkv``, on the
very input and weight that projection is about to use. Nothing replaces the model's own
computation, so the step result is unchanged; only small JSON reports are written.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from hyper_parallel.core.optimizer.dtensor_compat import to_local_if_dtensor

TARGET = 'model.layers.0.self_attn.linear_qkv'
SOURCE_FACTS = {
    'f_runtime': '/alignment/tools/precision_probe.py:487-493 SeparateLatentProjection.forward does '
                 'query = F.linear(values, weight[:query_rank]) and '
                 'key_value = F.linear(values, weight[query_rank:]), then cat -> TWO matmuls, '
                 'widths 1536 and 576 (kv 512 + rope 64); the kv raw is the first 512 columns of '
                 'the 576-wide result',
    'f_split': '/workspace/worktree/align-l2-f/hyper_parallel/models/jt_deepseek_v3/'
               'modeling_jt_deepseek_v3.py:193-197 split latent_states into query_local / kv_local / '
               'rope_local',
    'class_swap': '/alignment/tools/precision_probe.py:725-727 (qkv-split factor swaps linear_qkv '
                  'to SeparateLatentProjection and sets query_rank = q_lora_rank)',
    'mf_side': 'MindFormers has a dedicated linear_kv_down_proj (512-wide) matmul',
}


def _bits(tensor: torch.Tensor) -> np.ndarray:
    """Return the raw 16-bit patterns of a bf16 tensor as int16."""
    value = to_local_if_dtensor(tensor).detach().cpu().contiguous()
    return value.view(torch.int16).numpy()


def _digest_bits(tensor: torch.Tensor) -> str:
    """Hash the raw 16-bit patterns of a bf16 tensor."""
    return hashlib.sha256(np.ascontiguousarray(_bits(tensor)).tobytes()).hexdigest()


def _ulp(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the ULP distance between two int16 bit-pattern arrays."""
    first = left.astype(np.int32)
    second = right.astype(np.int32)
    first = np.where(first < 0, -32768 - first, first)
    second = np.where(second < 0, -32768 - second, second)
    return np.abs(first - second)


def _compare(name, tensor, mf_bits) -> dict:
    """Compare one variant's kv block with MF bit by bit."""
    bits = _bits(tensor)
    print(f'[GEMM-VARIANT] {name}: shape={tuple(bits.shape)} numel={bits.size} '
          f'mf_numel={mf_bits.size}', flush=True)
    if bits.size != mf_bits.size:
        record = {'variant': name, 'shape': list(bits.shape), 'element_count': int(bits.size),
                  'error': f'element count {bits.size} differs from MF {mf_bits.size}'}
        print(f'[GEMM-VARIANT] {name}: ERROR {record["error"]}', flush=True)
        return record
    value = bits.reshape(mf_bits.shape)
    distance = _ulp(value, mf_bits)
    histogram = {str(ulp): int((distance == ulp).sum()) for ulp in np.unique(distance) if ulp}
    record = {
        'variant': name,
        'shape': list(value.shape),
        'different_elements': int((distance != 0).sum()),
        'max_ulp': int(distance.max()),
        'ulp_histogram': histogram,
        'first_difference_index': (int(np.argmax(distance != 0)) if (distance != 0).any() else None),
        'bitwise_identical_to_mf': bool(not (distance != 0).any()),
        'sha256_of_variant_bits': hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
    }
    print(f"[GEMM-VARIANT] {name}: different={record['different_elements']} "
          f"max_ulp={record['max_ulp']} first={record['first_difference_index']} "
          f"identical={record['bitwise_identical_to_mf']} hist={histogram}", flush=True)
    return record


def attach(trainer, output: str | Path, mf_directory: str | Path, module_name: str = TARGET) -> dict:
    """Register the one-shot variant experiment on the attention projection.

    Args:
        trainer: Built HP trainer holding the model.
        output: Directory receiving the small JSON report.
        mf_directory: Directory holding the MF ``p2.bits.npy`` / ``p2.json`` truth.
        module_name: Fully qualified name of the fused qkv projection.

    Returns:
        The mutable experiment state.
    """
    output, mf_directory = Path(output), Path(mf_directory)
    output.mkdir(parents=True, exist_ok=True)
    module = dict(trainer.base.model.named_modules())[module_name]
    state = {'done': False, 'report': None}

    def run(module, inputs) -> None:
        """Compute every GEMM variant once, on the module's own input and weight."""
        values = to_local_if_dtensor(inputs[0]).detach()
        print(f'[GEMM-VARIANT] call values={tuple(values.shape)} {values.dtype} '
              f'done={state["done"]}', flush=True)
        if state['done'] or values.dim() != 3 or values.shape[1] != 4096:
            return
        state['done'] = True
        weight = to_local_if_dtensor(module.weight).detach()
        query_rank = int(module.query_rank)
        kv_rank = int(getattr(module, 'kv_lora_rank', 0)) or 512
        rope_rank = 64
        mf = np.load(mf_directory / 'p2.bits.npy').astype(np.int16).reshape(-1, kv_rank)
        print(f'[GEMM-VARIANT] values={tuple(values.shape)} {values.dtype} '
              f'weight={tuple(weight.shape)} {weight.dtype} query_rank={query_rank} '
              f'kv_rank={kv_rank} mf={mf.shape}', flush=True)
        transposed = weight.t().contiguous().t()
        with torch.no_grad():
            kv_view = weight[query_rank:]
            kv_only = weight[query_rank:query_rank + kv_rank]
            transposed_copy = kv_view.contiguous()
            kv_transposed = kv_only.t().contiguous()
            kv_view_transposed = kv_view.t().contiguous()
            # MF frame: the native kernels see (seq, batch, hidden), not a flattened 2-D input.
            x_mf = values.transpose(0, 1).contiguous()
            x_view = values.transpose(0, 1)
            weight_t = weight.t()
            weight_c = weight_t.contiguous()
            latent = weight[query_rank:query_rank + kv_rank]
            rope = weight[query_rank + kv_rank:]
            flat = values.reshape(-1, values.shape[-1])
            factories = [
                ('v0_current_576w_weight_view',
                 lambda: functional.linear(values, kv_view)[..., :kv_rank]),
                ('v1_576w_weight_contiguous_copy',
                 lambda: functional.linear(values, transposed_copy)[..., :kv_rank]),
                ('v2_576w_weight_strided_view',
                 lambda: functional.linear(values, transposed[query_rank:])[..., :kv_rank]),
                ('v3_2112w_single_mm_then_slice',
                 lambda: functional.linear(values, weight)[..., query_rank:query_rank + kv_rank]),
                ('v4_512w_weight_view', lambda: functional.linear(values, kv_only)),
                ('v5_512w_weight_contiguous_copy',
                 lambda: functional.linear(values, latent.contiguous())),
                ('v6_512w_matmul_contiguous_transpose',
                 lambda: torch.matmul(values, kv_transposed)),
                ('v7_512w_2d_mm_contiguous_transpose',
                 lambda: torch.mm(flat, kv_transposed).reshape(values.shape[0], values.shape[1],
                                                               kv_rank)),
                ('v8_576w_2d_mm_contiguous_transpose',
                 lambda: torch.mm(flat, kv_view_transposed).reshape(
                     values.shape[0], values.shape[1], kv_view.shape[0])[..., :kv_rank]),
                ('v10_3d_matmul_view_weight',
                 lambda: torch.matmul(x_mf, weight_t)[..., query_rank:query_rank + kv_rank]),
                ('v11_bmm_batch_one_hp_frame',
                 lambda: torch.bmm(values, weight_t.unsqueeze(0))[
                     ..., query_rank:query_rank + kv_rank]),
                ('v12_3d_matmul_contiguous_transpose',
                 lambda: torch.matmul(x_mf, weight_c)[..., query_rank:query_rank + kv_rank]),
                ('v13_3d_linear_3d_input',
                 lambda: functional.linear(x_mf, weight)[..., query_rank:query_rank + kv_rank]),
                ('v14_3d_matmul_noncontiguous_input',
                 lambda: torch.matmul(x_view, weight_t)[..., query_rank:query_rank + kv_rank]),
                ('v15_3d_two_mm_latent_only', lambda: torch.matmul(x_mf, latent.t())),
                ('v16_3d_two_mm_rope_only', lambda: torch.matmul(x_mf, rope.t())),
            ]
            records = []
            for name, factory in factories:
                try:
                    tensor = factory()
                except Exception as error:  # pylint: disable=broad-except
                    records.append({'variant': name, 'error': f'{type(error).__name__}: {error}'})
                    print(f'[GEMM-VARIANT] {name}: ERROR {type(error).__name__}: {error}', flush=True)
                    continue
                records.append(_compare(name, tensor, mf))
                del tensor
        state['report'] = {
            'module': module_name,
            'input': {'dtype': str(values.dtype), 'shape': list(values.shape),
                      'sha256': hashlib.sha256(
                          to_local_if_dtensor(values).detach().cpu().contiguous()
                          .view(torch.int16).numpy().tobytes()).hexdigest()},
            'weight': {'dtype': str(weight.dtype), 'shape': list(weight.shape),
                       'query_rank': query_rank, 'kv_rank': kv_rank, 'rope_rank': rope_rank,
                       'kv_weight_view_contiguous': bool(kv_view.is_contiguous()),
                       'strided_variant_strides': list(transposed[query_rank:].stride()),
                       'q_1536_sha256': _digest_bits(weight[:query_rank]),
                       'kv_576_sha256': _digest_bits(kv_view),
                       'kv_512_sha256': _digest_bits(kv_only),
                       'rope_64_sha256': _digest_bits(weight[query_rank + kv_rank:])},
            'mf_reference': str(mf_directory / 'p2.bits.npy'),
            'frames': {'hp_input_shape': list(values.shape),
                       'hp_input_strides': list(values.stride()),
                       'hp_input_contiguous': bool(values.is_contiguous()),
                       'mf_frame_shape': list(x_mf.shape),
                       'mf_frame_strides': list(x_mf.stride()),
                       'mf_frame_contiguous': bool(x_mf.is_contiguous()),
                       'transposed_view_strides': list(x_view.stride()),
                       'weight_transposed_contiguous': bool(weight_t.is_contiguous())},
            'variants': records,
            'bitwise_identical_variants': [row['variant'] for row in records
                                           if row.get('bitwise_identical_to_mf')],
            'source_facts': SOURCE_FACTS,
        }
        (output / 'gemm_variants_step301.json').write_text(json.dumps(state['report'], indent=2) + '\n')
        print('[GEMM-VARIANT] identical variants: '
              f"{state['report']['bitwise_identical_variants']}", flush=True)

    module.register_forward_pre_hook(run)
    print(f'[GEMM-VARIANT] registered on {module_name}', flush=True)
    return state
