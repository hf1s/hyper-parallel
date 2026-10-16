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

"""Compare an observed archived DVM reduction with NPU normalization, without updating a model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch_npu


_SHAPE = (5120, 3072)
_TILE_ELEMENTS = 9776
_LANES = 16
_VECTOR_CORES = 40
_EPSILON = 1.0011717677116394e-7


def _half_tree(values: torch.Tensor) -> torch.Tensor:
    work = values
    while work.shape[-1] > 1:
        size = work.shape[-1]
        half = size // 2
        reduced = work[..., :half] + work[..., half:2 * half]
        if size % 2:
            reduced[..., 0] = reduced[..., 0] + work[..., -1]
        work = reduced
    return work[..., 0]


def _archived_normalization(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    current = value.mT
    tile_count = (current.numel() + _TILE_ELEMENTS - 1) // _TILE_ELEMENTS
    padded = current.new_zeros(tile_count * _TILE_ELEMENTS)
    padded[:current.numel()] = current.reshape(-1).square()
    lanes = _half_tree(padded.reshape(tile_count, _TILE_ELEMENTS // _LANES, _LANES).transpose(-1, -2))
    while lanes.shape[-1] > 1:
        lanes = lanes[..., ::2] + lanes[..., 1::2]
    partials = lanes[..., 0].cpu().numpy()
    accumulators = np.zeros(_VECTOR_CORES, dtype=np.float32)
    for index, partial in enumerate(partials):
        core = index % _VECTOR_CORES
        accumulators[core] = np.float32(accumulators[core] + partial)
    total = np.float32(0)
    for partial in accumulators:
        total = np.float32(total + partial)
    norm = torch.tensor(total, device=value.device).sqrt()
    return (current / (norm + _EPSILON)).bfloat16(), norm, float(total)


def _tensor_hash(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy()
    return hashlib.sha256(raw.tobytes()).hexdigest()


def _precision_report(current: np.ndarray, ordinary: torch.Tensor, archived: torch.Tensor) -> dict:
    reference = current.astype(np.float64)
    norm64 = float(np.sqrt(np.sum(np.square(reference))))
    reference /= norm64 + _EPSILON
    ordinary_array = ordinary.float().cpu().numpy()
    archived_array = archived.float().cpu().numpy()
    different = ordinary_array != archived_array
    ordinary_distance = np.abs(ordinary_array.astype(np.float64) - reference)
    archived_distance = np.abs(archived_array.astype(np.float64) - reference)
    return {
        "norm_fp64": norm64,
        "bf16_different_elements": int(np.count_nonzero(different)),
        "ordinary_closer_to_fp64": int(np.count_nonzero(different & (ordinary_distance < archived_distance))),
        "archived_closer_to_fp64": int(np.count_nonzero(different & (archived_distance < ordinary_distance))),
        "equal_distance": int(np.count_nonzero(different & (archived_distance == ordinary_distance))),
    }


def compare_normalization(array: np.ndarray, device: int = 0) -> dict:
    """Compare the measured shared-down schedule using the caller's actual NS input.

    Args:
        array: Finite FP32 array of shape (5120, 3072), after any input rounding.
        device: Visible NPU index. The caller must reserve this device.

    Returns:
        Scalar and full-tensor hash evidence, including FP64 distance comparisons.

    Raises:
        ValueError: If the input is outside the measured shape/dtype contract.

    Note:
        This diagnostic performs device/host synchronization and changes the
        process compile mode temporarily. Run it in a separate process from training.
        Its schedule describes one observed archived kernel, not a general norm API.
    """
    if array.dtype != np.float32 or array.shape != _SHAPE or not np.isfinite(array).all():
        raise ValueError("Expected finite FP32 NS input with shape (5120, 3072)")
    torch.npu.set_device(device)
    previous = torch_npu.npu.is_jit_compile_false()
    torch_npu.npu.set_compile_mode(jit_compile=False)
    try:
        with torch.no_grad():
            value = torch.from_numpy(np.ascontiguousarray(array)).npu()
            current = value.mT
            ordinary_norm = current.norm(dim=(-2, -1), keepdim=True)
            ordinary = (current / (ordinary_norm + _EPSILON)).bfloat16()
            archived, archived_norm, square_sum = _archived_normalization(value)
            return {
                "shape": list(array.shape),
                "input_fp32_sha256": _tensor_hash(current),
                "ordinary_bf16_sha256": _tensor_hash(ordinary),
                "archived_bf16_sha256": _tensor_hash(archived),
                "ordinary_norm": float(ordinary_norm.cpu().reshape(-1)[0]),
                "archived_norm": float(archived_norm.cpu()),
                "archived_square_sum": square_sum,
                "epsilon": _EPSILON,
                "precision": _precision_report(array.T, ordinary, archived),
                "model_updated": False,
                "reference_values_injected": False,
                "schedule_scope": "Observed 910B3 / MindSpore 2.7.2 / CANN 8.5.0 shared-down DVM kernel",
            }
    finally:
        torch_npu.npu.set_compile_mode(jit_compile=not previous)


def main() -> None:
    """Read one caller-owned NPY input and write a new bounded JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("The report destination already exists")
    array = np.load(args.input, allow_pickle=False)
    report = compare_normalization(array, args.device)
    report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
