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
"""Convert one MF safetensors checkpoint into HP model/optimizer state artifacts.

The converter preserves the existing MF->HP model layout conversion and adds:
  - optimizer.npz: adam_m, adam_v, and muon_m states in HP parameter names;
  - state.json: step/epoch/global-batch/loss-scale and derived data cursor;
  - CONVERSION.json: source/output key counts and conversion metadata.

It does not calculate hashes or checksums. The offline artifacts are consumed
by a separate HP restore adapter; model.npz alone remains the weight-only path.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any


import numpy as np


STATE_PREFIXES = ("adam_m", "adam_v", "muon_m")


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Read a safetensors file into NumPy arrays.

    Args:
        path: Safetensors file to read.

    Returns:
        Mapping from tensor name to an independent NumPy array.
    """

    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        blob = handle.read()
    dtypes = {
        "F32": np.float32,
        "F16": np.float16,
        "I64": np.int64,
        "I32": np.int32,
    }
    out: dict[str, np.ndarray] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = dtypes.get(info["dtype"])
        if dtype is None:
            raise SystemExit(f"unsupported dtype {info['dtype']} for {name}")
        start, end = info["data_offsets"]
        out[name] = np.frombuffer(blob[start:end], dtype=dtype).reshape(info["shape"]).copy()
    return out


def deinterleave_fused(fused: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split interleaved [g0, u0, g1, u1, ...] rows into gate/up.

    Args:
        fused: Interleaved fused gate/up tensor.

    Returns:
        Separate gate and up tensors.
    """

    if fused.size == 1:
        # MF stores a one-element placeholder for states owned by the other
        # optimizer family; preserve one placeholder for each split parameter.
        return fused.copy(), fused.copy()
    return fused[0::2].copy(), fused[1::2].copy()


def convert_layer(
    src: dict[str, np.ndarray],
    src_prefix: str,
    dst_prefix: str,
    *,
    is_moe: bool,
) -> dict[str, np.ndarray]:
    """Convert one MF layer, including model or optimizer tensors.

    Args:
        src: Source tensors keyed by MF names.
        src_prefix: MF layer name prefix.
        dst_prefix: HP layer name prefix.
        is_moe: Whether the layer uses the MoE layout.

    Returns:
        Converted tensors keyed by HP names.
    """


    out: dict[str, np.ndarray] = {}

    def put(dst: str, source: str) -> None:
        """Copy one source tensor when it is present.

        Args:
            dst: Destination HP tensor name.
            source: Source MF tensor name.
        """
        if source in src:
            out[dst] = src[source]

    attn_s, attn_d = f"{src_prefix}.self_attention", f"{dst_prefix}.self_attn"
    put(f"{attn_d}.q_a_layernorm.weight", f"{attn_s}.q_layernorm.weight")
    put(f"{attn_d}.kv_a_layernorm.weight", f"{attn_s}.kv_layernorm.weight")
    put(f"{attn_d}.q_b_proj.weight", f"{attn_s}.linear_q_up_proj.weight")
    put(f"{attn_d}.kv_b_proj.weight", f"{attn_s}.linear_kv_up_proj.weight")
    put(f"{attn_d}.o_proj.weight", f"{attn_s}.linear_proj.weight")
    q_down = f"{attn_s}.linear_q_down_proj.weight"
    kv_down = f"{attn_s}.linear_kv_down_proj.weight"
    if q_down in src and kv_down in src:
        out[f"{attn_d}.linear_qkv.weight"] = np.concatenate([src[q_down], src[kv_down]], axis=0)

    put(f"{dst_prefix}.input_layernorm.weight", f"{src_prefix}.input_layernorm.weight")
    put(f"{dst_prefix}.post_attention_layernorm.weight", f"{src_prefix}.pre_mlp_layernorm.weight")

    mlp_s, mlp_d = f"{src_prefix}.mlp", f"{dst_prefix}.mlp"
    if not is_moe:
        fc1 = f"{mlp_s}.linear_fc1.weight"
        if fc1 in src:
            gate, up = deinterleave_fused(src[fc1])
            out[f"{mlp_d}.gate_proj.weight"] = gate
            out[f"{mlp_d}.up_proj.weight"] = up
        put(f"{mlp_d}.down_proj.weight", f"{mlp_s}.linear_fc2.weight")
        return out

    put(f"{mlp_d}.gate.weight", f"{mlp_s}.router.weight")
    put(f"{mlp_d}.gate.e_score_correction_bias", f"{mlp_s}.router.expert_bias")
    w1, w2 = f"{mlp_s}.experts.weight1", f"{mlp_s}.experts.weight2"
    if w1 in src and w2 in src:
        if src[w1].size == 1 or src[w2].size == 1:
            out[f"{mlp_d}.experts.gate_up_proj"] = src[w1].copy()
            out[f"{mlp_d}.experts.down_proj"] = src[w2].copy()
        else:
            num_experts = src[f"{mlp_s}.router.weight"].shape[0]
            hidden = src[w2].shape[-1]
            moe_i = src[w2].shape[0] // num_experts
            out[f"{mlp_d}.experts.gate_up_proj"] = (
                src[w1].reshape(num_experts, hidden, 2 * moe_i)
                .transpose(0, 2, 1).copy()
            )
            out[f"{mlp_d}.experts.down_proj"] = (
                src[w2].reshape(num_experts, moe_i, hidden)
                .transpose(0, 2, 1).copy()
            )

    shared_fc1 = f"{mlp_s}.shared_experts.linear_fc1.weight"
    if shared_fc1 in src:
        gate, up = deinterleave_fused(src[shared_fc1])
        out[f"{mlp_d}.shared_experts.gate_proj.weight"] = gate
        out[f"{mlp_d}.shared_experts.up_proj.weight"] = up
    put(f"{mlp_d}.shared_experts.down_proj.weight", f"{mlp_s}.shared_experts.linear_fc2.weight")
    return out


def convert_model(src: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert MF model tensors to HP parameter names.

    Args:
        src: Source tensors keyed by MF names.

    Returns:
        Converted tensors keyed by HP names.
    """

    out: dict[str, np.ndarray] = {}
    direct = {
        "model.embed_tokens.weight": "embedding.word_embeddings.weight",
        "lm_head.weight": "output_layer.weight",
        "model.norm.weight": "decoder.final_layernorm.weight",
    }
    for dst, source in direct.items():
        if source in src:
            out[dst] = src[source]
    for index in (0, 1):
        prefix = f"decoder.layers.{index}"
        if any(key.startswith(prefix + ".") for key in src):
            out.update(convert_layer(src, prefix, f"model.layers.{index}", is_moe=index != 0))
    mtp = "mtp.layers.0"
    for name in ("eh_proj", "enorm", "hnorm", "final_layernorm"):
        source = f"{mtp}.{name}.weight"
        if source in src:
            out[f"{mtp}.{name}.weight"] = src[source]
    if any(key.startswith(f"{mtp}.transformer_layer.") for key in src):
        out.update(convert_layer(src, f"{mtp}.transformer_layer", f"{mtp}.transformer_layer", is_moe=True))
    return out


def convert_optimizer(src: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert MF optimizer tensors while preserving family prefixes.

    Args:
        src: Source tensors keyed by MF names and optimizer prefixes.

    Returns:
        Converted optimizer tensors keyed by HP names and family prefixes.
    """

    out: dict[str, np.ndarray] = {}
    for prefix in STATE_PREFIXES:
        source = {
            key[len(prefix) + 1:]: value
            for key, value in src.items()
            if key.startswith(prefix + ".")
        }
        if not source:
            continue
        converted = convert_model(source)
        out.update({f"{prefix}.{name}": value for name, value in converted.items()})
    return out


def scalar(src: dict[str, np.ndarray], name: str, default: Any = None) -> Any:
    """Read a scalar metadata tensor.

    Args:
        src: Source tensors keyed by MF names.
        name: Metadata tensor name.
        default: Value returned when the metadata tensor is absent.

    Returns:
        Python scalar value or ``default``.
    """

    value = src.get(name)
    return default if value is None else value.reshape(()).item()



def write_artifacts(
    src: dict[str, np.ndarray],
    model: dict[str, np.ndarray],
    optimizer: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    """Write the converted model, optimizer and resume metadata.

    Args:
        src: Original source tensors used for resume metadata.
        model: Converted HP model tensors.
        optimizer: Converted HP optimizer tensors.
        out_dir: Destination directory for the offline artifacts.
    """

    np.savez(out_dir / "model.npz", **model)
    np.savez(out_dir / "optimizer.npz", **optimizer)
    step = int(scalar(src, "step_num", 0))
    global_batch = int(scalar(src, "global_batch_size", 1))
    state = {
        "global_step": step,
        "step_num": step,
        "epoch_num": int(scalar(src, "epoch_num", step)),
        "global_batch_size": global_batch,
        "consumed_samples": step * global_batch,
        "loss_scale": float(scalar(src, "loss_scale", 1.0)),
        "data_order": "sequential",
        "source_checkpoint": "CKP_rank_0-300_1.safetensors",
    }
    (out_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    manifest = {
        "format": "mf-to-hp-offline-state-v1",
        "model_tensor_count": len(model),
        "optimizer_tensor_count": len(optimizer),
        "source_tensor_count": len(src),
        "optimizer_prefixes": sorted({name.split(".", 1)[0] for name in optimizer}),
        "state_file": "state.json",
        "model_file": "model.npz",
        "optimizer_file": "optimizer.npz",
        "note": "No hashes/checksums are calculated by this converter.",
    }
    (out_dir / "CONVERSION.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    """Convert one source checkpoint from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out_dir is None and args.out is None:
        parser.error("one of --out-dir or --out is required")

    src = load_safetensors(args.ckpt)
    model = convert_model(src)
    optimizer = convert_optimizer(src)
    out_dir = args.out_dir or args.out.parent
    write_artifacts(src, model, optimizer, out_dir)
    if args.out is not None and args.out != out_dir / "model.npz":
        args.out.write_bytes((out_dir / "model.npz").read_bytes())
    print(f"source tensors: {len(src)}")
    print(f"model tensors: {len(model)}")
    print(f"optimizer tensors: {len(optimizer)}")
    print(f"step: {scalar(src, 'step_num', 0)} / global_batch_size: {scalar(src, 'global_batch_size', 1)}")
    print(f"written: {out_dir}")


if __name__ == "__main__":
    main()
