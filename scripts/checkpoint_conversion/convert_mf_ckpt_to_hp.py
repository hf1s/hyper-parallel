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

if __package__:
    from .native_input import normalize_native_checkpoint
else:
    from native_input import normalize_native_checkpoint


STATE_PREFIXES = ("adam_m", "adam_v", "muon_m")
_METADATA_KEYS = {"step_num", "epoch_num", "global_batch_size", "loss_scale"}


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


def load_checkpoint_input(
    path: Path, *, source_format: str = "auto",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read family-prefix tensors or the fixed MF callback's native schema.

    Args:
        path: Input safetensors checkpoint.
        source_format: ``auto``, ``family-prefix`` or ``mf-native``. Auto selects
            native validation only for a MindSpore producer header.

    Returns:
        Validated source tensor mapping and producer/runtime-field information.

    Raises:
        ValueError: The requested producer or native checkpoint state is unsupported.
    """
    if source_format not in ("auto", "family-prefix", "mf-native"):
        raise ValueError("source_format must be auto, family-prefix or mf-native")
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        metadata = json.loads(handle.read(header_size)).get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError("Safetensors producer metadata must be an object")
    native = source_format == "mf-native" or (source_format == "auto" and metadata.get("format") == "ms")
    if native and (metadata.get("format") != "ms" or metadata.get("remove_redundancy") not in (None, "False")):
        raise ValueError("mf-native requires a non-redundancy-removed MindSpore producer checkpoint")
    src = load_safetensors(path)
    if native:
        return normalize_native_checkpoint(src)
    return src, {"source_format": "family-prefix", "source_tensor_count": len(src)}


def deinterleave_fused(fused: np.ndarray, *, is_placeholder: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Split interleaved [g0, u0, g1, u1, ...] rows into gate/up.

    Args:
        fused: Interleaved fused gate/up tensor.
        is_placeholder: Whether this is an inactive optimizer-family slot.

    Returns:
        Separate gate and up tensors.
    """

    if is_placeholder:
        return fused.copy(), fused.copy()
    if fused.ndim != 2 or fused.shape[0] % 2 != 0:
        raise ValueError("Interleaved gate/up projection must be a matrix with an even number of rows")
    return fused[0::2].copy(), fused[1::2].copy()


def _convert_layer(
    src: dict[str, np.ndarray],
    src_prefix: str,
    dst_prefix: str,
    *,
    is_moe: bool,
    consumed: set[str],
    mlp_layout: str = "separate",
    layout_source: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Convert one MF layer, including model or optimizer tensors.

    Args:
        src: Source tensors keyed by MF names.
        src_prefix: MF layer name prefix.
        dst_prefix: HP layer name prefix.
        is_moe: Whether the layer uses the MoE layout.
        consumed: Source tensor names handled by this conversion.
        mlp_layout: Separate projections or the public SwiGLU module layout.
        layout_source: Model tensors used to identify optimizer placeholder shapes.

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
            consumed.add(source)

    def _placeholder(key: str) -> bool:
        return (layout_source is not None and src[key].size == 1
                and src[key].shape != layout_source[key].shape)

    def _concatenate(first: str, second: str) -> np.ndarray:
        consumed.update((first, second))
        if _placeholder(first) != _placeholder(second):
            raise ValueError(f"Incompatible optimizer families in fused parameter: {first}, {second}")
        return src[first].copy() if _placeholder(first) else np.concatenate([src[first], src[second]], axis=0)

    attn_s, attn_d = f"{src_prefix}.self_attention", f"{dst_prefix}.self_attn"
    put(f"{attn_d}.q_a_layernorm.weight", f"{attn_s}.q_layernorm.weight")
    put(f"{attn_d}.kv_a_layernorm.weight", f"{attn_s}.kv_layernorm.weight")
    put(f"{attn_d}.q_b_proj.weight", f"{attn_s}.linear_q_up_proj.weight")
    put(f"{attn_d}.kv_b_proj.weight", f"{attn_s}.linear_kv_up_proj.weight")
    put(f"{attn_d}.o_proj.weight", f"{attn_s}.linear_proj.weight")
    q_down = f"{attn_s}.linear_q_down_proj.weight"
    kv_down = f"{attn_s}.linear_kv_down_proj.weight"
    if q_down in src and kv_down in src:
        out[f"{attn_d}.linear_qkv.weight"] = _concatenate(q_down, kv_down)

    put(f"{dst_prefix}.input_layernorm.weight", f"{src_prefix}.input_layernorm.weight")
    put(f"{dst_prefix}.post_attention_layernorm.weight", f"{src_prefix}.pre_mlp_layernorm.weight")

    mlp_s, mlp_d = f"{src_prefix}.mlp", f"{dst_prefix}.mlp"
    if not is_moe:
        fc1 = f"{mlp_s}.linear_fc1.weight"
        if fc1 in src:
            consumed.add(fc1)
            gate, up = deinterleave_fused(src[fc1], is_placeholder=_placeholder(fc1))
            if mlp_layout == "swiglu":
                out[f"{mlp_d}.linear_fc1.weight"] = src[fc1].copy() if _placeholder(fc1) else np.concatenate([gate, up])
            else:
                out[f"{mlp_d}.gate_proj.weight"] = gate
                out[f"{mlp_d}.up_proj.weight"] = up
        down_name = "linear_fc2" if mlp_layout == "swiglu" else "down_proj"
        put(f"{mlp_d}.{down_name}.weight", f"{mlp_s}.linear_fc2.weight")
        return out

    put(f"{mlp_d}.gate.weight", f"{mlp_s}.router.weight")
    put(f"{mlp_d}.gate.e_score_correction_bias", f"{mlp_s}.router.expert_bias")
    w1, w2 = f"{mlp_s}.experts.weight1", f"{mlp_s}.experts.weight2"
    if w1 in src and w2 in src:
        consumed.update((w1, w2))
        if _placeholder(w1) != _placeholder(w2):
            raise ValueError(f"Incompatible optimizer families in expert projections: {mlp_s}")
        if _placeholder(w1):
            out[f"{mlp_d}.experts.gate_up_proj"] = src[w1].copy()
            out[f"{mlp_d}.experts.down_proj"] = src[w2].copy()
        else:
            reference = src if layout_source is None else layout_source
            num_experts = reference[f"{mlp_s}.router.weight"].shape[0]
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
        consumed.add(shared_fc1)
        gate, up = deinterleave_fused(src[shared_fc1], is_placeholder=_placeholder(shared_fc1))
        if mlp_layout == "swiglu":
            out[f"{mlp_d}.shared_experts.linear_fc1.weight"] = (
                src[shared_fc1].copy() if _placeholder(shared_fc1) else np.concatenate([gate, up])
            )
        else:
            out[f"{mlp_d}.shared_experts.gate_proj.weight"] = gate
            out[f"{mlp_d}.shared_experts.up_proj.weight"] = up
    down_name = "linear_fc2" if mlp_layout == "swiglu" else "down_proj"
    put(f"{mlp_d}.shared_experts.{down_name}.weight", f"{mlp_s}.shared_experts.linear_fc2.weight")
    return out


def convert_model(
        src: dict[str, np.ndarray],
        *,
        mlp_layout: str = "separate",
        layout_source: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Convert MF model tensors to HP parameter names.

    Args:
        src: Source tensors keyed by MF names.
        mlp_layout: Separate projections or the public SwiGLU module layout.
        layout_source: Full model tensors when converting optimizer slots.

    Returns:
        Converted tensors keyed by HP names.
    """

    if mlp_layout not in ("separate", "swiglu"):
        raise ValueError("mlp_layout must be separate or swiglu")
    out: dict[str, np.ndarray] = {}
    consumed: set[str] = set()
    direct = {
        "model.embed_tokens.weight": "embedding.word_embeddings.weight",
        "lm_head.weight": "output_layer.weight",
        "model.norm.weight": "decoder.final_layernorm.weight",
    }
    for dst, source in direct.items():
        if source in src:
            out[dst] = src[source]
            consumed.add(source)
    for index in (0, 1):
        prefix = f"decoder.layers.{index}"
        if any(key.startswith(prefix + ".") for key in src):
            out.update(_convert_layer(src, prefix, f"model.layers.{index}", is_moe=index != 0, consumed=consumed,
                                     mlp_layout=mlp_layout, layout_source=layout_source))
    mtp = "mtp.layers.0"
    for name in ("eh_proj", "enorm", "hnorm", "final_layernorm"):
        source = f"{mtp}.{name}.weight"
        if source in src:
            out[f"{mtp}.{name}.weight"] = src[source]
            consumed.add(source)
    if any(key.startswith(f"{mtp}.transformer_layer.") for key in src):
        out.update(_convert_layer(src, f"{mtp}.transformer_layer", f"{mtp}.transformer_layer",
                                 is_moe=True, consumed=consumed,
                                 mlp_layout=mlp_layout, layout_source=layout_source))
    unconsumed = {key for key in src if key not in _METADATA_KEYS
                  and key.split(".", 1)[0] not in STATE_PREFIXES} - consumed
    if unconsumed:
        raise ValueError(f"Unsupported or incomplete source mapping: {sorted(unconsumed)}")
    return out


def convert_optimizer(src: dict[str, np.ndarray], *, mlp_layout: str = "separate") -> dict[str, np.ndarray]:
    """Convert MF optimizer tensors while preserving family prefixes.

    Args:
        src: Source tensors keyed by MF names and optimizer prefixes.
        mlp_layout: Separate projections or the public SwiGLU module layout.

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
        for name, value in source.items():
            if name not in src:
                raise ValueError(f"Optimizer tensor has no source model tensor: {prefix}.{name}")
            if value.size != 1 and value.shape != src[name].shape:
                raise ValueError(f"Optimizer source shape differs from model tensor: {prefix}.{name}")
        converted = convert_model(source, mlp_layout=mlp_layout, layout_source=src)
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
    *,
    source_checkpoint: Path,
    input_report: dict[str, Any] | None = None,
) -> None:
    """Write the converted model, optimizer and resume metadata.

    Args:
        src: Original source tensors used for resume metadata.
        model: Converted HP model tensors.
        optimizer: Converted HP optimizer tensors.
        out_dir: Destination directory for the offline artifacts.
        source_checkpoint: Actual input checkpoint path recorded for provenance.
        input_report: Optional producer and validated non-model-field information.
    """

    step = scalar(src, "step_num")
    global_batch = scalar(src, "global_batch_size")
    if (not isinstance(step, int) or isinstance(step, bool)) or step < 0:
        raise ValueError("Source step_num must be a nonnegative integer scalar")
    if (not isinstance(global_batch, int) or isinstance(global_batch, bool)) or global_batch != 1:
        raise ValueError("This adapter requires source global_batch_size=1")
    if scalar(src, "loss_scale") != 1.0:
        raise ValueError("Dynamic or non-unit source loss scaling is unsupported")
    for family, arrays in (("model", model), ("optimizer", optimizer)):
        if not arrays:
            raise ValueError(f"No converted {family} tensors")
        for name, value in arrays.items():
            if value.dtype != np.dtype("float32") or not np.isfinite(value).all():
                raise ValueError(f"{family} tensor {name} must contain finite FP32 values")
    source_count = len(src)
    if input_report is not None:
        source_count = input_report.get("source_tensor_count") if isinstance(input_report, dict) else None
        if not isinstance(source_count, int) or isinstance(source_count, bool) or source_count < len(src):
            raise ValueError("Input metadata must declare an integer source_tensor_count covering the retained tensors")
    state = {
        "global_step": step,
        "step_num": step,
        "global_batch_size": global_batch,
        "consumed_samples": step * global_batch,
        "loss_scale": float(scalar(src, "loss_scale", 1.0)),
        "data_order": "sequential",
        "source_checkpoint": str(source_checkpoint.resolve()),
    }
    if "epoch_num" in src:
        state["source_epoch_num"] = scalar(src, "epoch_num")
    manifest = {
        "format": "mf-to-hp-offline-state-v1",
        "model_tensor_count": len(model),
        "optimizer_tensor_count": len(optimizer),
        "source_tensor_count": source_count,
        "optimizer_prefixes": sorted({name.split(".", 1)[0] for name in optimizer}),
        "state_file": "state.json",
        "model_file": "model.npz",
        "optimizer_file": "optimizer.npz",
        "note": "No hashes/checksums are calculated by this converter.",
    }
    if input_report is not None:
        manifest["source_input"] = input_report
    try:
        state_text = json.dumps(state, indent=2, allow_nan=False) + "\n"
        manifest_text = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint metadata must be finite and JSON serializable") from exc
    outputs = ("model.npz", "optimizer.npz", "state.json", "CONVERSION.json")
    if any((out_dir / name).exists() for name in outputs):
        raise ValueError("Refusing to replace existing converted checkpoint artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "model.npz", **model)
    np.savez(out_dir / "optimizer.npz", **optimizer)
    (out_dir / "state.json").write_text(state_text)
    (out_dir / "CONVERSION.json").write_text(manifest_text)


def main() -> None:
    """Convert one source checkpoint from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--mlp-layout", choices=("separate", "swiglu"), default="separate")
    parser.add_argument("--source-format", choices=("auto", "family-prefix", "mf-native"), default="auto")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out_dir is None and args.out is None:
        parser.error("one of --out-dir or --out is required")

    if args.out is not None and args.out.exists():
        raise ValueError(f"Refusing to replace existing output: {args.out}")
    src, input_report = load_checkpoint_input(args.ckpt, source_format=args.source_format)
    model = convert_model(src, mlp_layout=args.mlp_layout)
    optimizer = convert_optimizer(src, mlp_layout=args.mlp_layout)
    out_dir = args.out_dir or args.out.parent
    write_artifacts(src, model, optimizer, out_dir, source_checkpoint=args.ckpt, input_report=input_report)
    if args.out is not None and args.out != out_dir / "model.npz":
        args.out.write_bytes((out_dir / "model.npz").read_bytes())
    print(f"source tensors: {len(src)}")
    print(f"model tensors: {len(model)}")
    print(f"optimizer tensors: {len(optimizer)}")
    print(f"step: {scalar(src, 'step_num', 0)} / global_batch_size: {scalar(src, 'global_batch_size', 1)}")
    print(f"written: {out_dir}")


if __name__ == "__main__":
    main()
