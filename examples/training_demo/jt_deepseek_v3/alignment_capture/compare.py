#!/usr/bin/env python3
"""Compare the real paired capture manifests without hashes or model-sized copies.

Names come from the clean adapter's mapping_for; the concat, interleaved FC1,
expert reshape/transpose rules are the existing MF-to-HP checkpoint rules.
Raw-bit equality and numerical equality are deliberately separate results.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Callable

import numpy as np


STAGES = {name: index for index, name in enumerate((
    "pre_step", "pre_forward", "forward", "loss", "post_backward", "post_clip", "optimizer", "post_update",
))}
NON_PARAMETER_KINDS = {"activation", "input", "loss", "scalar", "optimizer_state"}
SCALAR_KINDS = {"loss", "scalar", "optimizer_state", "optimizer_step", "optimizer_group_step"}


def dtype_name(value: str) -> str:
    value = value.lower().removeprefix("torch.").removeprefix("mindspore.")
    return {"bf16": "bfloat16", "float": "float32", "half": "float16", "double": "float64"}.get(value, value)


def load_manifest(root: Path, backend: str) -> tuple[list[dict], dict[tuple, dict]]:
    records, index, occurrences = [], {}, Counter()
    with (root / "records.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            for field in ("step", "stage", "kind", "name", "dtype", "shape", "path"):
                if field not in item:
                    raise ValueError(f"{backend} manifest line {line_number} lacks {field}")
            if isinstance(item["step"], bool) or not isinstance(item["step"], int) or item["step"] < 0:
                raise ValueError(f"{backend} line {line_number} has an invalid logical step")
            if item.get("backend", backend) != backend:
                raise ValueError(f"{backend} line {line_number} belongs to another backend")
            base = (item["step"], item["stage"], item["kind"], item["name"])
            occurrence = item.setdefault("occurrence", occurrences[base])
            occurrences[base] += 1
            key = (*base, occurrence)
            if key in index:
                raise ValueError(f"Ambiguous repeated {backend} record: {key}")
            item["_key"], item["_order"] = key, len(records)
            records.append(item)
            index[key] = item
    return records, index


def load_mapping(path: Path) -> Callable:
    # Execute the existing dependency-free conversion file, not the HP package.
    # This neither imports the training backend nor creates a copied convention.
    namespace = {"__name__": "_paired_layout_mapping", "__file__": str(path)}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    mapping = namespace.get("mapping_for")
    if not callable(mapping):
        raise ValueError(f"No mapping_for function in {path}")
    return mapping


def boundary_key(record: dict) -> tuple:
    stage = record["stage"]
    phase = STAGES.get(stage, STAGES["optimizer"] if stage.startswith("muon_") else
                       STAGES["forward"] if stage.startswith("forward") else len(STAGES))
    return record["step"], phase, record["_order"], stage


def record_info(record: dict) -> dict:
    fields = ("name", "dtype", "shape", "storage_dtype", "encoding", "path", "occurrence",
              "bitwise_original", "source_dtype", "source_storage", "reconstruction", "axes_to_hp",
              "reference_step", "reference_stage", "reference_kind", "reference_name", "alias_of",
              "step_authority", "authoritative", "optimizer_name", "group_index", "lr", "weight_decay",
              "betas", "eps", "layout", "blocks", "block_axis", "block_ranges", "ns_call", "momentum_call",
              "optimizer_hyperparameters", "selected_source_name", "selected_index", "selected_momentum_call",
              "queue_sources", "normalization_coordinates", "fp32_numerator_class", "scaled_call",
              "group_step", "matched_adamw_rms", "apply_lr_in_update", "parameter_add_alpha", "update_coordinates",
              "coefficient_scope", "actual_call_occurrence", "actual_call_id_source")
    result = {key: record[key] for key in fields if key in record}
    if "_block_position" in record:
        position = record["_block_position"]
        result["selected_block_position"] = position
        result["selected_block"] = record["blocks"][position]
        result["selected_block_occurrence"] = record["_block_occurrence"]
    return result


def native_names(origin: str) -> list[str]:
    return [origin, origin.removeprefix("model.")] if origin.startswith("model.") else [origin]


def activation_origin(name: str, mapping: Callable) -> str:
    for marker in (".input", ".output"):
        offset = name.rfind(marker)
        if offset < 0:
            continue
        module, suffix = name[:offset], name[offset:]
        for ending in (".weight", ".input_layernorm.weight"):
            try:
                origins, transform = mapping(module + ending)
            except ValueError:
                continue
            if transform == "identity" and len(origins) == 1 and origins[0].endswith(ending):
                return origins[0][:-len(ending)] + suffix
        break
    raise ValueError(f"No proven identity boundary mapping for {name}")


def block_records(records: list[dict]) -> dict[tuple, dict]:
    """Index real NS blocks, not the optimizer's incidental batching/call names."""
    index, occurrences = {}, Counter()
    for record in records:
        if record.get("layout") != "optimizer_native":
            continue
        blocks = record.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            raise ValueError("optimizer_native records need the actual nonempty blocks metadata")
        for position, block in enumerate(blocks):
            base = (record["step"], record["stage"], record["kind"], block["name"], block["block_index"])
            occurrence = occurrences[base]
            occurrences[base] += 1
            index[(*base, occurrence)] = dict(record, _block_position=position, _block_occurrence=occurrence)
    return index


def native_block(block: dict, mapping: Callable) -> tuple[str, int, str]:
    """Use the existing optimizer split rules at the NS function boundary.

    The function has already undone its internal tall/wide transpose. FC1 has
    already been deinterleaved, and Q/KV has already been split into Q, KV, RoPE.
    Expert blocks retain their actual expert axis and need only last-two swap.
    """
    origins, transform = mapping(block["name"])
    number = block["block_index"]
    if transform == "concat":
        return (origins[0], 0, "identity") if number == 0 else (origins[1], number - 1, "identity")
    if transform in {"even", "odd"}:
        if number != 0:
            raise ValueError("A split HP gate/up NS projection must have its single block index 0")
        return origins[0], int(transform == "odd"), "identity"
    if transform == "expert_transpose":
        return origins[0], number, "expert_block_transpose"
    if transform == "identity":
        return origins[0], number, "identity"
    raise ValueError(f"Unsupported optimizer block rule {transform}")


def select_block(array: np.ndarray, record: dict) -> np.ndarray:
    if "_block_position" not in record:
        return array
    position = record["_block_position"]
    blocks = record["blocks"]
    shape = tuple(blocks[position]["shape"])
    if "block_axis" not in record:
        raise ValueError("NS capture must state block_axis; expert axis is not an implicit batch axis")
    axis = record["block_axis"]
    if axis is None:
        if len(blocks) != 1:
            raise ValueError("An unbatched NS capture must identify exactly one block")
        view = array
    else:
        if axis != 0 or not record.get("block_ranges") or len(record["block_ranges"]) != len(blocks):
            raise ValueError("Batched NS capture needs actual dim-0 block_ranges")
        interval = record["block_ranges"][position]
        start, stop = interval["start"], interval["stop"]
        if not isinstance(start, int) or not isinstance(stop, int) or not 0 <= start < stop <= array.shape[0]:
            raise ValueError("NS block range lies outside its real tensor")
        view = array[start:stop]
    if view.shape != shape:
        if not view.flags.c_contiguous:
            raise ValueError("Refusing a whole-block copy to reshape noncontiguous optimizer storage")
        view = view.reshape(shape)
    return view


def find_sources(hp: dict, mf_index: dict[tuple, dict], explicit_names: dict[tuple, list[dict]], mapping: Callable):
    if hp.get("layout") == "optimizer_native":
        raise ValueError("Optimizer-native tensors must be paired by their actual block metadata")
    prefix = (hp["step"], hp["stage"], hp["kind"])
    occurrence = hp["occurrence"]
    explicit = explicit_names.get((*prefix, hp["name"], occurrence), [])
    if explicit:
        if len(explicit) != 1:
            raise ValueError("Ambiguous explicit comparison_name")
        return explicit, "identity", []
    if hp["kind"] in NON_PARAMETER_KINDS:
        origins, transform = [hp["name"]], "identity"
        if hp["kind"] == "activation" and (*prefix, hp["name"], occurrence) not in mf_index:
            origins = [activation_origin(hp["name"], mapping)]
    else:
        origins, transform = mapping(hp["name"])
    sources, missing = [], []
    for origin in origins:
        candidates = [mf_index[(*prefix, name, occurrence)] for name in native_names(origin)
                      if (*prefix, name, occurrence) in mf_index]
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous native names for {origin}")
        if candidates:
            sources.append(candidates[0])
        else:
            missing.append(origin)
    return sources, transform, missing


def open_array(root: Path, record: dict, opened: list[np.memmap]) -> np.ndarray:
    relative = Path(record["path"])
    if relative.is_absolute() or relative.suffix != ".npy":
        raise ValueError("Capture path must be a relative .npy file")
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(array, np.memmap):
        raise ValueError("Capture is not a raw memory-mappable .npy array")
    opened.append(array)
    if list(array.shape) != record["shape"]:
        raise ValueError(f"Header shape {list(array.shape)} differs from manifest {record['shape']}")
    if array.dtype.hasobject or array.dtype.fields or array.dtype.kind not in "biufc":
        raise ValueError(f"Unsupported numeric storage {array.dtype}")
    if "storage_dtype" in record and array.dtype != np.dtype(record["storage_dtype"]):
        raise ValueError("Header storage dtype differs from manifest")
    logical = dtype_name(record["dtype"])
    if logical == "bfloat16":
        if array.dtype.kind != "u" or array.dtype.itemsize != 2 or record.get("encoding") != "bfloat16_bits":
            raise ValueError("BF16 requires recorded raw uint16 bits, not a float cast")
    elif logical != dtype_name(str(array.dtype)) or record.get("encoding", "native") != "native":
        raise ValueError("Logical dtype does not describe the stored native values")
    return array


def aligned_views(hp: np.ndarray, sources: list[np.ndarray], transform: str, kind: str):
    if kind in SCALAR_KINDS and hp.size == 1 and all(source.size == 1 for source in sources):
        # Compare every real origin counter separately; never add/average them.
        return [(hp.reshape(()), source.reshape(())) for source in sources]
    if transform == "identity":
        if len(sources) != 1 or hp.shape != sources[0].shape:
            raise ValueError("Identity-layout tensor shapes differ")
        return [(hp, sources[0])]
    if transform in {"even", "odd"}:
        if len(sources) != 1 or sources[0].ndim == 0:
            raise ValueError("FC1 interleaving requires a row axis")
        view = sources[0][0 if transform == "even" else 1::2]
        if hp.shape != view.shape:
            raise ValueError("Deinterleaved FC1 shape differs from HP")
        return [(hp, view)]
    if transform == "concat":
        if len(sources) != 2 or hp.ndim == 0 or any(source.ndim != hp.ndim for source in sources):
            raise ValueError("Q/KV concatenation requires two native tensors of matching rank")
        if any(source.shape[1:] != hp.shape[1:] for source in sources) or sum(source.shape[0] for source in sources) != hp.shape[0]:
            raise ValueError("Native Q/KV rows do not reconstruct HP packed shape")
        split = sources[0].shape[0]
        return [(hp[:split], sources[0]), (hp[split:], sources[1])]
    if transform == "expert_block_transpose":
        if len(sources) != 1 or hp.ndim != 3 or sources[0].ndim != 3:
            raise ValueError("Expert NS blocks must retain their real three-dimensional expert layout")
        view = sources[0].swapaxes(-1, -2)
        if view.shape != hp.shape:
            raise ValueError("Expert NS block shapes differ after the proven last-two-axis transpose")
        return [(hp, view)]
    if transform == "expert_transpose":
        if len(sources) != 1 or hp.ndim != 3 or sources[0].size != hp.size:
            raise ValueError("Expert packing requires HP [experts, output, input] and equal element counts")
        if not sources[0].flags.c_contiguous:
            raise ValueError("Native expert storage is not C-contiguous; refusing a full-tensor reshape copy")
        experts, output_width, input_width = hp.shape
        view = sources[0].reshape(experts, input_width, output_width).transpose(0, 2, 1)
        return [(hp, view)]
    raise ValueError(f"Unsupported existing layout transform {transform}")


def decoded(values: np.ndarray, record: dict) -> np.ndarray:
    if dtype_name(record["dtype"]) == "bfloat16":
        return (values.astype(np.uint32) << 16).view(np.float32)
    return values


def raw_bit_comparable(hp: dict, sources: list[dict], arrays: list[np.ndarray]) -> bool:
    records = [hp, *sources]
    return (all(record.get("bitwise_original") is True for record in records)
            and len({dtype_name(record["dtype"]) for record in records}) == 1
            and len({array.dtype.str for array in arrays}) == 1)


def compare_tensor(hp: dict, sources: list[dict], transform: str, hp_root: Path, mf_root: Path, chunk_bytes: int) -> dict:
    result = {"step": hp["step"], "stage": hp["stage"], "kind": hp["kind"], "name": hp["name"],
              "occurrence": hp["occurrence"], "transform": transform,
              "hp": record_info(hp), "mf": [record_info(source) for source in sources]}
    opened: list[np.memmap] = []
    try:
        for source in sources:
            if "micro_step" in hp and "micro_step" in source and hp["micro_step"] != source["micro_step"]:
                return dict(result, status="alignment_mismatch", reason="micro_step metadata differs")
            if hp.get("normalization_coordinates") != source.get("normalization_coordinates"):
                return dict(result, status="alignment_mismatch", reason="Normalized NS coordinate metadata differs")
        left = select_block(open_array(hp_root, hp, opened), hp)
        right = [select_block(open_array(mf_root, source, opened), source) for source in sources]
        if hp["kind"] == "activation" and len(right) == 1 and "axes_to_hp" in sources[0]:
            axes = sources[0]["axes_to_hp"]
            if sorted(axes) != list(range(right[0].ndim)):
                raise ValueError("axes_to_hp is not a complete permutation")
            right[0] = right[0].transpose(axes)
            result["activation_axes_to_hp"] = axes
        try:
            views = aligned_views(left, right, transform, hp["kind"])
        except ValueError as error:
            return dict(result, status="shape_mismatch", reason=str(error),
                        different_count=None, absmax=None, bitwise_different_count=None)
        bits_comparable = raw_bit_comparable(hp, sources, [left, *right])
        dtype_equal = all(dtype_name(hp["dtype"]) == dtype_name(source["dtype"]) for source in sources)
        numerical_count, bit_count, finite_absmax = 0, 0, 0.0
        nan_difference, inf_difference, elements = False, False, 0
        # nditer buffers only a bounded slice even for strided expert/FC1 views.
        chunk_elements = max(1, chunk_bytes // (2 * max(array.dtype.itemsize for array in [left, *right])))
        for part, (left_view, right_view) in enumerate(views):
            source = sources[part] if len(sources) > 1 else sources[0]
            iterator = np.nditer(
                [left_view, right_view], flags=["external_loop", "buffered", "zerosize_ok"],
                op_flags=[["readonly"], ["readonly"]], order="C", buffersize=chunk_elements,
            )
            for left_raw, right_raw in iterator:
                elements += left_raw.size
                if bits_comparable:
                    if left_raw.dtype.itemsize in (1, 2, 4, 8):
                        unsigned = np.dtype(f"u{left_raw.dtype.itemsize}")
                        bit_count += int(np.count_nonzero(left_raw.view(unsigned) != right_raw.view(unsigned)))
                    else:
                        left_bytes = left_raw.view(np.uint8).reshape(-1, left_raw.dtype.itemsize)
                        right_bytes = right_raw.view(np.uint8).reshape(-1, right_raw.dtype.itemsize)
                        bit_count += int(np.count_nonzero(np.any(left_bytes != right_bytes, axis=1)))
                left_values, right_values = decoded(left_raw, hp), decoded(right_raw, source)
                equal = np.equal(left_values, right_values)
                if left_values.dtype.kind in "fc" and right_values.dtype.kind in "fc":
                    equal |= np.isnan(left_values) & np.isnan(right_values)
                different = ~equal
                count = int(np.count_nonzero(different))
                numerical_count += count
                if count:
                    numeric_dtype = np.complex128 if left_values.dtype.kind == "c" or right_values.dtype.kind == "c" else np.float64
                    with np.errstate(over="ignore", invalid="ignore"):
                        delta = np.abs(left_values[different].astype(numeric_dtype) - right_values[different].astype(numeric_dtype))
                    nan_difference |= bool(np.any(np.isnan(delta)))
                    inf_difference |= bool(np.any(np.isinf(delta)))
                    finite = delta[np.isfinite(delta)]
                    if finite.size:
                        finite_absmax = max(finite_absmax, float(np.max(finite)))
        status = "different" if numerical_count or (bits_comparable and bit_count) else "equal" if bits_comparable else "numerically_equal"
        if not dtype_equal:
            status = "dtype_mismatch"
        return dict(result, status=status, compared_elements=elements, canonical_shape=list(left.shape),
                    dtype_equal=dtype_equal, different_count=numerical_count,
                    absmax="nan" if nan_difference else "inf" if inf_difference else finite_absmax,
                    finite_absmax=finite_absmax, original_bits_comparable=bits_comparable,
                    bitwise_different_count=bit_count if bits_comparable else None,
                    bitwise_equal=(bit_count == 0) if bits_comparable else None,
                    numerically_equal=numerical_count == 0,
                    numeric_nan_policy="NaN pairs equal numerically; payload/sign compared only in original bits")
    except (OSError, ValueError, TypeError) as error:
        return dict(result, status="invalid_artifact", reason=str(error))
    finally:
        for array in opened:
            array._mmap.close()


def short_result(result: dict) -> dict:
    fields = ("step", "stage", "kind", "name", "occurrence", "status", "different_count", "absmax",
              "bitwise_different_count", "original_bits_comparable", "reason", "hp", "mf")
    return {key: result[key] for key in fields if key in result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hp-root", type=Path, required=True)
    parser.add_argument("--mf-root", type=Path, required=True)
    parser.add_argument("--mapping-file", type=Path, required=True, help="Clean adapter/runtime/convert_state.py")
    parser.add_argument("--output", type=Path, required=True, help="New comparison directory; never overwritten")
    parser.add_argument("--chunk-mib", type=int, default=8)
    args = parser.parse_args()
    if args.chunk_mib < 1:
        parser.error("--chunk-mib must be positive")
    mapping = load_mapping(args.mapping_file)
    hp_records, _ = load_manifest(args.hp_root, "hp")
    mf_records, mf_index = load_manifest(args.mf_root, "mf")
    hp_blocks, mf_blocks = block_records(hp_records), block_records(mf_records)
    hp_blocks_by_record: dict[tuple, list[dict]] = {}
    for block in hp_blocks.values():
        hp_blocks_by_record.setdefault(block["_key"], []).append(block)
    explicit_names: dict[tuple, list[dict]] = {}
    for record in mf_records:
        if "comparison_name" in record:
            key = (record["step"], record["stage"], record["kind"], record["comparison_name"], record["occurrence"])
            explicit_names.setdefault(key, []).append(record)
    args.output.mkdir(parents=True, exist_ok=False)
    used, counts, boundaries = set(), Counter(), {}
    used_mf_blocks: set[tuple] = set()
    compared_files: dict[tuple, dict] = {}
    first_unequal = first_missing = first_values_only = first_numeric = first_bits = None
    with (args.output / "comparisons.jsonl").open("x", encoding="utf-8") as report:
        def emit(result: dict) -> None:
            nonlocal first_unequal, first_missing, first_values_only, first_numeric, first_bits
            report.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            counts[result["status"]] += 1
            boundary = (result["step"], result["stage"])
            boundaries.setdefault(boundary, Counter())[result["status"]] += 1
            if result["status"] in {"different", "dtype_mismatch", "shape_mismatch", "alignment_mismatch"} and first_unequal is None:
                first_unequal = short_result(result)
            if (result["status"].startswith("unpaired") or result["status"] == "invalid_artifact") and first_missing is None:
                first_missing = short_result(result)
            if result["status"] == "numerically_equal" and first_values_only is None:
                first_values_only = short_result(result)
            if result.get("different_count", 0) and first_numeric is None:
                first_numeric = short_result(result)
            if result.get("bitwise_different_count", 0) and first_bits is None:
                first_bits = short_result(result)

        def compare_and_emit(hp: dict, sources: list[dict], transform: str) -> None:
            identity = (transform, hp["kind"] in SCALAR_KINDS, tuple(
                (record["path"], record["dtype"], tuple(record["shape"]),
                 record.get("storage_dtype"), record.get("encoding"), record.get("bitwise_original"),
                 tuple(record.get("axes_to_hp", ())),
                 json.dumps({field: record.get(field) for field in
                             ("_block_position", "blocks", "block_axis", "block_ranges", "normalization_coordinates")}, sort_keys=True))
                for record in [hp, *sources]))
            micro_steps_align = all("micro_step" not in hp or "micro_step" not in source
                                    or hp["micro_step"] == source["micro_step"] for source in sources)
            previous = compared_files.get(identity) if micro_steps_align else None
            if previous is not None:
                result = dict(previous, step=hp["step"], stage=hp["stage"], kind=hp["kind"], name=hp["name"],
                              occurrence=hp["occurrence"], hp=record_info(hp),
                              mf=[record_info(source) for source in sources],
                              reused_comparison_of={field: previous[field] for field in ("step", "stage", "kind", "name")})
            else:
                result = compare_tensor(hp, sources, transform, args.hp_root, args.mf_root, args.chunk_mib << 20)
                if result["status"] not in {"alignment_mismatch", "invalid_artifact"}:
                    compared_files[identity] = result
            if hp.get("coefficient_scope") == "all_gradients" and all(
                    source.get("coefficient_scope") == "per_apply_global_norm" for source in sources):
                result["comparison_relation"] = "One actual HP common factor versus each actual MF per-apply factor"
            emit(result)

        for hp in sorted(hp_records, key=boundary_key):
            if (hp["stage"], hp["kind"], hp["name"]) == ("post_clip", "scalar", "clip_coefficient") and hp.get("coefficient_scope") == "all_gradients":
                actual_factors = [source for source in mf_records
                                  if (source["step"], source["stage"], source["kind"], source["name"]) ==
                                  (hp["step"], "post_clip", "scalar", "clip_coefficient")
                                  and source.get("coefficient_scope") == "per_apply_global_norm"]
                if actual_factors:
                    for source in sorted(actual_factors, key=lambda item: item.get("actual_call_occurrence", item["_order"])):
                        used.add(source["_key"])
                        compare_and_emit(hp, [source], "identity")
                    continue
            if hp.get("layout") == "optimizer_native":
                for selected in hp_blocks_by_record[hp["_key"]]:
                    block = selected["blocks"][selected["_block_position"]]
                    try:
                        origin, native_index, transform = native_block(block, mapping)
                        prefix = (hp["step"], hp["stage"], hp["kind"])
                        candidates = [mf_blocks[(*prefix, name, native_index, selected["_block_occurrence"])]
                                      for name in native_names(origin)
                                      if (*prefix, name, native_index, selected["_block_occurrence"]) in mf_blocks]
                        if len(candidates) != 1:
                            raise ValueError(f"Expected one native NS block {origin}[{native_index}], found {len(candidates)}")
                        source = candidates[0]
                    except ValueError as error:
                        emit({"step": hp["step"], "stage": hp["stage"], "kind": hp["kind"], "name": hp["name"],
                              "occurrence": hp["occurrence"], "status": "unpaired_hp", "reason": str(error),
                              "hp": record_info(selected)})
                        continue
                    used_mf_blocks.add((source["_key"], source["_block_position"]))
                    compare_and_emit(selected, [source], transform)
                continue
            try:
                sources, transform, missing = find_sources(hp, mf_index, explicit_names, mapping)
            except ValueError as error:
                emit({"step": hp["step"], "stage": hp["stage"], "kind": hp["kind"], "name": hp["name"],
                      "occurrence": hp["occurrence"], "status": "unpaired_hp", "reason": str(error), "hp": record_info(hp)})
                continue
            used.update(source["_key"] for source in sources)
            if missing:
                emit({"step": hp["step"], "stage": hp["stage"], "kind": hp["kind"], "name": hp["name"],
                      "occurrence": hp["occurrence"], "status": "unpaired_hp", "reason": "Missing native MF record",
                      "missing_names": missing, "hp": record_info(hp), "mf": [record_info(source) for source in sources]})
                continue
            compare_and_emit(hp, sources, transform)
        for mf in mf_records:
            if mf.get("layout") == "optimizer_native":
                missing_positions = [position for position in range(len(mf["blocks"]))
                                     if (mf["_key"], position) not in used_mf_blocks]
                if missing_positions:
                    mf["_unpaired_blocks"] = [mf["blocks"][position] for position in missing_positions]
                else:
                    used.add(mf["_key"])
        for mf in sorted(mf_records, key=boundary_key):
            if mf["_key"] not in used:
                emit({"step": mf["step"], "stage": mf["stage"], "kind": mf["kind"], "name": mf["name"],
                      "occurrence": mf["occurrence"], "status": "unpaired_mf", "reason": "No matching captured HP boundary",
                      "mf": [record_info(mf)], **({"unpaired_blocks": mf["_unpaired_blocks"]} if "_unpaired_blocks" in mf else {})})
    # Missing MF-only boundaries are emitted after paired rows; sort their first
    # boundary against an HP-only boundary instead of claiming emission is time.
    missing_mf = [record for record in mf_records if record["_key"] not in used]
    if missing_mf:
        earliest = min(missing_mf, key=boundary_key)
        if first_missing is None or boundary_key(dict(first_missing, _order=0))[:2] > boundary_key(earliest)[:2]:
            first_missing = {"step": earliest["step"], "stage": earliest["stage"], "kind": earliest["kind"],
                             "name": earliest["name"], "occurrence": earliest["occurrence"], "status": "unpaired_mf",
                             "mf": [record_info(earliest)]}
    summary = {
        "hp_root": str(args.hp_root), "mf_root": str(args.mf_root), "mapping_file": str(args.mapping_file),
        "hp_record_count": len(hp_records), "mf_record_count": len(mf_records), "status_counts": dict(counts),
        "first_unequal_boundary": first_unequal, "first_numeric_difference": first_numeric,
        "first_original_bit_difference": first_bits, "first_incomplete_boundary": first_missing,
        "first_values_only_agreement": first_values_only,
        "boundaries": [{"step": step, "stage": stage, "status_counts": dict(statuses)}
                       for (step, stage), statuses in sorted(boundaries.items(), key=lambda pair: boundary_key(
                           {"step": pair[0][0], "stage": pair[0][1], "_order": 0}))],
        "scope": "Earliest observed difference among matched captures; missing/invalid boundaries are not equality evidence. Numerical or reconstructed-bit agreement is never original-bit equality.",
        "tensor_report": "comparisons.jsonl", "hashes_or_checksums_calculated": False,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with (args.output / "summary.json").open("x", encoding="utf-8") as handle:
        handle.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
