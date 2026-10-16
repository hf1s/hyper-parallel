"""Small, uncompressed tensor writer shared by the paired step-302 captures."""
from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


def _dtype_name(dtype: Any) -> str:
    name = str(dtype).lower().removeprefix("torch.").removeprefix("mindspore.")
    return {"bf16": "bfloat16", "float": "float32", "double": "float64", "half": "float16"}.get(name, name)


def npy_storage_bytes(shape: tuple[int, ...] | list[int], storage_dtype: Any) -> int:
    """Exact uncompressed .npy v2 bytes, calculated from metadata without I/O."""
    dtype = np.dtype(storage_dtype)
    header = io.BytesIO()
    np.lib.format.write_array_header_2_0(header, {
        "descr": np.lib.format.dtype_to_descr(dtype),
        "fortran_order": False, "shape": tuple(shape),
    })
    return header.tell() + math.prod(shape) * dtype.itemsize


def record_cann_environment(root: str | Path, point: str) -> None:
    """Record the actual process bindings required by this CANN-8.5 experiment."""
    expected = Path("/workspace/Ascend/cann-8.5.0").resolve()
    actual = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
    libraries = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/") and "/Ascend/" in fields[5]:
            libraries.add(str(Path(fields[5]).resolve()))
    cann_libraries = sorted(path for path in libraries if "/cann" in path)
    if actual != expected or not cann_libraries or any(
            not path.startswith(str(expected) + "/") for path in cann_libraries):
        raise RuntimeError(f"CANN 8.5 required: home={actual}, loaded={cann_libraries}")
    payload = {
        "point": point,
        "ascend_home": str(actual),
        "ascend_opp_path": os.environ.get("ASCEND_OPP_PATH"),
        "ascend_aicpu_path": os.environ.get("ASCEND_AICPU_PATH"),
        "loaded_libraries": sorted(libraries),
    }
    with (Path(root) / f"environment-{point}.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"CANN_BOUNDARY point={point} home={actual} libraries={len(cann_libraries)}", flush=True)


class CaptureWriter:
    """Write one tensor at a time; refuse to reuse an existing capture manifest.

    BF16 is stored as uint16 without a numerical conversion. Callers exporting
    reconstructed bits must explicitly pass ``bitwise_original=False``. Files
    contain NumPy headers and raw tensor bytes, with no archive/checksum layer.
    A shared root has one writer; other ranks need separate roots.
    """

    def __init__(self, root: str | Path, backend: str, rank: int = 0, *, chunk_bytes: int = 8 << 20):
        if backend not in {"hp", "mf"}:
            raise ValueError("backend must be hp or mf")
        if chunk_bytes < 1:
            raise ValueError("chunk_bytes must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.rank = int(rank)
        self.chunk_bytes = int(chunk_bytes)
        self._manifest = (self.root / "records.jsonl").open("x", encoding="utf-8")
        self._sequence = 0
        self._occurrences: dict[tuple[int, str, str, str], int] = {}
        self._paths: dict[str, dict] = {}
        self.tensor_bytes = 0
        self.manifest_bytes = 0

    def write_tensor(self, step: int, stage: str, kind: str, name: str, tensor: Any, **metadata: Any) -> dict:
        """Export a real Torch/MindSpore tensor or a NumPy array, preserving dtype."""
        if hasattr(tensor, "to_local"):
            tensor = tensor.to_local()
        if hasattr(tensor, "detach") and hasattr(tensor, "cpu"):
            logical_dtype = _dtype_name(tensor.dtype)
            value = tensor.detach().cpu().contiguous()
            if logical_dtype == "bfloat16":
                import torch

                return self.write_array(
                    step, stage, kind, name, value.view(torch.uint16).numpy(),
                    dtype="bfloat16", encoding="bfloat16_bits", **metadata,
                )
            return self.write_array(step, stage, kind, name, value.numpy(), dtype=logical_dtype, **metadata)
        if hasattr(tensor, "asnumpy"):
            logical_dtype = _dtype_name(tensor.dtype)
            value = tensor.asnumpy()
            if logical_dtype == "bfloat16":
                if _dtype_name(value.dtype) != "bfloat16":
                    raise TypeError("BF16 asnumpy() cast its storage; export raw uint16 explicitly with write_array")
                return self.write_array(
                    step, stage, kind, name, value.view(np.uint16),
                    dtype="bfloat16", encoding="bfloat16_bits", **metadata,
                )
            return self.write_array(step, stage, kind, name, value, dtype=logical_dtype, **metadata)
        return self.write_array(step, stage, kind, name, np.asarray(tensor), **metadata)

    def write_array(
        self, step: int, stage: str, kind: str, name: str, array: np.ndarray,
        *, dtype: str | None = None, encoding: str = "native", **metadata: Any,
    ) -> dict:
        """Stream an array/memmap in C order without retaining another tensor copy."""
        value = np.asarray(array)
        if value.dtype.hasobject or value.dtype.fields or value.dtype.kind not in "biufc":
            raise TypeError(f"Unsupported raw tensor storage dtype: {value.dtype}")
        logical_dtype = _dtype_name(dtype if dtype is not None else value.dtype)
        if logical_dtype == "bfloat16":
            if value.dtype.kind != "u" or value.dtype.itemsize != 2 or encoding != "bfloat16_bits":
                raise TypeError("BF16 requires original uint16 storage and encoding='bfloat16_bits'")
        elif encoding != "native" or logical_dtype != _dtype_name(value.dtype):
            raise TypeError("Non-BF16 storage must match the logical dtype without a cast")
        if isinstance(step, bool) or int(step) != step or int(step) < 0:
            raise ValueError("step must be the nonnegative real logical step")
        if not all(isinstance(item, str) and item for item in (stage, kind, name)):
            raise ValueError("stage, kind and name must be nonempty strings")
        key = (int(step), stage, kind, name)
        occurrence = self._occurrences.get(key, 0)
        path = Path("tensors") / f"rank{self.rank:05d}" / f"{self._sequence:08d}.npy"
        record = {
            "step": int(step), "stage": stage, "kind": kind, "name": name,
            "dtype": logical_dtype, "shape": list(value.shape), "path": path.as_posix(),
            "storage_dtype": str(value.dtype), "encoding": encoding,
            "backend": self.backend, "rank": self.rank, "occurrence": occurrence,
            "bitwise_original": True,
        }
        reserved = set(record).intersection(metadata) - {"bitwise_original"}
        if reserved:
            raise ValueError(f"Capture metadata cannot replace reserved fields: {sorted(reserved)}")
        if "bitwise_original" in metadata and not isinstance(metadata["bitwise_original"], bool):
            raise TypeError("bitwise_original must be boolean")
        record.update(metadata)
        line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            np.lib.format.write_array_header_2_0(handle, {
                "descr": np.lib.format.dtype_to_descr(value.dtype),
                "fortran_order": False, "shape": value.shape,
            })
            iterator = np.nditer(
                value, flags=["external_loop", "buffered", "zerosize_ok"],
                op_flags=["readonly"], order="C",
                buffersize=max(1, self.chunk_bytes // value.dtype.itemsize),
            )
            for chunk in iterator:
                handle.write(chunk.tobytes(order="C"))
        self._manifest.write(line)
        self._manifest.flush()
        self._sequence += 1
        self._occurrences[key] = occurrence + 1
        self._paths[record["path"]] = record.copy()
        self.tensor_bytes += npy_storage_bytes(value.shape, value.dtype)
        self.manifest_bytes += len(line.encode("utf-8"))
        return record

    def write_reference(
        self, step: int, stage: str, kind: str, name: str, record: dict, **metadata: Any,
    ) -> dict:
        """Reference this writer's immutable file when the caller proves no mutation.

        This does not test tensor equality. The caller must establish unchanged
        live storage/version between the two boundaries, without hashing bytes.
        """
        if record.get("path") not in self._paths:
            raise ValueError("References must target a file already written by this writer")
        if isinstance(step, bool) or int(step) != step or int(step) < 0:
            raise ValueError("A reference needs a real logical step")
        if not all(isinstance(item, str) and item for item in (stage, kind, name)):
            raise ValueError("stage, kind and name must be nonempty strings")
        immutable = {"dtype", "shape", "path", "storage_dtype", "encoding", "backend", "rank",
                     "bitwise_original", "source_dtype", "source_storage", "reconstruction"}
        written = self._paths[record["path"]]
        if any(record.get(field) != written.get(field) for field in immutable):
            raise ValueError("Reference storage/provenance differs from the immutable source record")
        reserved = (immutable | {"step", "stage", "kind", "name", "occurrence"}).intersection(metadata)
        if reserved:
            raise ValueError(f"Reference metadata cannot replace immutable fields: {sorted(reserved)}")
        source = record
        key = (int(step), stage, kind, name)
        occurrence = self._occurrences.get(key, 0)
        result = dict(source, step=int(step), stage=stage, kind=kind, name=name, occurrence=occurrence,
                      reference_step=record["step"], reference_stage=record["stage"],
                      reference_kind=record["kind"], reference_name=record["name"])
        result.update(metadata)
        line = json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"
        self._manifest.write(line)
        self._manifest.flush()
        self.manifest_bytes += len(line.encode("utf-8"))
        self._occurrences[key] = occurrence + 1
        return result

    def register_array(
        self, step: int, stage: str, kind: str, name: str, path: str | Path,
        *, dtype: str, shape: tuple[int, ...] | list[int], encoding: str = "native",
        storage_dtype: str | None = None, **metadata: Any,
    ) -> dict:
        """Record an existing TensorDump .npy inside root, without reading/copying it.

        The producer supplies the actual header metadata. The comparison reader
        checks it against the file when the requested paired comparison is run.
        Registered files are not counted as bytes emitted by this writer.
        """
        supplied = Path(path)
        absolute = supplied if supplied.is_absolute() else self.root / supplied
        relative = absolute.resolve().relative_to(self.root.resolve())
        if relative.suffix != ".npy":
            raise ValueError("Only root-contained .npy files can be registered")
        logical_dtype = _dtype_name(dtype)
        storage = np.dtype(storage_dtype or ("uint16" if logical_dtype == "bfloat16" else logical_dtype))
        if logical_dtype == "bfloat16":
            if storage.kind != "u" or storage.itemsize != 2 or encoding != "bfloat16_bits":
                raise TypeError("BF16 registration requires raw uint16 and bfloat16_bits")
        elif encoding != "native" or logical_dtype != _dtype_name(storage):
            raise TypeError("Registered storage must match its logical dtype")
        if isinstance(step, bool) or int(step) != step or int(step) < 0:
            raise ValueError("step must be the real logical step")
        if not all(isinstance(item, str) and item for item in (stage, kind, name)):
            raise ValueError("stage, kind and name must be nonempty strings")
        dimensions = [int(size) for size in shape]
        if any(size < 0 or size != original for size, original in zip(dimensions, shape)):
            raise ValueError("shape must contain nonnegative integer dimensions")
        key = (int(step), stage, kind, name)
        occurrence = self._occurrences.get(key, 0)
        result = {
            "step": int(step), "stage": stage, "kind": kind, "name": name,
            "dtype": logical_dtype, "shape": dimensions, "path": relative.as_posix(),
            "storage_dtype": str(storage), "encoding": encoding,
            "backend": self.backend, "rank": self.rank, "occurrence": occurrence,
            "bitwise_original": True, "registered": True,
        }
        reserved = set(result).intersection(metadata) - {"bitwise_original"}
        if reserved:
            raise ValueError(f"Metadata cannot replace reserved fields: {sorted(reserved)}")
        if "bitwise_original" in metadata and not isinstance(metadata["bitwise_original"], bool):
            raise TypeError("bitwise_original must be boolean")
        result.update(metadata)
        line = json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"
        self._manifest.write(line)
        self._manifest.flush()
        self.manifest_bytes += len(line.encode("utf-8"))
        self._occurrences[key] = occurrence + 1
        self._paths.setdefault(result["path"], result.copy())
        return result

    @property
    def total_bytes(self) -> int:
        return self.tensor_bytes + self.manifest_bytes

    def close(self) -> None:
        self._manifest.close()

    def __enter__(self) -> "CaptureWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
