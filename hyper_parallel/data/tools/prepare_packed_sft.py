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
"""Convert local pre-tokenized packed Arrow SFT records without re-tokenization."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.data.text.pretokenized_sft import PreTokenizedSFTTransform
from hyper_parallel.data.tools.io import IndexedDatasetBuilder


def convert_sft(source: Path, output_prefix: Path, sequence_length: int | None = None) -> int:
    """Write tokens, shifted labels, supervision masks and original document boundaries.

    Args:
        source: Local Arrow IPC stream file from a saved Hugging Face dataset.
        output_prefix: Common prefix of the four indexed stream pairs.
        sequence_length: Optional window length; source records must divide exactly.

    Returns:
        Number of records with at least one supervised target.
    """
    # Arrow is needed only by this offline conversion command.
    import pyarrow as pa  # pylint: disable=import-outside-toplevel
    import pyarrow.ipc as ipc  # pylint: disable=import-outside-toplevel

    transform = PreTokenizedSFTTransform(max_seq_len=sequence_length)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    names = ("tokens", "labels", "loss_mask", "cu_seqlens")
    for name in names:
        for suffix in ("bin", "idx"):
            if Path(f"{output_prefix}.{name}.{suffix}").exists():
                raise FileExistsError(f"Output already exists: {output_prefix}.{name}.{suffix}")
    count = 0
    with ExitStack() as resources:
        writers = {}
        for name in names:
            dtype = np.uint8 if name == "loss_mask" else np.int32
            writer = IndexedDatasetBuilder(f"{output_prefix}.{name}.bin", dtype=dtype)
            resources.callback(writer.data_file.close)
            writers[name] = writer
        stream = resources.enter_context(pa.memory_map(str(source), "r"))
        reader = ipc.open_stream(stream)
        for batch in reader:
            for row in batch.to_pylist():
                if sequence_length is not None and len(row["input_ids"]) % sequence_length:
                    raise ValueError("Record length must be divisible by the requested window length")
                for sample in transform(row):
                    labels = sample["labels"].numpy()
                    fields = (sample["input_ids"].numpy(), labels, labels != IGNORE_INDEX,
                              sample["cu_seq_lens"].numpy())
                    for name, values in zip(names, fields):
                        writers[name].add_document(values, [len(values)])
                    count += 1
        if count == 0:
            raise ValueError("No supervised records were found")
        for name, writer in writers.items():
            writer.finalize(f"{output_prefix}.{name}.idx")
    return count


def main() -> None:
    """Convert a local Arrow file to indexed SFT streams."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int)
    args = parser.parse_args()
    convert_sft(args.input, args.output_prefix, args.sequence_length)


if __name__ == "__main__":
    main()
