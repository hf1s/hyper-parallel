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
"""Generate deterministic documents for the DeepSeek-V3.2 single-card demo."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

from transformers import AutoTokenizer


def prepare_single_card_data(
        output_path: Path,
        tokenizer_path: str,
        *,
        num_samples: int = 64,
        sequence_length: int = 4096,
) -> None:
    """Write JSONL records that reach both DSA attention patterns.

    Each record tokenizes to about ``2 * sequence_length + sequence_length // 4``
    tokens, which the Online transform cuts into two full windows and a short
    tail. A full window is a single document longer than ``index_topk`` (2048
    in the released config), so the indexer's top-k selection is active; the
    tails of several records pack into one row and exercise the per-document
    boundaries.

    Args:
        output_path: JSONL file to create.
        tokenizer_path: Local DeepSeek-V3.2 tokenizer directory used to size records.
        num_samples: Number of records.
        sequence_length: Model sequence length (``data_transform.max_seq_len``).
    """
    if num_samples <= 0 or sequence_length <= 0:
        raise ValueError("num_samples and sequence_length must be positive")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    target_tokens = 2 * sequence_length + sequence_length // 4
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for sample_index in range(num_samples):
            sentence = (
                f"DeepSeek-V3.2 single-card document {sample_index}: the lightning indexer scores "
                "earlier tokens and sparse attention keeps the strongest candidates. "
            )
            first = len(tokenizer.encode(sentence, add_special_tokens=False))
            # Repeated sentences merge at their boundary; measure the steady-state length.
            repeated = len(tokenizer.encode(sentence * 2, add_special_tokens=False)) - first
            repeats = 1 + math.ceil((target_tokens - first) / repeated)
            output_file.write(json.dumps({"text": sentence * repeats}) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare DeepSeek-V3.2 single-card demo data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=4096)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Generate the requested JSONL dataset."""
    args = _parse_args(argv)
    prepare_single_card_data(
        Path(args.output).expanduser().resolve(),
        args.tokenizer_path,
        num_samples=args.num_samples,
        sequence_length=args.sequence_length,
    )


if __name__ == "__main__":
    main()
