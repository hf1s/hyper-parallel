"""Print the shas of the already-published taps that a new run may overwrite."""

import json
import sys
from pathlib import Path

directory = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/zth/tmp/align-cmp/bwdtaps/hp')
prefixes = tuple(sys.argv[2].split(',')) if len(sys.argv) > 2 else (
    'model_norm', 'lm_head', 'l1_mlp', 'l0_')
index = json.loads((directory / 'INDEX.json').read_text())
for record in index['taps']:
    if record['name'].startswith(prefixes):
        print(f"{record['name']}__{record['role']:12s} {record.get('dtype'):9s} "
              f"{record.get('sha256_of_raw_bytes')}")
