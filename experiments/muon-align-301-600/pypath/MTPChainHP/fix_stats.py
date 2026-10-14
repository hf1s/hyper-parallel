"""Recompute the fp64 moments of every recorded tap from its raw payload.

The first version of :mod:`mtp_taps` computed ``fp64_ssum``/``min``/``max``/``absmax`` on the
*integer payload* of a bfloat16 tensor, i.e. on its bit patterns, instead of on its values.  The
sha256 is taken over the raw bytes and is therefore unaffected, no ``.raw`` file is touched, and
the comparison reads the raw bytes, so only this metadata needed repairing.

Usage: ``fix_stats.py <run-bwd-dir> [<published-dir>]``
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

#: on-disk payload dtype per tap dtype
KIND = {'bfloat16': '<u2', 'float32': '<f4', 'float16': '<f2'}

BLOCK = 1 << 22


def moments(path, dtype):
    """Return the fp64 moment dict of one raw payload, computed in blocks."""
    total = 0.0
    smallest = math.inf
    largest = -math.inf
    absmax = 0.0
    count = 0
    with open(path, 'rb') as stream:
        while True:
            chunk = np.fromfile(stream, dtype=KIND[dtype], count=BLOCK)
            if chunk.size == 0:
                break
            if dtype == 'bfloat16':
                values = (chunk.astype(np.uint32) << 16).view(np.float32)
            else:
                values = chunk.astype(np.float32)
            del chunk
            wide = values.astype(np.float64)
            total += float((wide * wide).sum())
            smallest = min(smallest, float(wide.min()))
            largest = max(largest, float(wide.max()))
            absmax = max(absmax, float(np.abs(wide).max()))
            count += wide.size
    return {'fp64_ssum': total, 'fp64_rms': math.sqrt(total / count) if count else None,
            'min': smallest, 'max': largest, 'absmax': absmax, 'numel': count}


def repair(record, path):
    """Return ``record`` with the fp64 moments recomputed from ``path``."""
    if record is None or not Path(path).exists():
        return False
    fixed = moments(path, record['dtype'])
    changed = any(record.get(key) != fixed[key] for key in fixed)
    record.update(fixed)
    return changed


def fix_run_dir(directory):
    """Repair ``index.json`` and every ``<key>.json`` of a run's tap directory."""
    directory = Path(directory)
    index = json.loads((directory / 'index.json').read_text())
    touched = 0
    for key, record in index['tensors'].items():
        if repair(record, directory / (key + '.raw')):
            touched += 1
        (directory / (key + '.json')).write_text(
            json.dumps(record, indent=2, sort_keys=True) + '\n')
        print(f'[fix] {key:34s} {record["dtype"]:9s} rms={record["fp64_rms"]:.6g} '
              f'max={record["max"]:.6g}', flush=True)
    (directory / 'index.json').write_text(json.dumps(index, indent=2, sort_keys=True) + '\n')
    print(f'[fix] run dir {directory}: {touched} record(s) corrected', flush=True)


def fix_published(directory):
    """Repair the JSON copies and ``INDEX.json`` of the published tap directory."""
    directory = Path(directory)
    index = json.loads((directory / 'INDEX.json').read_text())
    for record in index['taps']:
        if 'alias_of' in record:
            continue
        raw = (record.get('files') or {}).get('raw') or f"{record['key']}.raw"
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = directory / f"hp__{record['key']}.raw"
        if not candidate.exists():
            print(f'[fix] {record["key"]}: raw not present at {candidate}, left as is',
                  flush=True)
            continue
        fixed = moments(candidate, record['dtype'].replace('torch.', ''))
        record.update(fixed)
        for name in ('raw', 'json'):
            path = directory / (f"hp__{record['key']}.{name}")
            if path.exists() and name == 'json':
                on_disk = json.loads(path.read_text())
                on_disk.update(fixed)
                path.write_text(json.dumps(on_disk, indent=2, sort_keys=True) + '\n')
    (directory / 'INDEX.json').write_text(json.dumps(index, indent=1, sort_keys=True) + '\n')
    print(f'[fix] published {directory}: {len(index["taps"])} entries refreshed', flush=True)


if __name__ == '__main__':
    fix_run_dir(sys.argv[1])
    if len(sys.argv) > 2:
        fix_published(sys.argv[2])
