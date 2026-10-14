#!/usr/bin/env python3
"""Compare two ``param_native`` dumps (and the step losses) of the Muon switch states.

Usage::

    python cmp_states.py <base-dir> <other-dir> [more dirs ...]

Prints one row per native parameter: whether the state is bit-identical to the base, how many
elements differ and the largest absolute difference, plus the step-301/302 losses when the run
directories sit next to the dumps as ``runs/<label>``.

The comparison is streamed in chunks so that the ~2.7e9-element parameter set is never held in
double precision all at once.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

CHUNK = 1 << 24
ONLY = {name for name in os.environ.get('CMP_ONLY', '').split(',') if name}


def names(directory: Path) -> list[str]:
    """Return the sorted parameter names of one dump directory."""
    found = sorted(path.stem for path in directory.glob('*.npy'))
    if ONLY:
        found = [name for name in found if name in ONLY]
    return found


def compare_one(base_dir: Path, other_dir: Path, name: str) -> dict:
    """Return the difference record of one parameter between two dumps."""
    left = np.load(str(base_dir / f'{name}.npy'), mmap_mode='r').reshape(-1)
    right = np.load(str(other_dir / f'{name}.npy'), mmap_mode='r').reshape(-1)
    numel = int(left.size)
    changed = 0
    max_abs = 0.0
    max_abs_ref = 0.0
    for offset in range(0, numel, CHUNK):
        stop = offset + CHUNK
        a = np.asarray(left[offset:stop], dtype=np.float32)
        b = np.asarray(right[offset:stop], dtype=np.float32)
        delta = np.abs(a - b)
        changed += int(np.count_nonzero(delta))
        max_abs = max(max_abs, float(delta.max()))
        max_abs_ref = max(max_abs_ref, float(np.abs(a).max()))
    del left, right
    return {
        'name': name,
        'numel': numel,
        'changed': changed,
        'max_abs_delta': max_abs,
        'max_rel_delta': max_abs / max_abs_ref if max_abs_ref else 0.0,
        'bit_identical': changed == 0,
    }


def losses(run_dir: Path) -> dict:
    """Return the recorded per-step losses of a run directory."""
    path = run_dir / 'metrics-pr180.jsonl'
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row['step']: row['loss_17g'] for row in rows}


def run_dir_for(directory: Path, label: str) -> Path:
    """Return the run directory that matches a dump directory."""
    base = directory.parent / 'runs'
    name = label[len('params-'):] if label.startswith('params-') else label
    for candidate in (base / name, base / label):
        if (candidate / 'metrics-pr180.jsonl').exists():
            return candidate
    return base / name


def main(argv: list[str]) -> int:
    """Compare every dump against the first one."""
    base_dir = Path(argv[1]).resolve()
    others = [(Path(arg).resolve().name, Path(arg).resolve()) for arg in argv[2:]]
    base_label = base_dir.name
    print(f'base={base_label} losses={losses(run_dir_for(base_dir, base_label))}')
    summary = {}
    for label, directory in others:
        records = [compare_one(base_dir, directory, name) for name in names(directory)]
        changed = [record for record in records if not record['bit_identical']]
        summary[label] = {
            'params': len(records),
            'changed': len(changed),
            'names': [record['name'] for record in changed],
            'max_abs_delta': max((r['max_abs_delta'] for r in changed), default=0.0),
            'unchanged': [record['name'] for record in records if record['bit_identical']],
            'losses': losses(run_dir_for(directory, label)),
        }
        print(f'\n=== {label}: {len(changed)}/{len(records)} parameters differ from {base_label}')
        for record in changed:
            print(f"  {record['name']:64s} {record['numel']:>11d} "
                  f"{100.0 * record['changed'] / max(1, record['numel']):6.2f}%  "
                  f"max|d|={record['max_abs_delta']:.4e} rel={record['max_rel_delta']:.3e}")
        print(f'  losses={summary[label]["losses"]}')
    print('\n=== json ===')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
