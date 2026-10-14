"""Compare two tap directories key by key, in the same three tiers as :mod:`cmp_mtp`.

Either side may be a raw run directory (files named after the tap keys) or a published directory
(files prefixed with ``hp__`` / ``mf__``).

Usage: ``cmp_keys.py <left_dir> <right_dir> [--left-published hp__] [--right-published mf__]``
"""

import argparse
import json
from pathlib import Path

import numpy as np

import cmp_mtp as C


def _side(directory, prefix):
    """Return ``{stem: raw path}`` for one directory."""
    directory = Path(directory)
    found = {}
    for path in sorted(directory.glob('*.raw')):
        stem = path.name[:-4]
        if prefix:
            if not stem.startswith(prefix):
                continue
            stem = stem[len(prefix):]
        found[stem] = path
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left')
    parser.add_argument('right')
    parser.add_argument('--left-published', default='')
    parser.add_argument('--right-published', default='')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    left = _side(args.left, args.left_published)
    right = _side(args.right, args.right_published)
    rows = []
    for key in sorted(set(left) & set(right)):
        kind_l = C._dtype_of(json.loads((Path(args.left) /
                                         ((args.left_published + key) + '.json')).read_text()))
        kind_r = C._dtype_of(json.loads((Path(args.right) /
                                         ((args.right_published + key) + '.json')).read_text()))
        data_l, data_r = C._load(left[key], kind_l), C._load(right[key], kind_r)
        if data_l[0].size != data_r[0].size:
            rows.append({'key': key, 'status': f'numel {data_l[0].size} vs {data_r[0].size}'})
            continue
        stats = C._stats(data_l, data_r, same_kind=(kind_l == kind_r))
        rows.append({'key': key, 'status': 'compared', 'dtype': kind_l, 'numel': stats['n'],
                     'n_diff_fp32': stats['diff_fp32'], 'n_up': stats['n_up'],
                     'n_down': stats['n_down'],
                     'max_diff_in_bf16_quanta': stats['max_ulp'],
                     'maxabs_diff': stats['max_d'],
                     'rms_left': C._rms(stats['ssum_l'], stats['n']),
                     'rms_right': C._rms(stats['ssum_r'], stats['n']),
                     'rms_diff': C._rms(stats['ssum_d'], stats['n'])})
        del data_l, data_r
    for key in sorted(set(left) - set(right)):
        rows.append({'key': key, 'status': 'only left'})
    for key in sorted(set(right) - set(left)):
        rows.append({'key': key, 'status': 'only right'})
    print(f'{"key":34s} {"dtype":5s} {"numel":>10s} {"n_diff":>10s} {"up":>9s} {"down":>9s} '
          f'{"maxQ":>8s} {"maxabs_d":>10s} {"rms_d":>10s}')
    for row in rows:
        if row['status'] != 'compared':
            print(f"{row['key']:34s} {row['status']}")
            continue
        print(f"{row['key']:34s} {row['dtype']:5s} {row['numel']:10d} {row['n_diff_fp32']:10d} "
              f"{row['n_up']:9d} {row['n_down']:9d} {row['max_diff_in_bf16_quanta']:8.3f} "
              f"{row['maxabs_diff']:10.4g} {row['rms_diff']:10.4g}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {'left': args.left, 'right': args.right, 'rows': rows}, indent=2, sort_keys=True) + '\n')
        print('wrote', args.out)


if __name__ == '__main__':
    main()
