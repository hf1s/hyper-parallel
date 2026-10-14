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
"""Compare the HP and MF MTP-chain taps, one operator boundary at a time.

For every pair the comparison is reported in three tiers, exactly as the alignment protocol
asks:

``fp32``      the two float32 values are bit-identical (integer comparison of the fp32 words)
``bf16``      both sides rounded to bfloat16 with round-to-nearest-even before comparing, i.e.
              the difference is at most a bf16 quantum
``raw``       the payloads are compared in their own dtypes, without any rounding

A tensor whose ``bf16`` tier still differs is a difference that bf16 rounding alone cannot
explain; the first such tensor along the backward chain is the verdict the report wants.

Usage: ``cmp_mtp.py [hp_dir] [mf_dir] [--out report.json]``
"""

import argparse
import json
from pathlib import Path

import numpy as np

#: backward chain order, downstream (loss) first; each entry lists the names to look for
PAIRS = (
    ('mtp_loss', 'val', ('mtp_loss',)),
    ('mtp_loss', 'dy', ('mtp_loss',)),
    ('mtp_layer_loss', 'val', ('mtp_layer_loss',)),
    ('mtp_layer_loss', 'dy', ('mtp_layer_loss',)),
    ('mtp_logits', 'dy', ('mtp_logits',)),
    ('mtp_logits', 'act', ('mtp_logits',)),
    ('mtp_head', 'dy', ('mtp_head', 'mtp_head_input')),
    ('mtp_head', 'act', ('mtp_head', 'mtp_head_input')),
    ('mtp_head', 'inputgrad', ('mtp_head', 'mtp_transformer_out|dy')),
    ('mtp_transformer_in', 'dy', ('mtp_transformer_in', 'mtp_eh_proj')),
    ('mtp_eh_proj', 'dy', ('mtp_eh_proj', 'mtp_transformer_in|dy')),
    ('mtp_concat', 'dy', ('mtp_concat', 'mtp_eh_proj')),
    ('mtp_shared', 'dy', ('mtp_shared', 'mtp_l0_hnorm')),
    ('mtp_emb', 'dy', ('mtp_emb', 'mtp_l0_enorm')),
    ('mtp_shared', 'x', ('mtp_shared', 'mtp_l0_hnorm')),
    ('mtp_shared', 'out', ('mtp_shared', 'mtp_l0_hnorm')),
    ('mtp_emb', 'x', ('mtp_emb', 'mtp_l0_enorm')),
    ('mtp_emb', 'out', ('mtp_emb', 'mtp_l0_enorm')),
    ('mtp_concat', 'out', ('mtp_concat', 'mtp_eh_proj')),
    ('mtp_eh_proj', 'out', ('mtp_eh_proj',)),
    ('mtp_shared', 'rstd', ('mtp_shared', 'mtp_l0_hnorm')),
    ('mtp_emb', 'rstd', ('mtp_emb', 'mtp_l0_enorm')),
)

BLOCK = 1 << 23

def _dtype_of(record):
    """Return ``('bf16'|'fp32'|'fp16'|None)`` for a published tap record."""
    dtype = str((record or {}).get('dtype', '')).replace('torch.', '')
    if dtype == 'bfloat16':
        return 'bf16'
    if dtype == 'float32':
        return 'fp32'
    if dtype == 'float16':
        return 'fp16'
    if dtype in ('int64', 'int32'):
        return dtype
    if (record or {}).get('source_payload_is_bf16'):
        return 'bf16'
    return None


def _stems(prefix, name, role, aliases):
    """Return the published stems to try, covering ``a__b``, ``a_b`` and bare-name spellings.

    ``prefix`` is ``hp__`` / ``mf__`` for a published directory, or empty for a raw run
    directory, where the file names are the tap keys themselves.

    An alias may be a plain name (same role) or ``name|role`` when the peer published the same
    tensor under a different role word.
    """
    specs = [(name, role)]
    for alias in aliases:
        if '|' in alias:
            alias_name, alias_role = alias.split('|', 1)
            specs.append((alias_name, alias_role))
        else:
            specs.append((alias, role))
    stems = []
    for alias_name, alias_role in specs:
        stems.append(f'{prefix}{alias_name}__{alias_role}')
        stems.append(f'{prefix}{alias_name}_{alias_role}')
    for alias_name, _ in specs:
        stems.append(f'{prefix}{alias_name}')
    return stems


def _path_of(directory, prefix, name, role, aliases):
    """Return the ``.raw`` path and metadata of a tap, trying every spelling."""
    for stem in _stems(prefix, name, role, aliases):
        plain = directory / f'{stem}.raw'
        if plain.exists():
            record = None
            meta = directory / f'{stem}.json'
            if meta.exists():
                record = json.loads(meta.read_text())
            return plain, record
    return None, None


def _load(path, kind):
    """Memory-map a published payload as float32 values plus the raw integer words."""
    if kind == 'bf16':
        words = np.fromfile(path, dtype='<u2')
        return (words.astype(np.uint32) << 16).view('<f4'), words
    if kind == 'fp16':
        words = np.fromfile(path, dtype='<u2')
        return words.view('<f2').astype(np.float32), words
    if kind == 'fp32':
        words = np.fromfile(path, dtype='<u4')
        return words.view('<f4'), words
    if kind in ('int64', 'int32'):
        words = np.fromfile(path, dtype='<i8' if kind == 'int64' else '<i4')
        return words.astype(np.float32), words
    raise ValueError(f'unsupported payload {path} ({kind})')


def _to_bf16_bits(values):
    """Round float32 values to bfloat16 with round-to-nearest-even, returned as uint16 words."""
    words = np.ascontiguousarray(values).view(np.uint32).astype(np.uint32)
    rounding = np.uint32(0x7FFF) + ((words >> np.uint32(16)) & np.uint32(1))
    return ((words + rounding) >> np.uint32(16)).astype(np.uint16)


def _stats(left, right, same_kind=True):
    """Return the three-tier comparison of two float32 value arrays, in blocks.

    Besides the three protocol tiers this also measures how far apart the two sides are in
    *bfloat16 quanta* (``ulp`` of the magnitude) and which way each differing element moved, so a
    uniform scale difference can be told apart from an independent last-bit rounding.
    """
    total = {'n': 0, 'diff_fp32': 0, 'diff_bf16': 0, 'diff_raw': 0,
             'ssum_l': 0.0, 'ssum_r': 0.0, 'ssum_d': 0.0, 'max_d': 0.0,
             'maxabs_l': 0.0, 'maxabs_r': 0.0, 'l_is_rne_of_r': 0,
             'n_up': 0, 'n_down': 0, 'max_ulp': 0.0, 'ssum_ulp': 0.0, 'n_ulp_gt_half': 0}
    for offset in range(0, left[0].size, BLOCK):
        lv, lw = left[0][offset:offset + BLOCK], left[1][offset:offset + BLOCK]
        rv, rw = right[0][offset:offset + BLOCK], right[1][offset:offset + BLOCK]
        ld = lv.astype(np.float64)
        rd = rv.astype(np.float64)
        delta = ld - rd
        total['n'] += lv.size
        total['diff_fp32'] += int(np.count_nonzero(lv.view(np.uint32) != rv.view(np.uint32)))
        lb, rb = _to_bf16_bits(lv), _to_bf16_bits(rv)
        total['diff_bf16'] += int(np.count_nonzero(lb != rb))
        if same_kind and lw.dtype == rw.dtype:
            total['diff_raw'] += int(np.count_nonzero(lw != rw))
        total['l_is_rne_of_r'] += int(np.count_nonzero(lb == rb))
        total['n_up'] += int(np.count_nonzero(delta > 0))
        total['n_down'] += int(np.count_nonzero(delta < 0))
        # bfloat16 ulp of the magnitude of each element: 7 explicit mantissa bits
        magnitude = np.maximum(np.abs(lv), np.abs(rv))
        _, exponent = np.frexp(magnitude)
        quantum = np.ldexp(np.float32(1.0), (exponent - 8).astype(np.int32))
        ratio = (np.abs(delta) / quantum).astype(np.float64)
        total['max_ulp'] = max(total['max_ulp'], float(ratio.max()))
        total['ssum_ulp'] += float((ratio * ratio).sum())
        total['n_ulp_gt_half'] += int(np.count_nonzero(ratio > 0.5))
        total['ssum_l'] += float((ld * ld).sum())
        total['ssum_r'] += float((rd * rd).sum())
        total['ssum_d'] += float((delta * delta).sum())
        total['max_d'] = max(total['max_d'], float(np.abs(delta).max()))
        total['maxabs_l'] = max(total['maxabs_l'], float(np.abs(ld).max()))
        total['maxabs_r'] = max(total['maxabs_r'], float(np.abs(rd).max()))
    total['rms_ulp'] = (total['ssum_ulp'] / total['n']) ** 0.5 if total['n'] else None
    return total


def _rms(ssum, count):
    return float(np.sqrt(ssum / count)) if count else None


def compare(hp_dir, mf_dir, hp_prefix='hp__', mf_prefix='mf__'):
    """Compare every paired tap of the two directories and return the report."""
    hp_dir, mf_dir = Path(hp_dir), Path(mf_dir)
    rows = []
    for name, role, aliases in PAIRS:
        row = {'name': name, 'role': role}
        hp_path, hp_record = _path_of(hp_dir, hp_prefix, name, role, aliases)
        mf_path, mf_record = _path_of(mf_dir, mf_prefix, name, role, aliases)
        row['hp_file'] = None if hp_path is None else hp_path.name
        row['mf_file'] = None if mf_path is None else mf_path.name
        if hp_path is None or mf_path is None:
            row['status'] = 'missing ' + ('hp' if hp_path is None else 'mf')
            rows.append(row)
            continue
        hp_kind = _dtype_of(hp_record or {})
        mf_kind = _dtype_of(mf_record or {})
        row['hp_dtype'], row['mf_dtype'] = hp_kind, mf_kind
        left, right = _load(hp_path, hp_kind), _load(mf_path, mf_kind)
        if left[0].size != right[0].size:
            row['status'] = f'numel mismatch {left[0].size} vs {right[0].size}'
            rows.append(row)
            continue
        stats = _stats(left, right, same_kind=(hp_kind == mf_kind))
        row.update({'numel': stats['n'], 'status': 'compared',
                    'n_diff_fp32': stats['diff_fp32'],
                    'n_diff_bf16_unified': stats['diff_bf16'],
                    'n_diff_raw': stats['diff_raw'] if hp_kind == mf_kind else None,
                    'n_bf16_equal': stats['l_is_rne_of_r'],
                    'n_up': stats['n_up'], 'n_down': stats['n_down'],
                    'max_diff_in_bf16_quanta': stats['max_ulp'],
                    'rms_diff_in_bf16_quanta': stats['rms_ulp'],
                    'n_diff_over_half_quantum': stats['n_ulp_gt_half'],
                    'rms_hp': _rms(stats['ssum_l'], stats['n']),
                    'rms_mf': _rms(stats['ssum_r'], stats['n']),
                    'rms_diff': _rms(stats['ssum_d'], stats['n']),
                    'maxabs_diff': stats['max_d'],
                    'absmax_hp': stats['maxabs_l'], 'absmax_mf': stats['maxabs_r'],
                    'fp32_bitwise_equal': stats['diff_fp32'] == 0,
                    'bf16_unified_equal': stats['diff_bf16'] == 0})
        del left, right
        rows.append(row)
    # the hnorm dy of each side is the reference magnitude the ratios are quoted against
    reference = {}
    for row in rows:
        if row['name'] == 'mtp_shared' and row['role'] == 'dy' and row.get('status') == 'compared':
            reference = row
    for row in rows:
        if row.get('status') != 'compared' or not reference:
            continue
        row['ratio_hp_to_hnorm_dy'] = row['rms_hp'] / reference['rms_hp']
        row['ratio_mf_to_hnorm_dy'] = row['rms_mf'] / reference['rms_mf']
        row['amp_hp_over_mf'] = row['rms_hp'] / row['rms_mf'] if row['rms_mf'] else None
    verdict = None
    for row in rows:
        if row.get('status') != 'compared':
            continue
        if not row['bf16_unified_equal']:
            verdict = {'tensor': f"{row['name']}__{row['role']}",
                       'rms_hp': row['rms_hp'], 'rms_mf': row['rms_mf'],
                       'rms_diff': row['rms_diff'],
                       'ratio_rms_diff_over_hnorm_dy': (
                           row['rms_diff'] / reference['rms_hp'] if reference else None),
                       'n_diff_bf16_unified': row['n_diff_bf16_unified'],
                       'numel': row['numel'],
                       'ratio_hp_to_hnorm_dy': row.get('ratio_hp_to_hnorm_dy'),
                       'amp_hp_over_mf': row.get('amp_hp_over_mf')}
            break
    return {'hp_dir': str(hp_dir), 'mf_dir': str(mf_dir), 'reference': 'mtp_shared__dy',
            'rows': rows, 'first_unexplained_bf16_difference': verdict}


def main():
    """Print the comparison table and write the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hp', nargs='?', default='/home/zth/tmp/align-cmp/bwdtaps/hp')
    parser.add_argument('mf', nargs='?', default='/home/zth/tmp/align-cmp/bwdtaps/mf')
    parser.add_argument('--left-run', action='store_true',
                        help='the left directory is a raw run tap directory (no hp__ prefix)')
    parser.add_argument('--right-run', action='store_true',
                        help='the right directory is a raw run tap directory (no mf__ prefix)')
    parser.add_argument('--out', default='/home/zth/tmp/align-cmp/bwdtaps/mtp_chain_compare.json')
    args = parser.parse_args()
    report = compare(args.hp, args.mf,
                     hp_prefix='' if args.left_run else 'hp__',
                     mf_prefix='' if args.right_run else 'mf__')
    header = (f'{"tensor":24s} {"hp":5s} {"mf":5s} {"numel":>10s} {"d_fp32":>10s} '
              f'{"d_bf16":>10s} {"up":>9s} {"down":>9s} {"maxQ":>8s} {"rmsQ":>9s} '
              f'{"rms_d":>10s} {"hp/hnorm":>9s} {"amp":>7s}')
    print(header)
    print('-' * len(header))
    for row in report['rows']:
        tag = f"{row['name']}__{row['role']}"
        if row.get('status') != 'compared':
            print(f'{tag:24s} {row.get("status")}')
            continue
        print(f'{tag:24s} {str(row["hp_dtype"]):5s} {str(row["mf_dtype"]):5s} '
              f'{row["numel"]:10d} {row["n_diff_fp32"]:10d} {row["n_diff_bf16_unified"]:10d} '
              f'{row["n_up"]:9d} {row["n_down"]:9d} {row["max_diff_in_bf16_quanta"]:8.3f} '
              f'{row["rms_diff_in_bf16_quanta"]:9.2e} {row["rms_diff"]:10.4g} '
              f'{row["ratio_hp_to_hnorm_dy"]:9.4g} {row["amp_hp_over_mf"]:7.4g}')
    print()
    print('first tensor whose difference bf16 rounding cannot explain:',
          json.dumps(report['first_unexplained_bf16_difference'], indent=2))
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print('wrote', args.out)


if __name__ == '__main__':
    main()
