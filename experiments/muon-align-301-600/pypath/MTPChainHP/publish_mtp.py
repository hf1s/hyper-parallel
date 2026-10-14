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
"""Publish the HP MTP-chain taps into the shared MF/HP comparison directory.

Layout per tensor, matching the existing ``bwdtaps`` convention::

    hp__<name>__<role>.raw    the tensor in its own dtype, little-endian, headerless
    hp__<name>__<role>.json   dtype/shape/order/sha256-of-raw/fp64 moments/note
    hp__<name>__<role>.npy    source-dtype copy, only for tensors below the size limit

``INDEX.json`` collects one entry per published name; names that hold exactly the same bytes
(the MF-side spellings) are listed as ``alias_of`` entries and add no file.
"""

import json
import shutil
import sys
from pathlib import Path

#: files above this many bytes are copied only when the destination has this much headroom
BIG = 512 << 20
HEADROOM = 6 << 30


def _name_role(key):
    name, _, role = key.rpartition('__')
    return name, role


def _copy(source, destination, copied, skipped):
    """Copy ``source`` next to the index unless it is big and the filesystem is nearly full."""
    if not source.exists():
        return 'missing'
    if source.stat().st_size > BIG:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(destination.parent).free < source.stat().st_size + HEADROOM:
            skipped.append(source.name)
            print(f'[MTPTAP] {source.name} kept on the source filesystem '
                  f'({source.stat().st_size / 2**30:.2f} GiB, destination nearly full)',
                  flush=True)
            return 'deferred'
    shutil.copy2(source, destination)
    copied.append(destination.name)
    return 'copied'


def _load_existing(dest):
    """Return the previously published tap entries, keyed by ``(name, role)``.

    Merging - rather than replacing - keeps the taps of the earlier HP runs visible in the same
    index.  The first HP publication wrote its entries to ``manifest.json`` instead of
    ``INDEX.json``, so those are folded in as well.
    """
    known = {}
    index_path = dest / 'INDEX.json'
    if index_path.exists():
        try:
            for entry in json.loads(index_path.read_text()).get('taps', []):
                known[(entry.get('name'), entry.get('role'))] = entry
        except (ValueError, OSError):
            pass
    manifest_path = dest / 'manifest.json'
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (ValueError, OSError):
            manifest = {}
        for key, meta in (manifest.get('tensors') or {}).items():
            name, _, role = key.rpartition('__')
            known.setdefault((name, role), {
                'name': name, 'role': role, 'key': key, 'from': 'manifest.json',
                'files': {'raw': meta.get('raw_file')},
                'dtype': meta.get('dtype'), 'shape': meta.get('shape'),
                'numel': meta.get('numel'), 'order': meta.get('order'),
                'sha256_of_raw_bytes': meta.get('sha256'),
                'fp64_ssum': meta.get('fp64_ssum'), 'min': meta.get('min'),
                'max': meta.get('max'), 'absmax': meta.get('absmax'),
                'note': 'published by the earlier HP backward sweep (see manifest.json)'})
    return known


def publish(run_dir, destination, side='hp', tag='mtpchain'):
    """Publish every recorded tensor of ``run_dir`` and refresh ``INDEX.json``."""
    run = Path(run_dir) / 'bwd'
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    source = json.loads((run / 'index.json').read_text())
    tensors, aliases = source['tensors'], source['aliases']
    known = _load_existing(dest)
    copied, skipped, deferred = [], [], []
    taps = []
    for key in sorted(tensors):
        record = tensors[key]
        name, role = _name_role(key)
        base = f'{side}__{key}'
        files = {}
        for suffix in ('.raw', '.json', '.npy', '.head.npy'):
            origin = run / (key + suffix)
            if not origin.exists():
                continue
            target = dest / (base + suffix)
            status = _copy(origin, target, copied, skipped)
            files[suffix.lstrip('.')] = str(origin) if status == 'deferred' else target.name
            if status == 'deferred':
                deferred.append(str(target))
        entry = {'name': name, 'role': role, 'key': key, 'files': files,
                 'dtype': record['dtype'], 'shape': record['shape'],
                 'numel': record['numel'], 'order': record['order'],
                 'sha256_of_raw_bytes': record['sha256_of_raw_bytes'],
                 'sha256_convention': 'sha256 over the raw original-dtype payload bytes '
                                      '(bf16: 2 bytes/element, fp32: 4 bytes/element, '
                                      'little-endian, headerless)',
                 'raw_bytes': record['raw_bytes'], 'fp64_ssum': record['fp64_ssum'],
                 'fp64_rms': record['fp64_rms'], 'min': record['min'], 'max': record['max'],
                 'absmax': record['absmax'], 'note': record['note'],
                 'hp_tap': key, 'device': record['device'], 'from': tag}
        if record.get('reconstruction'):
            entry['reconstruction'] = record['reconstruction']
        known[(name, role)] = entry
    for key in sorted(aliases):
        if key not in tensors:
            continue
        for alias in aliases[key]:
            name, role = _name_role(alias)
            known.setdefault((name, role), {
                'name': name, 'role': role, 'key': alias, 'alias_of': f'{side}__{key}',
                'files': {suffix.lstrip('.'): f'{side}__{key}.{suffix.lstrip(".")}'
                          for suffix in ('.raw', '.json')},
                'dtype': tensors[key]['dtype'], 'shape': tensors[key]['shape'],
                'numel': tensors[key]['numel'],
                'sha256_of_raw_bytes': tensors[key]['sha256_of_raw_bytes'],
                'fp64_ssum': tensors[key]['fp64_ssum'],
                'fp64_rms': tensors[key]['fp64_rms'],
                'note': f'identical bytes to {side}__{key} ({tensors[key]["note"]})',
                'from': tag})
    taps = [known[key] for key in sorted(known)]
    index = {'container': 'hp-910b-arrow', 'side': side, 'run': str(run_dir),
             'run_tag': tag,
             'protocol': 'HP single step 301, both alignment fixes (CLOSE_MATMUL_K_SHIFT popped '
                         'after trainer construction, loss summed as (lm+aux)+mtp); '
                         'mtp_taps attached to mtp.layers.0',
             'method': 'torch forward hooks and tensor/parameter gradient hooks (read-only) plus '
                       'a wrapper around projected_cross_entropy for the MTP head and CE',
             'naming': 'HP module mtp.layers.0.hnorm consumes the shared decoder output '
                       '(the tensor model.norm also consumes); mtp.layers.0.enorm consumes the '
                       'token-embedding branch',
             'merged_with': 'the earlier HP backward-sweep taps (see manifest.json)',
             'deferred': deferred, 'taps': taps}
    (dest / 'INDEX.json').write_text(json.dumps(index, indent=1, sort_keys=True) + '\n')
    print(f'[MTPTAP] published {len(copied)} files; index now holds {len(taps)} tap entries '
          f'({len(known)} names) in {dest}; skipped {len(skipped)} big file(s)', flush=True)
    return index


if __name__ == '__main__':
    publish(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else
            '/home/zth/tmp/align-cmp/bwdtaps/hp')
