#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$ROOT/output/hp/SUCCESS.json" ]]; then
  echo "Missing completed HP run: $ROOT/output/hp/SUCCESS.json" >&2
  exit 2
fi

mkdir -p "$ROOT/output/compare"
python "$ROOT/tools/convert_metrics.py" \
  "$ROOT/output/hp/metrics.jsonl" \
  "$ROOT/output/hp/metrics-rank0.jsonl"
python "$ROOT/tools/compare_hp_mf.py" \
  --mf "$ROOT/reference/mf-reference.info.log" \
  --run hp="$ROOT/output/hp" \
  --steps 301-600 \
  --out "$ROOT/output/compare" \
  --first 25 \
  --smooth 50 \
  --early 50
