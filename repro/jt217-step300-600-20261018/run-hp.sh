#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/inputs/paths.env"
if [[ -d "$ROOT/source/hyper_parallel" ]]; then
  SOURCE_ROOT="$ROOT/source"
elif [[ -d "$ROOT/../source/hyper_parallel" ]]; then
  SOURCE_ROOT="$ROOT/../source"
elif [[ -d "$ROOT/../../hyper_parallel" ]]; then
  SOURCE_ROOT="$ROOT/../.."
else
  echo "Cannot locate exact source tree" >&2
  exit 2
fi
source /workspace/env.sh
export CLOSE_MATMUL_K_SHIFT=0
export PYTHONPATH="$SOURCE_ROOT:/workspace/Ascend/cann-9.2.0-beta.2/python/site-packages${PYTHONPATH:+:$PYTHONPATH}"

if [[ -e "$ROOT/output/hp/SUCCESS.json" ]]; then
  echo "Refusing to overwrite completed output: $ROOT/output/hp" >&2
  exit 2
fi

mkdir -p "$ROOT/output"
exec python -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port=31990 \
  "$ROOT/runner/observe_archive_nohash.py" \
  --config "$ROOT/config/hp-archive200-r2.yaml" \
  --output "$ROOT/output/hp" \
  --offset 300 \
  --training.train_iters=600 \
  --model.reference_weights="$JT217_MODEL_STATE" \
  --dataset.source_dataset.dataset_path="$JT217_DATASET" \
  --checkpoint.restore_from="$JT217_NATIVE_STATE" \
  --checkpoint.checkpoint_dir="$ROOT/output/checkpoints"
