#!/usr/bin/env bash
# MuonFp32UpdateSwitch: run the alignment session (Fix 1-4) against the patched
# hyper_parallel overlay that adds the Muon/AdamW alignment switches.
set -euo pipefail
source /workspace/env.sh
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="/workspace/worktree/align-l2-f"
CONFIG="${CONFIG:-$SCRIPT_DIR/cfg-base.yaml}"
ENTRY="${ENTRY:-$SCRIPT_DIR/entry_mfs.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/dev/shm/mfs/runs}"
RUN_NAME="${RUN_NAME:-mfs-base}"
STEPS="${STEPS:-302}"
MASTER_PORT="${MASTER_PORT:-31691}"
RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
test ! -e "$RUN_DIR" || { echo "Output already exists: $RUN_DIR" >&2; exit 2; }
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/launcher.log") 2>&1
trap 'result=$?; printf "%s\n" "$result" > "$RUN_DIR/EXIT_CODE"; exit "$result"' EXIT
cd "$WORKTREE"
export PYTHONPATH="$SCRIPT_DIR:$WORKTREE/examples/training_demo/align-l2-f:$WORKTREE:/alignment/tools:$SCRIPT_DIR:/workspace/align-work/MTPChainHP${PYTHONPATH:+:$PYTHONPATH}"
export JT_RUN_DIR="$RUN_DIR"
export PR180_FIX5=0
export PR180_FP64=0
export PR180_TAPS=0
export PR180_TAG="$RUN_NAME"
if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  export ASCEND_RT_VISIBLE_DEVICES="$(
    /workspace/venv/bin/python /workspace/align-work/MTPChainHP/pick_card_mtp.py \
      "${NEED_MB:-30000}" 3600 | tail -n 1)"
fi
echo "[run] card=${ASCEND_RT_VISIBLE_DEVICES} run=${RUN_DIR} config=${CONFIG}" >&2
command=(/workspace/venv/bin/torchrun --nnodes=1 --nproc_per_node=1
  --master_addr=127.0.0.1 --master_port="$MASTER_PORT"
  --log-dir="$RUN_DIR/ranks" --redirects=3 --tee=3
  "$ENTRY" "$CONFIG"
  "--training.train_iters=$STEPS"
  "--training.global_batch_size=1"
  "--accelerator.tp_size=1"
  "--accelerator.ep_size=1"
  "--accelerator.sequence_parallel=false"
  "--accelerator.loss_parallel=false"
  "--checkpoint.checkpoint_dir=$RUN_DIR/checkpoints")
printf '%q ' "${command[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
"${command[@]}"
printf '{"run_dir":"%s","entry":"%s","steps":%s,"metrics":"%s/metrics-pr180.jsonl"}\n' \
  "$RUN_DIR" "$ENTRY" "$STEPS" "$RUN_DIR" > "$RUN_DIR/SUCCESS.json"
printf 'MuonFp32UpdateSwitch training complete: %s\n' "$RUN_DIR"
