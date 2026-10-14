#!/usr/bin/env bash
# MTPChainHP: one alignment step 301 with both known fixes plus operator-boundary MTP taps.
#
# Same launcher shape as BackwardSweep/run.sh but the entry point is this directory's
# entry_mtpchain.py, the run directory lives on /dev/shm (the container's / has almost no room),
# and the card is chosen with the shared waitcard helper.
set -euo pipefail
source /workspace/env.sh
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="/workspace/worktree/align-l2-f"
CONFIG="${CONFIG:-$WORKTREE_ROOT/examples/training_demo/align-l2-f/train-301.yaml}"
ENTRY="${ENTRY:-$SCRIPT_DIR/entry_mtpchain.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/dev/shm/mtpchainhp/runs}"
RUN_NAME="${RUN_NAME:-base}"
MASTER_PORT="${MASTER_PORT:-$(( 31000 + $(printf '%s' "$RUN_NAME" | cksum | cut -d" " -f1) % 3000 ))}"
RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
test ! -e "$RUN_DIR" || { echo "Output already exists: $RUN_DIR" >&2; exit 2; }
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/launcher.log") 2>&1
trap 'result=$?; printf "%s\n" "$result" > "$RUN_DIR/EXIT_CODE"; exit "$result"' EXIT
cd "$WORKTREE_ROOT"
export PYTHONPATH="$WORKTREE_ROOT:/alignment/tools:$SCRIPT_DIR:/workspace${PYTHONPATH:+:$PYTHONPATH}"
export JT_RUN_DIR="$RUN_DIR"
if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  # the picker also rejects chips on which a dead process still holds device memory, and only
  # its last stdout line is the answer
  export ASCEND_RT_VISIBLE_DEVICES="$(
    /workspace/venv/bin/python "$SCRIPT_DIR/pick_card_mtp.py" "${NEED_MB:-30000}" 3600 | tail -n 1)"
fi
echo "[run] card=${ASCEND_RT_VISIBLE_DEVICES} run=${RUN_DIR}" >&2
command=(/workspace/venv/bin/torchrun --nnodes=1 --nproc_per_node=1
  --master_addr=127.0.0.1 --master_port="$MASTER_PORT"
  --log-dir="$RUN_DIR/ranks" --redirects=3 --tee=3
  "$ENTRY" "$CONFIG"
  "--training.train_iters=301"
  "--training.global_batch_size=1"
  "--accelerator.tp_size=1"
  "--accelerator.ep_size=1"
  "--accelerator.sequence_parallel=false"
  "--accelerator.loss_parallel=false"
  "--checkpoint.checkpoint_dir=$RUN_DIR/checkpoints")
printf '%q ' "${command[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
"${command[@]}"
