#!/usr/bin/env bash
set -euo pipefail
source /workspace/env.sh
WORKTREE=/workspace/worktree/align-prs
SCRIPT_DIR=/workspace/align-work/MuonFp32UpdateSwitch
CONFIG=/workspace/align-work/MuonFp32UpdateSwitch/cfg-mf-ns2.yaml
RUN_DIR=/dev/shm/mfs/runs/pr-merge-301
ENTRY=$SCRIPT_DIR/entry_mfs.py
mkdir -p "$RUN_DIR"
cd "$WORKTREE"
export PYTHONPATH="$SCRIPT_DIR:$WORKTREE/examples/training_demo/align-l2-f:$WORKTREE:/alignment/tools:$SCRIPT_DIR:${PYTHONPATH:-}"
export JT_RUN_DIR="$RUN_DIR"
export MF_NS_DUMP=/dev/shm/mfs/nscap/prtest
export ASCEND_RT_VISIBLE_DEVICES=0
export PR180_FIX5=0 PR180_FP64=0 PR180_TAPS=0 PR180_TAG=pr-merge-301
exec /workspace/venv/bin/torchrun --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=31761 --log-dir="$RUN_DIR/ranks" --redirects=3 --tee=3 "$ENTRY" "$CONFIG" "--training.train_iters=301" "--training.global_batch_size=1" "--accelerator.tp_size=1" "--accelerator.ep_size=1" "--accelerator.sequence_parallel=false" "--accelerator.loss_parallel=false" "--checkpoint.checkpoint_dir=$RUN_DIR/checkpoints"
