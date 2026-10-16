#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/zth-jt-precision-scripts
RUN="${1:?usage: run_hp_two_steps.sh RUN_DIRECTORY}"
case "$RUN" in "$ROOT"/alignment/*) ;; *) echo "Run directory must be under $ROOT/alignment" >&2; exit 2;; esac
test ! -e "$RUN/launcher.log"
source /workspace/env-850.sh
export PYTHONPATH="$RUN/source:$RUN/tools"
export PYTHONDONTWRITEBYTECODE=1
export ASCEND_RT_VISIBLE_DEVICES="${HP_DEVICE:-6}"
export TMPDIR="$RUN/tmp" HOME="$RUN/home" XDG_CACHE_HOME="$RUN/cache"
export HF_HOME="$RUN/cache/huggingface" HF_DATASETS_CACHE="$RUN/cache/datasets"
export TORCH_HOME="$RUN/cache/torch"
export ASCEND_CACHE_PATH="$RUN/cache/ascend" ASCEND_WORK_PATH="$RUN/cache/ascend"
export ASCEND_PROCESS_LOG_PATH="$RUN/logs/ascend" MS_OM_PATH="$RUN/logs/om"
export HP_OBSERVATION_ROOT="$RUN/hp-observation"
export MF_CAP=0 MF_CAP_SCALARS=0 MF_CAP_ANCHORS=0 MF_CAP_SUB=0
unset JT_GRAD_TAP MF_BOUNDARY_TAP MF_NS_TRACE PR180_NS_DUMP JT_STEP302_CAPTURE_DIR
mkdir -p "$TMPDIR" "$HOME" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$ASCEND_CACHE_PATH" "$ASCEND_PROCESS_LOG_PATH" "$MS_OM_PATH"
exec > >(tee "$RUN/launcher.log") 2>&1
trap 'result=$?; printf "%s\n" "$result" > "$RUN/EXIT_CODE"; exit "$result"' EXIT
cd "$RUN/source"
command=("$ROOT/tools/env/venv/bin/python" -B -m torch.distributed.run
  --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29971
  --log-dir="$RUN/ranks" --redirects=3 --tee=3
  "$RUN/tools/observe_hp_two_steps.py"
  "$RUN/source/examples/training_demo/jt_deepseek_v3/mf-alignment-2step.yaml"
  --training.train_iters=302 --training.global_batch_size=1
  --accelerator.tp_size=1 --accelerator.ep_size=1
  --accelerator.sequence_parallel=false --accelerator.loss_parallel=false
  "--model.reference_weights=$ROOT/reference/weights/full-state-step300"
  "--dataset.source_dataset.dataset_path=$ROOT/reference/data/hp-step300"
  "--checkpoint.restore_from=$ROOT/reference/weights/native-step300/global_step_300"
  "--checkpoint.checkpoint_dir=$RUN/checkpoints")
printf '%q ' "${command[@]}" > "$RUN/command.txt"
printf '\n' >> "$RUN/command.txt"
printf 'HP_TWO_STEPS_START device=%s CANN=%s run=%s\n' "$ASCEND_RT_VISIBLE_DEVICES" "$ASCEND_HOME_PATH" "$RUN"
"${command[@]}"
