#!/usr/bin/env bash
# CANN initialization scripts may read unset variables.
_hp_had_nounset=0
case $- in *u*) _hp_had_nounset=1; set +u ;; esac
export LD_LIBRARY_PATH="/usr/local/python3.12.14/lib:${LD_LIBRARY_PATH:-}"
source /workspace/Ascend/cann/set_env.sh
export PATH="/workspace/venv/bin:/usr/local/python3.12.14/bin:/usr/local/sbin:$PATH"
export HYPER_PARALLEL_PLATFORM=torch
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
if [[ $_hp_had_nounset == 1 ]]; then set -u; fi
unset _hp_had_nounset
export PYTHONPATH="/workspace${PYTHONPATH:+:$PYTHONPATH}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline

export PIP_NO_INDEX=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
