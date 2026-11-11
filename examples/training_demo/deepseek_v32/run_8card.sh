#!/bin/bash
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

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/output/training_demo/deepseek_v32"}
PYTHON_BIN=${PYTHON_BIN:-python}

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 /path/to/DeepSeek-V3.2-crop /path/to/data_prefix [trainer overrides...]" >&2
    exit 1
fi

# Resolve input paths before switching to the repository root.
MODEL_PATH=$(cd "$1" 2>/dev/null && pwd) || {
    echo "Model directory does not exist: $1" >&2
    exit 1
}
DATA_PREFIX=$2
shift 2
if [[ "${DATA_PREFIX}" != /* ]]; then
    DATA_PREFIX="${PWD}/${DATA_PREFIX}"
fi
if [[ ! -s "${DATA_PREFIX}.bin" || ! -s "${DATA_PREFIX}.idx" ]]; then
    echo "Indexed Dataset files are missing: ${DATA_PREFIX}.bin/.idx" >&2
    exit 1
fi
if [[ ! -s "${MODEL_PATH}/config.json" || ! -s "${MODEL_PATH}/tokenizer.json" ]] \
        || ! compgen -G "${MODEL_PATH}/model*.safetensors" > /dev/null; then
    echo "Cropped DeepSeek-V3.2 checkpoint is incomplete: ${MODEL_PATH}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"
# Launch with the YAML settings; trailing Trainer arguments override its defaults.
"${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=8 \
    --module examples.training_demo.train_text \
    "${SCRIPT_DIR}/train_8card.yaml" \
    --model.pretrained_model_name_or_path="${MODEL_PATH}" \
    --dataset.model_assets.tokenizer.pretrained_model_name_or_path="${MODEL_PATH}" \
    --dataset.data_path="${DATA_PREFIX}" \
    --checkpoint.checkpoint_dir="${OUTPUT_DIR}/checkpoints_8card" \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/run_8card.log"
