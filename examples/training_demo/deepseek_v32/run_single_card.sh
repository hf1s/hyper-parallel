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

# Keep generated data and run outputs under the repository output directory.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/output/training_demo/deepseek_v32"}
DEFAULT_DATA_PATH="${OUTPUT_DIR}/single_card.jsonl"
PYTHON_BIN=${PYTHON_BIN:-python}

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 /path/to/DeepSeek-V3.2 [path/to/train.jsonl] [trainer overrides...]" >&2
    exit 1
fi

# Resolve input paths before switching to the repository root.
MODEL_PATH=$1
shift
DATA_PATH="${DEFAULT_DATA_PATH}"
if [[ $# -gt 0 && ${1} != --* ]]; then
    DATA_PATH=$1
    if [[ ! -s "${DATA_PATH}" ]]; then
        echo "Text JSONL data is missing or empty: ${DATA_PATH}" >&2
        exit 1
    fi
    shift
fi
MODEL_PATH=$(cd "${MODEL_PATH}" 2>/dev/null && pwd) || {
    echo "Model directory does not exist: ${MODEL_PATH}" >&2
    exit 1
}
if [[ "${DATA_PATH}" != /* ]]; then
    DATA_PATH="${PWD}/${DATA_PATH}"
fi
if [[ ! -s "${MODEL_PATH}/config.json" || ! -s "${MODEL_PATH}/tokenizer.json" ]]; then
    echo "Local DeepSeek-V3.2 config/tokenizer assets are incomplete: ${MODEL_PATH}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"
# Generate deterministic text samples when the default dataset is absent.
if [[ ! -s "${DATA_PATH}" ]]; then
    "${PYTHON_BIN}" -m examples.training_demo.deepseek_v32.prepare_single_card_data \
        --output "${DATA_PATH}" \
        --tokenizer-path "${MODEL_PATH}"
fi

# Launch with the YAML settings; trailing Trainer arguments override its defaults.
"${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=1 \
    --module examples.training_demo.train_text \
    "${SCRIPT_DIR}/train_single_card.yaml" \
    --model.config_path="${MODEL_PATH}" \
    --dataset.model_assets.tokenizer.pretrained_model_name_or_path="${MODEL_PATH}" \
    --dataset.data_path="${DATA_PATH}" \
    --checkpoint.checkpoint_dir="${OUTPUT_DIR}/checkpoints" \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/run_single_card.log"
