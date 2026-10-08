#!/bin/bash
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
OUTPUT_DIR="${PROJECT_ROOT}/output/training_demo/jt_deepseek_v3"
RECIPE_NAME=${JT_RECIPE_NAME:-jt_deepseek_v3.yaml}
RECIPE_BASENAME=${RECIPE_NAME##*/}
CONFIG_FILE="${SCRIPT_DIR}/${RECIPE_NAME}"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 /path/to/jt_initial_weights /path/to/supervised_prefix [trainer overrides...]" >&2
    echo "Set JT_RECIPE_NAME to select a recipe in ${SCRIPT_DIR}." >&2
    exit 1
fi
REFERENCE_WEIGHTS=$1
DATA_PATH=$2
shift 2

if [[ ! -e "${REFERENCE_WEIGHTS}" ]]; then
    echo "Reference weights path does not exist: ${REFERENCE_WEIGHTS}" >&2
    exit 1
fi
for field in tokens labels loss_mask; do
    for extension in bin idx; do
        if [[ ! -s "${DATA_PATH}.${field}.${extension}" ]]; then
            echo "Indexed dataset file does not exist or is empty: ${DATA_PATH}.${field}.${extension}" >&2
            exit 1
        fi
    done
done
if [[ ! -s "${CONFIG_FILE}" ]]; then
    echo "Recipe does not exist: ${CONFIG_FILE}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

torchrun \
    --standalone \
    --nproc_per_node=8 \
    --module examples.training_demo.train_text \
    "${CONFIG_FILE}" \
    --model.reference_weights="${REFERENCE_WEIGHTS}" \
    --dataset.data_path="${DATA_PATH}" \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/run_${RECIPE_BASENAME%.yaml}.log"
