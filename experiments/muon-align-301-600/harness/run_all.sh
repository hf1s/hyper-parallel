#!/usr/bin/env bash
# Run the four Muon-switch states back to back, each 301 -> 302 with a native fp32 parameter dump.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
export OUTPUT_ROOT=/dev/shm/mfs/runs
run_state() {
  local name="$1" cfg="$2" port="$3"
  echo "=== $name ($cfg) ==="
  CONFIG="$PWD/$cfg" RUN_NAME="$name" MASTER_PORT="$port" \
    PR180_PARAM_DUMP="/dev/shm/mfs/params-$name" ./run.sh
}
run_state base cfg-base.yaml 31691
run_state sw-a cfg-a.yaml 31692
run_state sw-b cfg-b.yaml 31693
run_state sw-c cfg-c.yaml 31694
echo "=== all states done ==="
