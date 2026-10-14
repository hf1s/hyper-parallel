#!/usr/bin/env bash
# Re-run the two states whose first attempt hit NPU OOM (another tenant grabbed the chip).
set -uo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
export OUTPUT_ROOT=/dev/shm/mfs/runs
attempt() {
  local name="$1" cfg="$2" port="$3"
  for tries in 1 2 3; do
    echo "=== $name attempt $tries ($cfg) ==="
    rm -rf "$OUTPUT_ROOT/$name" "/dev/shm/mfs/params-$name"
    CONFIG="$PWD/$cfg" RUN_NAME="$name" MASTER_PORT="$port" \
      PR180_PARAM_DUMP="/dev/shm/mfs/params-$name" ./run.sh && return 0
    echo "=== $name attempt $tries FAILED ==="
    sleep 20
  done
  return 1
}
attempt sw-b cfg-b.yaml 31693 || echo "SW_B_FAILED"
attempt sw-c cfg-c.yaml 31694 || echo "SW_C_FAILED"
echo "=== retries done ==="
