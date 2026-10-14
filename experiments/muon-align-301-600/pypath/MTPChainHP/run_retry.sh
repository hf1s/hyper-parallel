#!/usr/bin/env bash
# Run the tapped step 301 repeatedly, each attempt on a freshly picked idle card, until one
# succeeds.  The eight cards are shared with the other agents, so an attempt can lose the race
# between the pick and the first allocation and die with an NPU OOM; a new one is cheap.
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/dev/shm/mtpchainhp
TAG="${TAG:-base}"
ATTEMPTS="${ATTEMPTS:-8}"
NEED_MB="${NEED_MB:-30000}"
mkdir -p "$ROOT"
for attempt in $(seq 1 "$ATTEMPTS"); do
  name="${TAG}${attempt}"
  card="$(/workspace/venv/bin/python "$SCRIPT_DIR/pick_card_mtp.py" "$NEED_MB" 900 | tail -n 1)"
  echo "[retry] attempt=$attempt card=$card run=$name" >&2
  rm -f "$ROOT/runs/$name/EXIT_CODE" 2>/dev/null
  (
    cd "$SCRIPT_DIR" || exit 1
    RUN_NAME="$name" ASCEND_RT_VISIBLE_DEVICES="$card" NEED_MB="$NEED_MB" bash run.sh
  ) > "$ROOT/$name.out" 2>&1
  code="$(cat "$ROOT/runs/$name/EXIT_CODE" 2>/dev/null || echo 1)"
  echo "[retry] attempt=$attempt card=$card rc=$code" >&2
  if [[ "$code" == "0" ]]; then
    printf '%s\n' "$name" > "$ROOT/last-ok.txt"
    exit 0
  fi
  sleep 20
done
echo "[retry] all attempts failed" >&2
exit 1
