#!/usr/bin/env bash
# Line 2: clean B0 worktree plus the full alignment overlay, resumed at step 300.
set -euo pipefail
source /workspace/env.sh
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
CONFIG="${CONFIG:-$SCRIPT_DIR/train-301.yaml}"
ENTRY="${ENTRY:-$SCRIPT_DIR/train_session.py}"
WORKER_NUM="${WORKER_NUM:-1}"
STEPS="${STEPS:-301}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/align-work/Line2F301}"
RUN_NAME="${RUN_NAME:-run}"
MASTER_PORT="${MASTER_PORT:-30301}"
TP_SIZE="${TP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-false}"
LOSS_PARALLEL="${LOSS_PARALLEL:-false}"
RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
test ! -e "$RUN_DIR" || { echo "Output already exists: $RUN_DIR" >&2; exit 2; }
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/launcher.log") 2>&1
trap 'result=$?; printf "%s\n" "$result" > "$RUN_DIR/EXIT_CODE"; exit "$result"' EXIT
cd "$WORKTREE_ROOT"
export PYTHONPATH="$WORKTREE_ROOT:/alignment/tools:/workspace${PYTHONPATH:+:$PYTHONPATH}"
export JT_RUN_DIR="$RUN_DIR"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-2}"
command=(/workspace/venv/bin/torchrun --nnodes=1 --nproc_per_node="$WORKER_NUM"
  --master_addr=127.0.0.1 --master_port="$MASTER_PORT"
  --log-dir="$RUN_DIR/ranks" --redirects=3 --tee=3
  "$ENTRY" "$CONFIG"
  "--training.train_iters=$STEPS"
  "--training.global_batch_size=$DP_SIZE"
  "--accelerator.tp_size=$TP_SIZE"
  "--accelerator.ep_size=$EP_SIZE"
  "--accelerator.sequence_parallel=$SEQUENCE_PARALLEL"
  "--accelerator.loss_parallel=$LOSS_PARALLEL"
  "--checkpoint.checkpoint_dir=$RUN_DIR/checkpoints")
printf '%q ' "${command[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
"${command[@]}"
/workspace/venv/bin/python - "$RUN_DIR/launcher.log" "$RUN_DIR/metrics-parsed-rank0.jsonl" "$RUN_DIR/factor-report.json" <<'PY'
import json
import re
import sys
from pathlib import Path
source, target, factors = map(Path, sys.argv[1:])
number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
field = re.compile(r"(loss|grad_norm|foundation_loss/(?:lm|mtp|aux))=(" + number + ")")
step = re.compile(r"(?:\|\s*step=|Training:\s+.*?\|\s*)(\d+)")
rows = {}
for line in source.read_text(errors="replace").splitlines():
    match = step.search(line)
    if not match:
        continue
    values = dict(field.findall(line))
    if not {"loss", "grad_norm", "foundation_loss/lm", "foundation_loss/mtp"}.issubset(values):
        continue
    rows[int(match.group(1))] = {
        "step": int(match.group(1)),
        "loss": float(values["loss"]),
        "grad_norm": float(values["grad_norm"]),
        "loss_dict": {key: float(values[key]) for key in values if key.startswith("foundation_loss/")},
    }
if not rows:
    raise SystemExit(f"no per-step metrics parsed from {source}")
target.write_text("".join(json.dumps(rows[key]) + "\n" for key in sorted(rows)))
print(f"Wrote {len(rows)} metrics to {target}")
print("last step:", json.dumps(rows[max(rows)]))
if factors.exists():
    report = json.loads(factors.read_text())
    print("factor replacements:", json.dumps({row["factor"]: row["replaced"] for row in report["factors"]}))
    print("total replaced:", report["total_replaced"], "no_module_swap:", json.dumps(report["no_module_swap"]))
else:
    print(f"factor report missing: {factors}")
norms = target.parent / "grad_norms.jsonl"
print("grad_norms.jsonl:", norms.read_text().strip() if norms.exists() else "MISSING")
for pattern in ("grad_hashes.jsonl", "grad_hashes/step-*.jsonl", "step*_param_hashes.jsonl", "name_mapping.json"):
    for path in sorted(target.parent.glob(pattern)):
        lines = sum(1 for _ in path.open())
        print(f"{path.name}: {lines} lines" if path.suffix == ".jsonl" else f"{path.name}: present")
callback = target.parent / "metrics-rank0.jsonl"
print("metrics-rank0.jsonl:", callback.read_text().strip() if callback.exists() else "MISSING")
anchors = target.parent / "anchors_step301.json"
if anchors.exists():
    payload = json.loads(anchors.read_text())
    for row in payload["anchors"]:
        print(f"anchor {row['anchor']} {row['name']} {row['dtype']} {row['shape']} {row['sha256']}")
    for row in payload["attention_max_logits"]:
        print(f"max-logit {row['name']} {row['sha256']}")
else:
    print("anchors_step301.json: MISSING")
PY
printf '{"worktree":"%s","steps":%s,"metrics":"%s","metrics_parsed":"%s","factor_report":"%s","anchors":"%s"}\n' \
  "$WORKTREE_ROOT" "$STEPS" "$RUN_DIR/metrics-rank0.jsonl" "$RUN_DIR/metrics-parsed-rank0.jsonl" \
  "$RUN_DIR/factor-report.json" "$RUN_DIR/anchors_step301.json" > "$RUN_DIR/SUCCESS.json"
printf 'Training complete: %s\n' "$RUN_DIR"
