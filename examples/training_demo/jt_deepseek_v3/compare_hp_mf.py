#!/usr/bin/env python3
"""Compare HP runs with one MindFormers (MF) reference, step by step (HP - MF).

MF:  ``info.log`` (``launcher.log`` also works) with MF ``print_output_info`` lines:
     ``step:[  301/ 4708], loss: ..., lm_loss: ..., load_balancing_loss: ..., mtp_loss: ..., global_norm: [...]``
HP:  a run directory containing ``metrics-rank0.jsonl``. ``launcher.log`` in the same directory is optional;
     its ``data/step_tokens`` check that the HP runs read the same batches and weight the smoothing.

Example:
    python compare_hp_mf.py \
        --mf /workspace/reference/mf-4k-from-256k/info.log \
        --run b0=/workspace/outputs/align-b0 \
        --run ulp=/workspace/outputs/align-b0-ulp \
        --pair b0,ulp \
        --steps 301-600 --out /workspace/outputs/compare/b0

Writes summary.md, loss-delta.png, loss-smoothed.png and early.png to --out. Needs only matplotlib.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STEP_RE = re.compile(r"step:\[\s*(\d+)/\s*\d+\]")
NUMBER_RE = re.compile(r"(\w+): \[?([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)")
TOKENS_RE = re.compile(r"\| step=(\d+) .*?data/step_tokens=(\d+)")
HP_KEYS = {"lm": "foundation_loss/lm", "mtp": "foundation_loss/mtp"}
METRICS = ("total", "lm", "mtp")
TITLES = {"total": "total loss", "lm": "LM loss", "mtp": "MTP loss (x0.3)"}
COLORS = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#7f7f7f")
PRINT_TOLERANCE = 1e-6  # MF prints losses with six decimals


def load_mf(path: Path) -> dict[int, dict[str, float]]:
    """Per-step MF losses and global norm; a repeated step keeps its last line (resumed runs)."""
    out = {}
    for line in path.open(errors="replace"):
        match = STEP_RE.search(line)
        if not match or "lm_loss" not in line:
            continue
        values = dict(NUMBER_RE.findall(line[match.end():]))
        out[int(match.group(1))] = {"total": float(values["loss"]), "lm": float(values["lm_loss"]),
                                    "mtp": float(values["mtp_loss"]), "gn": float(values["global_norm"])}
    return out


def load_hp(run: Path) -> dict[int, dict[str, float]]:
    """Per-step HP losses and grad norm from ``metrics-rank0.jsonl``; a repeated step keeps its last record."""
    path = run / "metrics-rank0.jsonl" if run.is_dir() else run
    out = {}
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if "step" not in record or "loss" not in record:
            continue
        losses = record.get("loss_dict", {})
        out[int(record["step"])] = {"total": float(record["loss"]), "lm": float(losses[HP_KEYS["lm"]]),
                                    "mtp": float(losses[HP_KEYS["mtp"]]),
                                    "gn": float(record.get("grad_norm", "nan"))}
    return out


def load_tokens(run: Path) -> dict[int, int]:
    """Supervised tokens per step from the HP ``launcher.log`` next to the metrics, if present."""
    log = (run if run.is_dir() else run.parent) / "launcher.log"
    out = {}
    if log.exists():
        for line in log.open(errors="replace"):
            match = TOKENS_RE.search(line)
            if match:
                out[int(match.group(1))] = int(match.group(2))
    return out


def mean_abs(values: list[float]) -> float:
    return sum(abs(v) for v in values) / len(values) if values else float("nan")


def deltas(run: dict, reference: dict, metric: str, steps: list[int]) -> tuple[list[int], list[float]]:
    xs = [s for s in steps if s in run and s in reference]
    return xs, [run[s][metric] - reference[s][metric] for s in xs]


def run_summary(label: str, run: dict, mf: dict, steps: list[int], first: int) -> dict:
    xs, total = deltas(run, mf, "total", steps)
    if not xs:
        raise SystemExit(f"{label}: no step overlaps with MF in the requested range")
    head = xs[:first]
    leading_equal = 0
    for value in total:
        if abs(value) > PRINT_TOLERANCE:
            break
        leading_equal += 1
    row = {"label": label, "steps": f"{xs[0]}-{xs[-1]} ({len(xs)})", "leading_equal": leading_equal}
    for metric in METRICS:
        _, values = deltas(run, mf, metric, xs)
        row[f"first_{metric}"] = values[0]
        row[f"head_{metric}"] = mean_abs(values[:len(head)])
        row[f"all_{metric}"] = mean_abs(values)
    row["signed_total"] = sum(total) / len(total)
    row["max_total"] = max(abs(v) for v in total)
    row["over_5e-3"] = sum(abs(v) > 5e-3 for v in total)
    row["gn_ratio"] = statistics.median(run[s]["gn"] / mf[s]["gn"] for s in xs)
    return row


def pair_summary(name_a: str, a: dict, name_b: str, b: dict, steps: list[int], first: int,
                 tokens: dict[str, dict[int, int]]) -> dict:
    xs = [s for s in steps if s in a and s in b]
    if not xs:
        raise SystemExit(f"{name_a},{name_b}: no common steps")
    first_diff = next((s for s in xs if any(a[s][m] != b[s][m] for m in METRICS)), None)
    row = {"pair": f"{name_a} - {name_b}", "steps": f"{xs[0]}-{xs[-1]} ({len(xs)})",
           "first_diff": "none" if first_diff is None else str(first_diff)}
    for metric in METRICS:
        values = [a[s][metric] - b[s][metric] for s in xs]
        row[f"head_{metric}"] = mean_abs(values[:first])
        row[f"all_{metric}"] = mean_abs(values)
    ta, tb = tokens.get(name_a, {}), tokens.get(name_b, {})
    common = [s for s in xs if s in ta and s in tb]
    mismatch = [s for s in common if ta[s] != tb[s]]
    row["tokens"] = ("n/a" if not common else "same" if not mismatch
                     else f"DIFFER on {len(mismatch)} steps, first {mismatch[0]}")
    return row


def markdown(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    def cell(text: str) -> str:
        return text.replace("|", "\\|")

    lines = ["| " + " | ".join(cell(title) for _, title, _ in columns) + " |",
             "|" + "---|" * len(columns)]
    for row in rows:
        cells = [cell(format(row[key], fmt) if fmt else str(row[key])) for key, _, fmt in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def smoothed(run: dict, mf: dict, metric: str, steps: list[int], weights: dict[int, int], width: int):
    """Trailing ``width``-step mean of HP - MF, weighted by supervised tokens when known; full windows only."""
    available = set(steps) & set(run) & set(mf)
    xs, ys = [], []
    for end in steps:
        window = list(range(end - width + 1, end + 1))
        if window[0] < steps[0] or any(s not in available for s in window):
            continue
        w = [weights.get(s, 1) for s in window]
        xs.append(end)
        ys.append(sum((run[s][metric] - mf[s][metric]) * wi for s, wi in zip(window, w)) / sum(w))
    return xs, ys


def plot(runs: dict[str, dict], mf: dict, steps: list[int], weights: dict[int, int], args, out: Path) -> None:
    colors = {label: COLORS[i % len(COLORS)] for i, label in enumerate(runs)}
    fig, axes = plt.subplots(2, 2, figsize=(16, 9.5), sharex=True)
    xs = [s for s in steps if s in mf]
    axes[0][0].plot(xs, [mf[s]["total"] for s in xs], color="black", lw=1.2, label="MF")
    for label, run in runs.items():
        rx = [s for s in steps if s in run]
        axes[0][0].plot(rx, [run[s]["total"] for s in rx], color=colors[label], lw=0.9, label=label)
    axes[0][0].set_title("total loss")
    axes[0][0].legend(loc="upper right", fontsize=9)
    for ax, metric in zip((axes[0][1], axes[1][0], axes[1][1]), METRICS):
        for label, run in runs.items():
            dx, dy = deltas(run, mf, metric, steps)
            ax.plot(dx, dy, color=colors[label], lw=1.0, label=f"{label} - MF (mean|d| {mean_abs(dy):.4f})")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"d {TITLES[metric]} (HP - MF)")
        ax.legend(loc="upper left", fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("step")
    fig.suptitle(f"HP - MF per step ({steps[0]}-{steps[-1]})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "loss-delta.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for ax, metric in zip(axes, METRICS):
        for label, run in runs.items():
            sx, sy = smoothed(run, mf, metric, steps, weights, args.smooth)
            if sx:
                ax.plot(sx, sy, color=colors[label], lw=1.6, label=f"{label} - MF (mean|d| {mean_abs(sy):.4f})")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"d {TITLES[metric]}, trailing {args.smooth}-step mean"
                     + (" weighted by supervised tokens" if weights else ""))
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("window end step")
    fig.tight_layout()
    fig.savefig(out / "loss-smoothed.png", dpi=130)
    plt.close(fig)

    head = steps[:args.early]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, metric in zip(axes, ("lm", "mtp")):
        for label, run in runs.items():
            dx, dy = deltas(run, mf, metric, head)
            ax.plot(dx, dy, color=colors[label], lw=1.2, marker="o", ms=3,
                    label=f"{label} - MF (first {args.first} mean|d| {mean_abs(dy[:args.first]):.4f})")
        ax.axhline(0, color="black", lw=0.8)
        ax.axvspan(head[0] - 0.5, head[min(args.first, len(head)) - 1] + 0.5, color="#f2f2f2", zorder=0)
        ax.set_title(f"d {TITLES[metric]} (HP - MF), first {len(head)} steps")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel(f"step (shaded: first {args.first})")
    fig.tight_layout()
    fig.savefig(out / "early.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mf", type=Path, required=True, help="MF info.log")
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=PATH",
                        help="HP run directory (or its metrics-rank0.jsonl); repeat for several runs")
    parser.add_argument("--pair", action="append", default=[], metavar="A,B",
                        help="HP-HP comparison between two run labels, e.g. a noise-floor pair")
    parser.add_argument("--steps", help="inclusive range such as 301-600 (default: all MF steps)")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--first", type=int, default=25, help="length of the early window (default 25)")
    parser.add_argument("--smooth", type=int, default=50, help="smoothing window (default 50)")
    parser.add_argument("--early", type=int, default=50, help="steps shown in early.png (default 50)")
    args = parser.parse_args()

    mf = load_mf(args.mf)
    if not mf:
        raise SystemExit(f"no MF step lines found in {args.mf}")
    runs, tokens = {}, {}
    for spec in args.run:
        label, _, path = spec.partition("=")
        runs[label] = load_hp(Path(path))
        tokens[label] = load_tokens(Path(path))
    if args.steps:
        first_step, last_step = (int(v) for v in args.steps.split("-"))
        steps = list(range(first_step, last_step + 1))
    else:
        steps = sorted(mf)
    weights = next((t for t in tokens.values() if t), {})

    args.out.mkdir(parents=True, exist_ok=True)
    run_rows = [run_summary(label, run, mf, steps, args.first) for label, run in runs.items()]
    pair_rows = []
    for spec in args.pair:
        a, b = spec.split(",")
        pair_rows.append(pair_summary(a, runs[a], b, runs[b], steps, args.first, tokens))
    labelled = [label for label in runs if tokens[label]]
    token_note = "no HP launcher.log token lines found"
    if len(labelled) >= 2:
        ref_label = labelled[0]
        bad = [label for label in labelled[1:]
               if any(tokens[label][s] != tokens[ref_label][s] for s in steps
                      if s in tokens[label] and s in tokens[ref_label])]
        token_note = (f"supervised tokens per step identical across {', '.join(labelled)}" if not bad
                      else f"WARNING: supervised tokens differ from {ref_label} in {', '.join(bad)}")
    elif labelled:
        token_note = f"token counts from {labelled[0]} (only run with launcher.log)"

    first = args.first
    text = [f"# HP - MF, steps {steps[0]}-{steps[-1]}", "", f"MF: `{args.mf}`  ", f"Data check: {token_note}", "",
            "## Each run against MF", "",
            markdown(run_rows, [
                ("label", "run", ""), ("steps", "steps", ""),
                ("first_lm", "first dLM", "+.6f"), ("first_mtp", "first dMTP", "+.6f"),
                ("first_total", "first dtotal", "+.6f"), ("leading_equal", "leading steps |d|<=1e-6", ""),
                ("head_total", f"first {first} mean|dtotal|", ".4f"), ("all_total", "mean|dtotal|", ".4f"),
                ("all_lm", "mean|dLM|", ".4f"), ("all_mtp", "mean|dMTP|", ".4f"),
                ("signed_total", "mean dtotal", "+.4f"), ("max_total", "max|dtotal|", ".4f"),
                ("over_5e-3", "steps |dtotal|>5e-3", ""), ("gn_ratio", "grad-norm ratio median", ".4f"),
            ])]
    if pair_rows:
        text += ["", "## HP - HP pairs", "", markdown(pair_rows, [
            ("pair", "pair", ""), ("steps", "steps", ""), ("first_diff", "first differing step", ""),
            ("head_total", f"first {first} mean|dtotal|", ".4f"), ("all_total", "mean|dtotal|", ".4f"),
            ("all_lm", "mean|dLM|", ".4f"), ("all_mtp", "mean|dMTP|", ".4f"), ("tokens", "tokens", ""),
        ])]
    summary = "\n".join(text) + "\n"
    (args.out / "summary.md").write_text(summary)
    print(summary)
    plot(runs, mf, steps, weights, args, args.out)
    print(f"wrote {args.out}/summary.md, loss-delta.png, loss-smoothed.png, early.png")


if __name__ == "__main__":
    main()
