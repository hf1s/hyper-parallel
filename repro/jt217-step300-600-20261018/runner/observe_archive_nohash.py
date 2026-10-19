from __future__ import annotations

import argparse
import json
from pathlib import Path

from hyper_parallel.trainer.callbacks.base import Callback
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.archived import create_archived_trainer


class Metrics(Callback):
    def __init__(self, trainer, output: Path, offset: int) -> None:
        super().__init__(trainer)
        self.output = output
        self.offset = offset
        self.output.mkdir(parents=True, exist_ok=False)
        self.metrics = output / "metrics.jsonl"

    def on_step_end(self, state, *, loss, loss_dict, grad_norm, **kwargs) -> None:
        del kwargs
        record = {
            "step": int(state.global_step),
            "loss": float(loss),
            "components": {name: float(value) for name, value in loss_dict.items()},
            "grad_norm": float(grad_norm),
        }
        with self.metrics.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print("[CLEAN-HP-NOHASH] " + json.dumps(record), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=300)
    args, overrides = parser.parse_known_args()
    trainer = create_archived_trainer(parse_training_args([args.config, *overrides]))
    trainer.base._callbacks.append(Metrics(trainer.base, args.output, args.offset))
    trainer.train()
    steps = [int(json.loads(line)["step"]) for line in (args.output / "metrics.jsonl").read_text().splitlines()]
    expected = list(range(args.offset + 1, trainer.base.train_iters + 1))
    if steps != expected:
        raise ValueError(f"Incomplete continuation: first={steps[:1]} last={steps[-1:]}")
    (args.output / "SUCCESS.json").write_text(json.dumps({"steps": len(steps), "first": steps[0], "last": steps[-1]}) + "\n")


if __name__ == "__main__":
    main()
