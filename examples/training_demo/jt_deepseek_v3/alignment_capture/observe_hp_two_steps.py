"""Observe the requested HP steps 301/302; save only step301 pre-clip gradients."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from capture_io import CaptureWriter, record_cann_environment
from hyper_parallel.core.utils.clip_grad import _get_local_grad
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.text_trainer import TextTrainer


def main() -> None:
    root = Path(os.environ["HP_OBSERVATION_ROOT"])
    root.mkdir(parents=True, exist_ok=False)
    trainer = TextTrainer(parse_training_args())
    base = trainer.base
    record_cann_environment(root, "hp-initialized")
    original_postforward = base.postforward
    original_prepare = base.prepare_optimizer_step
    original_train_step = trainer.train_step
    latest_losses = {}
    completed_steps = []

    with CaptureWriter(root, backend="hp", rank=base.global_rank) as writer, \
            (root / "metrics.jsonl").open("x", encoding="utf-8") as metrics:
        def postforward(outputs, labels):
            loss, losses = original_postforward(outputs, labels)
            step = base.state.global_step + 1
            latest_losses.clear()
            latest_losses.update({name: float(value.detach()) for name, value in losses.items()})
            if step == 301:
                writer.write_tensor(step, "loss", "loss", "loss", loss.detach())
                for name, value in losses.items():
                    suffix = name.rsplit("/", 1)[-1]
                    canonical = {"lm": "lm_loss", "mtp": "mtp_loss", "aux": "aux_loss"}.get(suffix, name)
                    writer.write_tensor(step, "loss", "loss", canonical, value.detach())
            return loss, losses

        def prepare_optimizer_step():
            step = base.state.global_step + 1
            if step == 301:
                count = 0
                for name, parameter in base.model.named_parameters():
                    gradient = _get_local_grad(parameter)
                    if gradient is None:
                        continue
                    canonical = getattr(parameter, "model_name", name).replace("_checkpoint_wrapped_module.", "")
                    writer.write_tensor(step, "post_backward", "grad", canonical, gradient,
                                        layout="hp_native")
                    count += 1
                    print(f"HP_STEP301_GRADIENT {count} {canonical}", flush=True)
                print(f"HP_STEP301_GRADIENTS_COMPLETE count={count}", flush=True)
            norm = original_prepare()
            if step == 301:
                if torch.is_tensor(norm):
                    writer.write_tensor(step, "post_clip", "scalar", "total_norm", norm.detach())
                else:
                    writer.write_array(step, "post_clip", "scalar", "total_norm", norm)
            return norm

        def train_step(iterator):
            if not completed_steps:
                if base.state.global_step != 300 or base.train_iters != 302:
                    raise RuntimeError("This run must restore step300 and finish at step302")
                print("HP_CHECKPOINT_RESTORED step=300 target=302", flush=True)
            result = original_train_step(iterator)
            row = {"step": base.state.global_step,
                   "loss": float(result["loss"]), "grad_norm": float(result["grad_norm"]),
                   "loss_dict": dict(latest_losses)}
            completed_steps.append(base.state.global_step)
            metrics.write(json.dumps(row, allow_nan=False) + "\n")
            metrics.flush()
            print("HP_STEP_COMPLETE " + json.dumps(row, allow_nan=False), flush=True)
            return result

        base.postforward = postforward
        base.prepare_optimizer_step = prepare_optimizer_step
        trainer.train_step = train_step
        trainer.train()
        if completed_steps != [301, 302]:
            raise RuntimeError(f"Expected completed steps [301, 302], got {completed_steps}")
        print(f"HP_TWO_STEPS_COMPLETE tensor_bytes={writer.tensor_bytes}", flush=True)


if __name__ == "__main__":
    main()
