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
"""Real Gloo reductions check DP/CP accounting and TP replica exclusion."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F

from hyper_parallel.components.losses.mtp import calculate_mtp_loss
from hyper_parallel.models.jt_deepseek_v3.adapter.distributed.context_parallel import _JTTrainingContext
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3Output
from hyper_parallel.trainer.callbacks.environ_meter_callback import EnvironMeterCallback
from hyper_parallel.trainer.callbacks.logging_callback import LoggingCallback
from hyper_parallel.trainer.runtime.flops import TransformerFlopsEstimator
from hyper_parallel.trainer.state import TrainerState
from tests.ut.trainer.test_flops import decoder_config


def _sum_ce(*, logits: torch.Tensor, shift_labels: torch.Tensor, **_kwargs: object) -> torch.Tensor:
    return F.cross_entropy(logits.flatten(0, 1), shift_labels.flatten(), reduction="sum")


def _check_jt_cp_diagnostics() -> None:
    """Exercise JT's real CP CE, target halos and root hook with unequal valid counts."""
    rank, degree = dist.get_rank(), dist.get_world_size()
    context = object.__new__(_JTTrainingContext)
    context.cp_group, context.rank, context.degree = None, rank, degree
    context.model = SimpleNamespace(config=SimpleNamespace(vocab_size=4), loss_function=_sum_ce)
    torch.manual_seed(rank + 51)
    logits = torch.randn(1, 4, 4, requires_grad=True)
    labels = torch.tensor([[0, 1, 2, 3]])
    labels[:, rank:] = -100
    lm_loss = context.token_loss(logits, labels, labels >= 0)
    metrics = {"lm_loss": lm_loss.detach(), "load_balancing_loss": torch.tensor(1.0 / degree)}

    def _mtp_ce(*, logits: torch.Tensor, shift_labels: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        return context.token_loss(logits, shift_labels, shift_labels >= 0)

    mtp_loss = calculate_mtp_loss(
        [logits, logits], labels, _mtp_ce, vocab_size=4, loss_factor=0.3,
        shift_fn=lambda value: context.shift_inputs(value, pad_value=-100), loss_metrics=metrics,
    )
    partial_loss = lm_loss + mtp_loss
    output = JTDeepseekV3Output(loss={"foundation_loss/lm": lm_loss, "foundation_loss/mtp": mtp_loss},
                               loss_metrics=metrics)
    output = context.reduce_losses(None, (), output)
    torch.testing.assert_close(output.loss_metrics["lm_loss"], output.loss["foundation_loss/lm"].detach())
    torch.testing.assert_close(
        (output.loss_metrics["mtp_1_loss"] + output.loss_metrics["mtp_2_loss"]) * 0.15,
        output.loss["foundation_loss/mtp"].detach(),
    )
    torch.testing.assert_close(output.loss_metrics["load_balancing_loss"], torch.tensor(1.0))
    actual = torch.autograd.grad(sum(output.loss.values()), logits, retain_graph=True)[0]
    expected = torch.autograd.grad(partial_loss * degree, logits)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_parallel_metrics() -> None:
    """Use four ranks to verify units, slowest rank timing and replicated diagnostics."""
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    try:
        for dp, cp, tp in ((4, 1, 1), (2, 2, 1), (2, 1, 2), (1, 2, 2), (1, 1, 4)):
            groups = [dist.new_group(list(range(index, world, tp))) for index in range(tp)]
            group = groups[rank % tp]
            mesh = SimpleNamespace(dp_size=dp, cp_size=cp, tp_size=tp, pp_size=1,
                                   dp_cp_mesh=None if dp * cp == 1 else SimpleNamespace(get_group=lambda: group))
            trainer = SimpleNamespace(
                mesh=mesh, model_config=decoder_config(), global_rank=rank, lr_scheduler=None,
                optimizer=SimpleNamespace(param_groups=[{"lr": 0.001}]),
                config=SimpleNamespace(training=SimpleNamespace(logging_steps=1)),
            )
            meter = EnvironMeterCallback(trainer)
            state = TrainerState(global_step=1)
            replica = rank // (tp * cp)
            with patch("hyper_parallel.trainer.callbacks.environ_meter_callback.time.perf_counter",
                       side_effect=(100.0, 101.0 + rank)):
                meter.on_step_begin(state, micro_batches=[{
                    "input_ids": torch.zeros(1, 8 // cp), "token_count": torch.tensor(replica + 1),
                }])
                meter.record_loss_metrics({"loss_metrics": {"mtp_1_loss": torch.tensor(float(replica + 1))}})
                meter.on_step_end(state, loss=2.0, loss_dict=None, grad_norm=0.5)
            metrics = trainer.step_env_metrics
            expected = {
                "data/step_tokens": cp * dp * (dp + 1) / 2,
                "data/step_samples": dp,
                "data/step_input_tokens": dp * 8,
                "performance/step_time": 4.0,
                "performance/samples_per_second": dp / 4,
                "performance/input_tokens_per_second": dp * 2,
                "training/mtp_1_loss": (dp + 1) / 2,
                "performance/throughput_tflops_per_device": (
                    TransformerFlopsEstimator(trainer.model_config)({"input_ids": torch.zeros(dp, 8)})
                    / (4 * world * 1e12)
                ),
            }
            for name, target in expected.items():
                actual = metrics[name]
                assert abs(actual - target) <= abs(target) * 1e-6, (
                    f"layout={(dp, cp, tp)} rank={rank} metric={name} expected={target}, actual={actual}"
                )
            callback = LoggingCallback(trainer)
            with patch.object(callback, "_write") as write:
                callback.on_step_end(state, loss=2.0, loss_dict={}, grad_norm=0.5)
                assert write.call_count == int(rank == 0), (
                    f"rank={rank}, calls={write.call_count}, expected={int(rank == 0)}"
                )
            original_estimator = meter._flops_estimator
            meter._flops_estimator = lambda batch, cp_size: None if replica == 0 else original_estimator(
                batch, cp_size=cp_size,
            )
            meter.on_step_begin(state, micro_batches=[{"input_ids": torch.zeros(1, 8 // cp)}])
            meter.on_step_end(state, loss=2.0, loss_dict=None, grad_norm=0.0)
            assert "performance/throughput_tflops_per_device" not in trainer.step_env_metrics, (
                f"layout={(dp, cp, tp)} rank={rank} incomplete estimator must omit TFLOP/s"
            )
            for created in groups:
                if created != dist.GroupMember.NON_GROUP_MEMBER:
                    dist.destroy_process_group(created)
        _check_jt_cp_diagnostics()
    finally:
        dist.destroy_process_group()
