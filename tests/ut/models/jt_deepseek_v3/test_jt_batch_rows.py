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
"""Batch rows are independent packed sequences inside one optimizer step."""

import copy
import unittest
from unittest.mock import patch

import torch
from torch.utils.checkpoint import DefaultDeviceType

from hyper_parallel.data.batching.runtime_input import RuntimeInputContext
from hyper_parallel.models.jt_deepseek_v3.adapter.data.runtime import JTSequenceRuntime
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3ForCausalLM
from tests.ut.models.jt_deepseek_v3.test_jt_packed import reference_grouped_swiglu
from tests.ut.models.jt_deepseek_v3.test_modeling_jt_deepseek_v3 import small_config


class TestJTBatchRows(unittest.TestCase):
    """Two rows must reproduce two independent batch-one runs."""

    def setUp(self) -> None:
        """Keep checkpoint recomputation and expert math independent of accelerators."""
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")
        self.enterContext(patch(
            "hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3.npu_grouped_swiglu",
            side_effect=reference_grouped_swiglu))

    def test_runtime_translates_flattened_row_boundaries(self) -> None:
        """Global ends span both rows, not one row's local interval."""
        runtime = JTSequenceRuntime()
        context = RuntimeInputContext(local_input_shape=(2, 4), parallel_ranks={"cp": 0},
                                      parallel_sizes={"cp": 1}, options={})

        metadata = runtime.build_runtime_inputs(
            batch={"cu_seq_lens": torch.tensor([0, 2, 4, 6, 8])}, context=context)

        self.assertEqual(metadata, {"actual_seq_len": (2, 4, 6, 8)})

    def test_runtime_rejects_single_row_boundaries_for_two_rows(self) -> None:
        """Forgetting to flatten the rows would drop the second row's tokens."""
        runtime = JTSequenceRuntime()
        context = RuntimeInputContext(local_input_shape=(2, 4), parallel_ranks={"cp": 0},
                                      parallel_sizes={"cp": 1}, options={})

        with self.assertRaisesRegex(ValueError, "flattened global batch"):
            runtime.build_runtime_inputs(batch={"cu_seq_lens": torch.tensor([0, 2, 4])}, context=context)

    @staticmethod
    def _losses_and_grads(model: JTDeepseekV3ForCausalLM, tokens: torch.Tensor, labels: torch.Tensor,
                          ends: tuple[int, ...]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Backpropagate the LM and MTP objectives and retain their gradients."""
        model.zero_grad(set_to_none=True)
        losses = model(tokens, labels, actual_seq_len=ends).loss
        total = losses["foundation_loss/lm"] + losses["foundation_loss/mtp"]
        total.backward()
        gradients = {name: parameter.grad.detach().clone()
                     for name, parameter in model.named_parameters() if parameter.grad is not None}
        return losses, gradients

    def test_two_rows_match_two_independent_runs(self) -> None:
        """Row-batched loss and gradients equal the token-weighted two-run mean."""
        torch.manual_seed(23)
        config = small_config()
        config.moe_aux_loss_coeff = 0.0
        model = JTDeepseekV3ForCausalLM(config)

        tokens = torch.arange(1, 17).view(2, 8)
        labels = tokens + 1
        labels[:, 3] = -100
        labels[:, 7] = -100
        batch_losses, batch_grads = self._losses_and_grads(model, tokens, labels, (4, 8, 12, 16))

        first_losses, first_grads = self._losses_and_grads(
            copy.deepcopy(model), tokens[:1], labels[:1], (4, 8))
        second_losses, second_grads = self._losses_and_grads(
            copy.deepcopy(model), tokens[1:], labels[1:], (4, 8))

        counts = torch.tensor([int((row != -100).sum()) for row in labels], dtype=torch.float32)
        weights = (counts / counts.sum()).tolist()
        for key in ("foundation_loss/lm", "foundation_loss/mtp"):
            expected = first_losses[key] * weights[0] + second_losses[key] * weights[1]
            torch.testing.assert_close(batch_losses[key], expected, rtol=1e-5, atol=1e-6)
        for name, gradient in batch_grads.items():
            expected = first_grads[name] * weights[0] + second_grads[name] * weights[1]
            torch.testing.assert_close(gradient, expected, rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
