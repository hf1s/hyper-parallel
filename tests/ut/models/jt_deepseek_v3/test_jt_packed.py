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
"""Packed JT objectives keep document semantics with chunked output projection."""

import copy
from functools import partial
from itertools import product
from typing import Any
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import DefaultDeviceType

from hyper_parallel.components.losses.chunked_cross_entropy import projected_cross_entropy
from hyper_parallel.distributed.activation_checkpoint import _apply_activation_checkpointing
from hyper_parallel.models.jt_deepseek_v3.adapter.distributed.context_parallel import _JTTrainingContext
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (
    JTDeepseekV3ForCausalLM, JTDeepseekV3MoE, TreeBackwardCE, archived_cross_entropy, tiled_sum,
)
from tests.ut.models.jt_deepseek_v3.test_modeling_jt_deepseek_v3 import small_config


def reference_grouped_swiglu(inputs: torch.Tensor, gate_up: torch.Tensor,
                            down: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Use differentiable CPU expert math to test loss policy independently of NPU kernels."""
    outputs = []
    for index, tokens in enumerate(torch.split(inputs, counts.tolist())):
        gate, up = (tokens @ gate_up[index].t()).chunk(2, dim=-1)
        outputs.append((F.silu(gate) * up) @ down[index].t())
    return torch.cat(outputs)


VOCAB = 152064
TILE = 108 * 64
TINY = 2.0 ** -24  # Half an FP32 ulp of 1.0: 1 + TINY rounds back to 1 (ties to even).


def _rows() -> tuple[torch.Tensor, torch.Tensor]:
    """Return three full-vocabulary FP32 logit rows and targets with one ignored row."""
    logits = torch.randn(3, VOCAB, generator=torch.Generator().manual_seed(2026)) * 2
    logits[0] += 100  # exp would overflow FP32 without the max subtraction
    return logits, torch.tensor([7, -100, VOCAB - 1])


class TestArchivedCrossEntropy(unittest.TestCase):
    """Chunk-loss CE: native forward value, FP64-accurate derivative and the fixed reduction schedule."""

    def test_forward_is_native_sum_cross_entropy(self) -> None:
        """The loss value is exactly native sum-reduced CE, including the ignored target."""
        logits, targets = _rows()
        expected = F.cross_entropy(logits, targets, reduction="sum", ignore_index=-100)
        torch.testing.assert_close(TreeBackwardCE.apply(logits, targets, -100), expected, rtol=0, atol=0)

    def test_backward_matches_double_precision_gradient(self) -> None:
        """The recomputed softmax derivative matches FP64 autograd and is zero on ignored rows."""
        logits, targets = _rows()
        count = (targets != -100).sum()
        inputs = logits.clone().requires_grad_()
        (TreeBackwardCE.apply(inputs, targets, -100) / count).backward()
        reference = logits.double().requires_grad_()
        (F.cross_entropy(reference, targets, reduction="sum", ignore_index=-100) / count).backward()
        self.assertEqual(inputs.grad.dtype, torch.float32)
        torch.testing.assert_close(inputs.grad[1], torch.zeros(VOCAB), rtol=0, atol=0)
        torch.testing.assert_close(inputs.grad.double(), reference.grad, rtol=1e-5, atol=0)

    def test_tiled_sum_matches_double_precision_row_sums(self) -> None:
        """Rows inside and after the complete 40-row blocks keep FP32 summation accuracy."""
        generator = torch.Generator().manual_seed(11)
        for rows in (3, 41):
            scale = torch.arange(1, rows + 1, dtype=torch.float32)[:, None]
            values = torch.rand(rows, VOCAB, generator=generator) * scale
            with self.subTest(rows=rows):
                sums = tiled_sum(values)
                self.assertEqual((sums.shape, sums.dtype), ((rows, 1), torch.float32))
                # Each value meets at most 36 FP32 adds (9 in the 108-tree, 6 over lanes,
                # 21 over tiles); with nonnegative values the relative error is below 36 * 2**-24.
                torch.testing.assert_close(sums.double(), values.double().sum(-1, keepdim=True), rtol=2.2e-6, atol=0)

    def test_tiled_sum_follows_archived_pairing_and_40_row_tail(self) -> None:
        """Hand-rounded FP32 sums pin the tile order before and after the last complete 40-row block."""
        # Tile sums 1, TINY, ..., TINY. Left-to-right tile adds round every TINY away; the tail
        # adds the 11 odd tiles first, and 1 + 11 * TINY rounds to 1 + 12 * TINY.
        tiles = torch.zeros(VOCAB)
        tiles[0] = 1.0
        tiles[TILE::TILE] = TINY
        expected = torch.tensor([1.0] * 40 + [1.0 + 12 * TINY])[:, None]
        torch.testing.assert_close(tiled_sum(tiles.expand(41, VOCAB).contiguous()), expected, rtol=0, atol=0)
        torch.testing.assert_close(tiled_sum(tiles[None]), expected[-1:], rtol=0, atol=0)
        # Within a tile, both TINY values survive only if they meet each other before meeting 1:
        # entries 1 and 55 of lane 0 pair in the first halving level.
        values = torch.zeros(1, VOCAB)
        values[0, [0, 64, 55 * 64]] = torch.tensor([1.0, TINY, TINY])
        torch.testing.assert_close(tiled_sum(values), torch.full((1, 1), 1.0 + 2 * TINY), rtol=0, atol=0)

    def test_wrapper_uses_caller_denominator_and_ignores_labels(self) -> None:
        """The caller's valid-token count divides the native sum; ``labels`` is accepted and unused."""
        logits, targets = _rows()
        count = torch.tensor(2)
        expected = F.cross_entropy(logits, targets, reduction="sum", ignore_index=-100) / count
        for labels in (None, torch.zeros(3, dtype=torch.long)):
            with self.subTest(labels=labels is not None):
                actual = archived_cross_entropy(logits=logits, labels=labels, vocab_size=VOCAB,
                                                shift_labels=targets, num_items_in_batch=count)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_ignore_index_outside_the_vocabulary_is_masked_in_the_derivative(self) -> None:
        """An ignore_index beyond the vocabulary behaves like -100: native forward, zero gradient on its rows."""
        logits, targets = _rows()
        targets[1] = VOCAB
        inputs = logits.clone().requires_grad_()
        loss = TreeBackwardCE.apply(inputs, targets, VOCAB)
        torch.testing.assert_close(loss, F.cross_entropy(logits, targets, reduction="sum", ignore_index=VOCAB),
                                   rtol=0, atol=0)
        loss.backward()
        torch.testing.assert_close(inputs.grad[1], torch.zeros(VOCAB), rtol=0, atol=0)
        self.assertTrue(bool((inputs.grad[[0, 2]] != 0).any()))

    def test_model_chunk_loss_uses_archived_cross_entropy(self) -> None:
        """Feature: JT chunk-loss CE.

        Description: Run the model's checkpointed projection loss over one full and one partial chunk.
        Expectation: Loss and hidden gradient equal per-chunk ``archived_cross_entropy`` over the
            whole-sequence valid-target count, bit for bit.
        """
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")
        torch.manual_seed(3)
        config = small_config()
        config.vocab_size = VOCAB
        config.enable_chunk_loss = True
        config.chunk_loss_size = 4
        model = JTDeepseekV3ForCausalLM(config).requires_grad_(False)
        hidden = torch.randn(1, 6, config.hidden_size)
        targets = torch.tensor([[3, -100, VOCAB - 1, 7, -100, 11]])
        states = hidden.clone().requires_grad_()
        loss = model._projection_loss(states, targets)  # pylint: disable=protected-access
        loss.backward()
        count = (targets != -100).sum()
        expected = torch.zeros(())
        gradients = []
        for chunk in (slice(0, 4), slice(4, 6)):
            part = hidden[:, chunk].clone().requires_grad_()
            value = archived_cross_entropy(logits=model.lm_head(part), labels=None, vocab_size=VOCAB,
                                           shift_labels=targets[:, chunk], num_items_in_batch=count)
            expected = expected + value
            gradients.append(torch.autograd.grad(value, part)[0])
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        torch.testing.assert_close(states.grad, torch.cat(gradients, dim=1), rtol=0, atol=0)


class TestJTPacked(unittest.TestCase):
    """Compare actual model losses and gradients across projection policies."""

    def setUp(self) -> None:
        """Keep CPU checkpoint recomputation independent of installed accelerators."""
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")
        self.enterContext(patch(
            "hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3.npu_grouped_swiglu",
            side_effect=reference_grouped_swiglu))

    def test_packed_lm_mtp_equal_independent_documents(self) -> None:
        """Packed losses and gradients equal token-weighted independent document runs."""
        torch.manual_seed(8)
        config = small_config()
        config.moe_aux_loss_coeff = 0.0
        config.num_nextn_predict_layers = 2
        packed = JTDeepseekV3ForCausalLM(config)
        separate = copy.deepcopy(packed)
        tokens = torch.arange(1, 9).unsqueeze(0)
        labels = torch.tensor([[2, 3, 4, -100, 6, 7, 8, -100]])
        output = packed(tokens, labels, actual_seq_len=(4, 8)).loss
        left = separate(tokens[:, :4], labels[:, :4]).loss
        right = separate(tokens[:, 4:], labels[:, 4:]).loss
        keys = ("foundation_loss/lm", "foundation_loss/mtp")
        expected = {key: (left[key] + right[key]) / 2 for key in keys}
        for key in keys:
            torch.testing.assert_close(output[key], expected[key], rtol=2e-5, atol=2e-6)
        sum(output[key] for key in keys).backward()
        sum(expected.values()).backward()
        torch.testing.assert_close(
            {name: parameter.grad for name, parameter in packed.named_parameters()},
            {name: parameter.grad for name, parameter in separate.named_parameters()}, rtol=3e-4, atol=2e-6)

    def test_chunked_lm_mtp_preserves_packed_losses_and_gradients(self) -> None:
        """Feature: Chunk loss for packed LM and MTP objectives.

        Description: Compare two supervised MTP depths with unequal counts and a partial chunk.
        Expectation: All objective values and parameter gradients match the eager model.
        """
        torch.manual_seed(9)
        config = small_config()
        config.num_nextn_predict_layers = 2
        # Chunk loss uses archived_cross_entropy, whose derivative tiles a 152064-wide vocabulary.
        config.vocab_size = VOCAB
        eager = JTDeepseekV3ForCausalLM(config)
        chunked = copy.deepcopy(eager)
        inputs = torch.arange(12).unsqueeze(0)
        targets = torch.tensor([[1, 2, 3, -100, 5, -100, 7, -100, -100, 10, 11, -100]])
        expected = eager(inputs, targets, actual_seq_len=(4, 8, 12)).loss
        sum(expected.values()).backward()
        chunked.enable_chunk_loss = True
        chunked.chunk_loss_size = 5
        actual = chunked(inputs, targets, actual_seq_len=(4, 8, 12)).loss
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=2e-5, atol=2e-6)
        sum(actual.values()).backward()
        torch.testing.assert_close(
            {name: parameter.grad for name, parameter in chunked.named_parameters()},
            {name: parameter.grad for name, parameter in eager.named_parameters()}, rtol=3e-4, atol=2e-6)

    def test_chunked_cp_mean_matches_global_loss_and_gradients(self) -> None:
        """Feature: Checkpointed projection loss with CP token normalization.

        Description: Compare uneven, empty-local and empty-global target shards with global CE.
        Expectation: Loss and gradients agree; backward does not repeat the count collective.
        """
        torch.manual_seed(19)
        context = object.__new__(_JTTrainingContext)
        context.cp_group = object()

        def loss_fn(*, logits: torch.Tensor, shift_labels: torch.Tensor,
                    num_items_in_batch: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
            """Use independent summed CE for each checkpointed projection."""
            return F.cross_entropy(logits.flatten(0, 1), shift_labels.flatten(), reduction="sum") / num_items_in_batch

        label_cases = ((1, -100, -100, -100, 2, 3, 4, 5),
                       (-100, -100, -100, -100, 2, 3, 4, 5),
                       (-100,) * 8)
        for labels in label_cases:
            with self.subTest(labels=labels):
                hidden = torch.randn(1, 8, 4, requires_grad=True)
                head = torch.nn.Linear(4, 11, bias=False)
                targets = torch.tensor([labels])
                count = targets.ne(-100).sum()
                expected = F.cross_entropy(head(hidden).flatten(0, 1), targets.flatten(), reduction="sum")
                expected = expected / count.clamp_min(1)
                local_loss = partial(projected_cross_entropy, head=head, loss_fn=loss_fn,
                                     vocab_size=11, chunk_size=3)

                def reduce(value: torch.Tensor, **kwargs: Any) -> None:
                    """Supply the global count without creating a real process group."""
                    self.assertIs(kwargs["group"], context.cp_group)
                    value.copy_(count)

                with patch("hyper_parallel.models.jt_deepseek_v3.adapter.distributed.context_parallel.dist.all_reduce",
                           side_effect=reduce) as collective:
                    actual = sum(context.projection_loss(local_loss, states, shard)
                                 for states, shard in zip(hidden.chunk(2, 1), targets.chunk(2, 1)))
                    actual_gradients = torch.autograd.grad(actual, (hidden, head.weight))
                    self.assertEqual(collective.call_count, 2)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(actual_gradients, torch.autograd.grad(expected, (hidden, head.weight)))

    def test_layer_checkpoint_and_chunk_loss_gradients(self) -> None:
        """Feature: Layer checkpointing with chunk loss.

        Description: Exercise full/selective layer wrappers with chunk loss enabled and disabled.
        Expectation: Losses and gradients match eager execution without retaining auxiliary graphs.
        """
        torch.manual_seed(19)
        config = small_config()
        # Chunk loss uses archived_cross_entropy, whose derivative tiles a 152064-wide vocabulary.
        config.vocab_size = VOCAB
        reference = JTDeepseekV3ForCausalLM(config)
        self.assertFalse(reference.enable_chunk_loss)
        tokens = torch.arange(9).unsqueeze(0)
        targets = torch.tensor([[-100, -100, 3, 4, -100, 6, 7, 8, -100]])
        expected = reference(tokens, targets, actual_seq_len=(3, 6, 9)).loss
        sum(expected.values()).backward()
        for mode, enabled in product(("full", "selective"), (False, True)):
            with self.subTest(mode=mode, enable_chunk_loss=enabled):
                chunk_config = copy.deepcopy(config)
                chunk_config.chunk_loss_size = 3
                chunk_config.enable_chunk_loss = enabled
                model = JTDeepseekV3ForCausalLM(chunk_config)
                model.load_state_dict(reference.state_dict())
                self.assertEqual(model.enable_chunk_loss, enabled)
                model = _apply_activation_checkpointing(model, mode)
                head_calls = []
                handle = model.lm_head.register_forward_hook(
                    lambda _module, _args, output: head_calls.append(output.shape[1]))
                self.addCleanup(handle.remove)
                actual = model(tokens, targets, actual_seq_len=(3, 6, 9)).loss
                for key, loss in expected.items():
                    torch.testing.assert_close(actual[key], loss, rtol=2e-5, atol=2e-6)
                auxiliary_owners = [module for module in model.modules() if isinstance(module, JTDeepseekV3MoE)]
                self.assertTrue(all(module.auxiliary_loss is None for module in auxiliary_owners))
                forward_calls = len(head_calls)
                sum(actual.values()).backward()
                self.assertEqual(len(head_calls), forward_calls * (2 if enabled else 1))
                self.assertTrue(all(module.auxiliary_loss is None for module in auxiliary_owners))
                for parameter, original in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(parameter.grad, original.grad, rtol=3e-4, atol=2e-6)

    def test_chunk_loss_configuration(self) -> None:
        """Feature: Explicit chunk loss configuration.

        Description: Check defaults, enabled configuration and invalid option values.
        Expectation: Only a boolean switch and a positive integer size configure chunk loss.
        """
        config = small_config()
        default = JTDeepseekV3ForCausalLM(config)
        self.assertFalse(default.enable_chunk_loss)
        self.assertEqual(default.chunk_loss_size, 1024)
        config.enable_chunk_loss = True
        config.chunk_loss_size = 7
        enabled = JTDeepseekV3ForCausalLM(config)
        self.assertTrue(enabled.enable_chunk_loss)
        self.assertEqual(enabled.chunk_loss_size, 7)
        invalid = [("enable_chunk_loss", value) for value in ("false", 0, 1, None)]
        invalid += [("chunk_loss_size", value) for value in (0, -1, True, "1024", None)]
        for option, value in invalid:
            with self.subTest(option=option, value=value), self.assertRaisesRegex(ValueError, option):
                config = small_config()
                setattr(config, option, value)
                JTDeepseekV3ForCausalLM(config)
