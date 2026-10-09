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
"""Sequence-wise aux loss and its group-replicated reduction against independent oracles."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.components.losses.aux_loss import _replicated_group_mean, calculate_seq_aux_loss
from tests.common.mark_utils import arg_mark


def sequence_aux_oracle(scores: torch.Tensor, selected: torch.Tensor, coeff: float) -> torch.Tensor:
    """DeepSeek-V3 ``coeff * E * sum_i(f_i * P_i)`` of one complete sequence."""
    num_experts = scores.shape[-1]
    load = torch.bincount(selected.flatten(), minlength=num_experts) / selected.numel()
    affinity = (scores / scores.sum(-1, keepdim=True)).mean(0)
    return coeff * num_experts * (load * affinity).sum()


class TestParallelReduction(unittest.TestCase):
    """Distinguish one global objective from independently consumed rank losses."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_partition_gradients_match_global_objective(self):
        """Feature: Replicated loss ownership.

        Description: Average unequal local losses across two or eight emulated ranks.
        Expectation: Every rank matches its slice of a dense mean's gradient, without backward communication.
        """
        for size in (2, 8):
            full = torch.arange(1., size + 1, requires_grad=True)
            expected = full.square().mean()
            (expected * 3).backward()
            for rank in range(size):
                local = full[rank].detach().clone().requires_grad_()
                group = object()
                with patch("torch.distributed.get_world_size", return_value=size), \
                        patch("torch.distributed.all_reduce") as reduce:
                    reduce.side_effect = lambda tensor, **_: tensor.copy_(full.detach().square().sum())
                    result = _replicated_group_mean(local.square(), group)
                    (result * 3).backward()
                    self.assertEqual(reduce.call_count, 1)
                    self.assertIs(reduce.call_args.kwargs["group"], group)
                torch.testing.assert_close(result, expected)
                torch.testing.assert_close(local.grad, full.grad[rank])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_no_group_is_local_even_with_default_group(self):
        """Feature: Explicit collective ownership.

        Description: Omit the model-parallel group.
        Expectation: No default group is queried and the tensor is returned unchanged.
        """
        value = torch.tensor(2., requires_grad=True)
        with patch("torch.distributed.all_reduce", side_effect=AssertionError), \
                patch("torch.distributed.get_world_size", side_effect=AssertionError):
            result = _replicated_group_mean(value)
            result.backward()
        self.assertIs(result, value)
        self.assertEqual(value.grad.item(), 1.)


class TestSequenceAuxLoss(unittest.TestCase):
    """The DeepSeek-V3 sequence-wise aux loss against complete-sequence oracles."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_sequences_are_balanced_independently(self):
        """Feature: Sequence-wise aux loss.

        Description: Route two sequences of a batch to disjoint expert halves.
        Expectation: The loss averages each sequence's own statistic, not the statistic of the merged tokens.
        """
        torch.manual_seed(0)
        scores = torch.rand(2, 6, 4)
        selected = torch.stack((torch.randint(0, 2, (6, 2)), torch.randint(2, 4, (6, 2))))
        result = calculate_seq_aux_loss(scores, selected, coeff=0.1)
        expected = (sequence_aux_oracle(scores[0], selected[0], 0.1)
                    + sequence_aux_oracle(scores[1], selected[1], 0.1)) / 2
        torch.testing.assert_close(result, expected)
        merged = sequence_aux_oracle(scores.reshape(12, 4), selected.reshape(12, 2), 0.1)
        self.assertFalse(torch.allclose(result, merged))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_partitioned_sequence_matches_complete_sequence(self):
        """Feature: Sequence-parallel aux loss.

        Description: Split one sequence across two emulated ranks of the sequence-partition group.
        Expectation: Each rank returns the complete-sequence loss and the gradient of its own token slice.
        """
        torch.manual_seed(1)
        full = torch.rand(8, 4, requires_grad=True)
        selected = torch.randint(0, 4, (8, 2))
        expected = sequence_aux_oracle(full, selected, 0.1)
        expected.backward()
        global_load = torch.bincount(selected.flatten(), minlength=4) / selected.numel()
        shards = (slice(0, 4), slice(4, 8))
        for rank, shard in enumerate(shards):
            peer = shards[1 - rank]
            peer_load = torch.bincount(selected[peer].flatten(), minlength=4) / selected[peer].numel()
            peer_scores = full.detach()[peer]
            peer_loss = 0.1 * 4 * (global_load * (peer_scores / peer_scores.sum(-1, keepdim=True)).mean(0)).sum()

            def all_reduce(tensor: torch.Tensor, op: object = None, group: object = None,
                           peer_load: torch.Tensor = peer_load, peer_loss: torch.Tensor = peer_loss) -> None:
                """Add the peer's expert load, then its loss, as a two-rank sum."""
                del op, group
                tensor.add_(peer_load if tensor.dim() else peer_loss)

            local = full.detach()[shard].clone().requires_grad_()
            with patch("torch.distributed.get_world_size", return_value=2), \
                    patch("torch.distributed.all_reduce", side_effect=all_reduce):
                result = calculate_seq_aux_loss(local, selected[shard], coeff=0.1, sequence_partition_group=object())
                result.backward()
            torch.testing.assert_close(result, expected)
            torch.testing.assert_close(local.grad, full.grad[shard])
