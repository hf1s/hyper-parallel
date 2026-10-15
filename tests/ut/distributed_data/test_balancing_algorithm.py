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
"""Bounded feasibility repair for full packing bins."""

import random
import unittest
from collections import Counter
from unittest.mock import patch

from hyper_parallel.distributed_data import FeasibleLPTBalancingAlgorithm, LPTBalancingAlgorithm
from hyper_parallel.distributed_data.balancing_algorithm import resolve_balancing_algorithm
from hyper_parallel.distributed_data.planner import DynamicPackingPlanner
from hyper_parallel.distributed_data.schema import PackingConstraints, SampleKey, SampleMetadata, WorkloadCost
from tests.common.mark_utils import arg_mark


def _full_pair(scale: int = 1) -> tuple:
    samples = tuple(
        SampleMetadata(tokens * scale, key=SampleKey(0, 0, index), cost=WorkloadCost(llm=cost),
                       packing_costs={"encoder": 1})
        for index, (tokens, cost) in enumerate(zip((6, 2, 2, 5, 3, 2), (90, 70, 50, 100, 80, 60)))
    )
    return samples, (tuple(item.key for item in samples[:3]), tuple(item.key for item in samples[3:]))


def _random_full_bins(count: int, seed: int) -> tuple:
    generator = random.Random(seed)
    samples, bins = [], []
    for _ in range(count):
        cuts = [0, *sorted(generator.sample(range(1, 64), 5)), 64]
        keys = []
        for start, end in zip(cuts, cuts[1:]):
            length = end - start
            key = SampleKey(0, 0, len(samples))
            samples.append(SampleMetadata(length, key=key, cost=WorkloadCost(llm=length ** 2)))
            keys.append(key)
        bins.append(tuple(keys))
    return tuple(samples), tuple(bins)


class TestFeasibleLPT(unittest.TestCase):
    """Check useful repairs and bounded fallback without accelerator hardware."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_two_full_bins_improve_without_accumulation(self) -> None:
        """Feature: Single-round feasibility repair.
        Description: Repair the capacity-ten greedy dead end with two physical DP ranks.
        Expectation: Cost drops from 240 to 230 with exact capacities and repeated indices preserved.
        """
        samples, reference = _full_pair()
        options = {"reference_bins": reference, "constraints": PackingConstraints(10),
                   "data_parallel_size": 2, "local_batch_size": 1}
        self.assertIs(LPTBalancingAlgorithm().assign(samples, **options), reference)
        self.assertIs(type(resolve_balancing_algorithm()), FeasibleLPTBalancingAlgorithm)
        for validate in (False, True):
            with self.subTest(validate=validate):
                planner = DynamicPackingPlanner(
                    data_parallel_size=2, local_batch_size=1, seq_len=10, validate=validate,
                    cost_model=lambda item: item.cost,
                )
                plan = planner.plan(samples, reference_bins=reference, step=0)
                self.assertEqual([batch[0].pack_tokens for batch in plan.local_batches], [10, 10])
                self.assertEqual(Counter(plan.selected_keys), Counter(item.key for item in samples))
                self.assertEqual(planner.last_balance_decision["original_objective"], 240)
                self.assertEqual(planner.last_balance_decision["accepted_objective"], 230)
                self.assertTrue(planner.last_balance_decision["accepted"])
                planner.min_balance_gain = 0.1
                rejected = planner.plan(samples, reference_bins=reference, step=0)
                self.assertEqual(tuple(batch[0].sample_keys for batch in rejected.local_batches), reference)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_multi_bin_repairs_are_deterministic_and_conservative(self) -> None:
        """Feature: Bounded multi-bin repair.
        Description: Repack fixed-seed full domains of four or twelve bins.
        Expectation: Accepted plans conserve every occurrence and capacity with a non-worsening objective.
        """
        for count in (4, 12):
            failures = accepted = 0
            for seed in range(6):
                samples, reference = _random_full_bins(count, seed)
                options = {"reference_bins": reference, "constraints": PackingConstraints(64),
                           "data_parallel_size": count, "local_batch_size": 1}
                failed = LPTBalancingAlgorithm().assign(samples, **options) is reference
                failures += failed
                planner = DynamicPackingPlanner(
                    data_parallel_size=count, local_batch_size=1, seq_len=64,
                    cost_model=lambda item: item.cost,
                )
                plan = planner.plan(samples, reference_bins=reference, step=0)
                accepted += failed and planner.last_balance_decision["accepted"]
                with self.subTest(count=count, seed=seed):
                    self.assertEqual([batch[0].pack_tokens for batch in plan.local_batches], [64] * count)
                    self.assertEqual(Counter(plan.selected_keys), Counter(item.key for item in samples))
                    self.assertLessEqual(planner.last_balance_decision["accepted_objective"],
                                         planner.last_balance_decision["original_objective"])
                    self.assertEqual(plan, planner.plan(tuple(reversed(samples)), reference_bins=reference, step=0))
            self.assertGreater(failures, 0)
            self.assertGreater(accepted, 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_unsupported_domains_keep_the_original_lpt_policy(self) -> None:
        """Feature: Repair scope limits.
        Description: Use non-full bins, stage budgets, multiple bins per rank, or more than sixteen bins.
        Expectation: Unsupported cases retain ordinary LPT without running subset-sum repair.
        """
        pair, reference = _full_pair()
        four, four_reference = _random_full_bins(4, 0)
        many, many_reference = _random_full_bins(17, 0)
        cases = (
            (pair, reference, PackingConstraints(11), 2, 1),
            (pair, reference, PackingConstraints(10, packing_budgets={"encoder": 10}), 2, 1),
            (four, four_reference, PackingConstraints(64), 2, 2),
            (many, many_reference, PackingConstraints(64), 17, 1),
        )
        algorithm = FeasibleLPTBalancingAlgorithm()
        for samples, bins, constraints, dp_size, local_size in cases:
            options = {"reference_bins": bins, "constraints": constraints,
                       "data_parallel_size": dp_size, "local_batch_size": local_size}
            expected = LPTBalancingAlgorithm().assign(samples, **options)
            with self.subTest(dp=dp_size, local=local_size, constraints=constraints), patch.object(
                    algorithm, "_repair_full_bins", side_effect=AssertionError("Unsupported repair"),
            ):
                self.assertEqual(algorithm.assign(samples, **options), expected)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_table_budget_rejects_large_token_domains(self) -> None:
        """Feature: Bounded subset-sum allocation.
        Description: Scale a known dead end beyond the thirty-two MiB table budget.
        Expectation: The feasible reference is returned without constructing an oversized table.
        """
        samples, reference = _full_pair(scale=5_000_000)
        result = FeasibleLPTBalancingAlgorithm().assign(
            samples, reference_bins=reference, constraints=PackingConstraints(50_000_000),
            data_parallel_size=2, local_batch_size=1,
        )
        self.assertIs(result, reference)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_successful_greedy_assignments_are_retained(self) -> None:
        """Feature: Successful LPT compatibility.
        Description: Balance equal-length documents whose greedy packing succeeds.
        Expectation: The repair policy retains exactly the ordinary LPT candidate.
        """
        samples = tuple(SampleMetadata(5, key=SampleKey(0, index, index), cost=WorkloadCost(llm=cost))
                        for index, cost in enumerate((9, 9, 1, 1)))
        reference = ((samples[0].key, samples[1].key), (samples[2].key, samples[3].key))
        options = {"reference_bins": reference, "constraints": PackingConstraints(10),
                   "data_parallel_size": 2, "local_batch_size": 1}
        algorithm = FeasibleLPTBalancingAlgorithm()
        with patch.object(algorithm, "_repair_full_bins", side_effect=AssertionError("Unexpected repair")):
            self.assertEqual(algorithm.assign(samples, **options), LPTBalancingAlgorithm().assign(samples, **options))
