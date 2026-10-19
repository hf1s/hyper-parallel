# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Focused tests for the opt-in archived JT objective policy."""

import unittest

import torch

from hyper_parallel.models.jt_deepseek_v3.archive_objective import (
    absolute_positions,
    archive_rope_embeddings,
    iter_packed_mtp_targets,
    split_archive_branches,
)


class TestArchiveObjective(unittest.TestCase):
    """Target shifting, positions, and branch-gradient joins."""

    def test_targets_cross_packed_document_boundary(self) -> None:
        labels = torch.tensor([[10, 11, 12, -100, 20, 21]])
        actual = list(iter_packed_mtp_targets(labels, 2))
        self.assertEqual(actual[0].tolist(), [[11, 12, -100, 20, 21, -100]])
        self.assertEqual(actual[1].tolist(), [[12, -100, 20, 21, -100, -100]])

    def test_absolute_positions_do_not_reset_at_document_boundary(self) -> None:
        positions = absolute_positions(4, sequence_start=3)
        torch.testing.assert_close(positions, torch.tensor([[3, 4, 5, 6]]))
        cos, sin = archive_rope_embeddings(positions, rope_theta=10000.0, rope_dim=4)
        self.assertEqual(cos.dtype, torch.float32)
        self.assertEqual(sin.dtype, torch.float32)
        frequency = torch.tensor([3.0, 0.03, 3.0, 0.03], dtype=torch.float32)
        torch.testing.assert_close(cos[0, 0], frequency.cos())
        torch.testing.assert_close(sin[0, 0], frequency.sin())

    def test_branch_gradients_join_from_live_outputs(self) -> None:
        value = torch.tensor([1.0, 2.0], dtype=torch.bfloat16, requires_grad=True)
        first, second = split_archive_branches(value)
        (first.float().sum() + (second.float() * 3).sum()).backward()
        self.assertEqual(value.grad.dtype, torch.bfloat16)
        torch.testing.assert_close(value.grad.float(), torch.full((2,), 4.0))


if __name__ == "__main__":
    unittest.main()
