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
"""Checkpoint policy enum shared by the activation-checkpoint implementations."""
import enum


class CheckpointPolicy(enum.Enum):
    """
    Enum for specifying the policy for checkpointing during backpropagation.

    This enum extends PyTorch's selective activation checkpointing policies
    by introducing a SWAP-based strategy, which allows activation tensors
    to be offloaded during the forward pass and loaded back before backward
    computation.

    For PyTorch native policies (SAVE / RECOMPUTE semantics and MUST vs PREFER),
    see: https://docs.pytorch.org/docs/2.6/checkpoint.html#torch.utils.checkpoint.CheckpointPolicy

    Additional policy:

    - ``MUST_SWAP``: The operation's output is offloaded to host memory during the
      forward pass and loaded back asynchronously before backward computation. The backward
      pass reuses the loaded activations without recomputation.

      This policy must be used together with :class:`SwapManager` to coordinate
      asynchronous offload/load and stream synchronization.

    .. note::
        ``MUST_SWAP`` is typically applied to operations that are either
        computationally expensive or have large memory footprints. Note that
        swapping very small outputs may introduce additional overhead and
        reduce the effectiveness of asynchronous copy.
    """
    MUST_SAVE = 0
    PREFER_SAVE = 1
    MUST_RECOMPUTE = 2
    PREFER_RECOMPUTE = 3

    # Offload during forward, reload before backward. Requires SwapManager.
    MUST_SWAP = 4
