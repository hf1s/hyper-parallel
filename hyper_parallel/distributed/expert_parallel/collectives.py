# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""expert_parallel.collectives: backend-dispatched EP all_to_all.

NCCL/HCCL use the ragged a2a (``_EPAllToAllUneven``, zero-padding); gloo and
other backends that do not support ragged a2a use pad-to-max +
``all_to_all_single`` (``_EPAllToAllPadded``). Both paths are numerically
equivalent (padding only adds filler rows that do not participate in
computation).

Split out of components/distributed/ep_utils.py in stage 4e.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist


_UNEVEN_A2A_BACKENDS = ("nccl", "hccl")


def _backend_supports_uneven_a2a(group) -> bool:
    return dist.get_backend(group) in _UNEVEN_A2A_BACKENDS


class _EPAllToAllUneven(torch.autograd.Function):  # pylint: disable=abstract-method
    """Ragged all_to_all (NCCL/HCCL production path): split by send/recv counts.

    forward:  split(x, send_counts) -> dist.all_to_all(out_list, in_list) -> cat
    backward: swap send/recv counts and run the ragged all_to_all again
              (a2a is self-inverse).
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        send_counts: list[int],
        recv_counts: list[int],
        group: Any,
    ) -> torch.Tensor:  # pylint: disable=arguments-differ
        """Run the ragged all_to_all and retain the counts for backward."""
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.group = group
        out = x.new_empty((sum(recv_counts),) + tuple(x.shape[1:]))
        dist.all_to_all(list(out.split(recv_counts)),
                        list(x.split(send_counts)), group=group)
        return out

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:  # pylint: disable=arguments-differ
        """Swap send/recv counts and re-run the self-inverse ragged all_to_all."""
        grad = _EPAllToAllUneven.apply(
            grad_output.contiguous(), ctx.recv_counts, ctx.send_counts, ctx.group)
        return grad, None, None, None


class _EPAllToAllPadded(torch.autograd.Function):  # pylint: disable=abstract-method
    """pad-to-max + all_to_all_single (gloo test path).

    forward:  pad each dest chunk to the global max(counts) (a2a_single
              requires equal-length chunks per rank -> pad_to must be
              globally consistent, obtained via all_reduce MAX)
              -> a2a_single -> unpad by recv_counts;
    backward: pad by recv_counts -> a2a_single (equal-length self-inverse)
              -> unpad by send_counts.
    """

    @staticmethod
    def _pad(x, counts, pad_to):
        """Pad destination chunks to a globally consistent row count."""
        chunks = []
        for chunk, n in zip(x.split(counts), counts):
            if n < pad_to:
                pad = x.new_zeros((pad_to - n,) + tuple(x.shape[1:]))
                chunk = torch.cat([chunk, pad])
            chunks.append(chunk)
        return torch.cat(chunks).contiguous()

    @staticmethod
    def _pad_and_exchange(x, counts, pad_to, group):
        """Pad each chunk to pad_to, run equal-length a2a_single, return [ep*pad_to, ...]."""
        send = _EPAllToAllPadded._pad(x, counts, pad_to)
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)
        return recv

    @staticmethod
    def _unpad(recv, counts, pad_to):
        """Take the valid rows from the equal-length buffer per counts and cat."""
        pieces = []
        for i, n in enumerate(counts):
            if n > 0:
                pieces.append(recv[i * pad_to: i * pad_to + n])
        if not pieces:
            return recv[:0]
        return torch.cat(pieces)

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        send_counts: list[int],
        recv_counts: list[int],
        group: Any,
    ) -> torch.Tensor:  # pylint: disable=arguments-differ
        """Exchange padded expert-token chunks and retain counts for backward."""
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.group = group
        local_max = max([*send_counts, *recv_counts, 1])
        pad_to = x.new_tensor([local_max], dtype=torch.int64)
        dist.all_reduce(pad_to, op=dist.ReduceOp.MAX, group=group)
        ctx.pad_to = pad_to = int(pad_to.item())
        recv = _EPAllToAllPadded._pad_and_exchange(x, send_counts, pad_to, group)
        return _EPAllToAllPadded._unpad(recv, recv_counts, pad_to)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:  # pylint: disable=arguments-differ
        """Reverse the exchange: pad by recv_counts, a2a_single, unpad by send_counts."""
        # backward = reversed a2a: pad by recv_counts -> a2a_single -> unpad by send_counts
        recv = _EPAllToAllPadded._pad_and_exchange(
            grad_output.contiguous(), ctx.recv_counts, ctx.pad_to, ctx.group)
        grad = _EPAllToAllPadded._unpad(recv, ctx.send_counts, ctx.pad_to)
        return grad, None, None, None


@dataclass
class _PendingEPExchange:
    """Retain send storage only until the outstanding exchange is waited."""

    work: Any
    send_buffer: torch.Tensor | None
    pad_to: int = 0

    def wait(self) -> None:
        """Join communication and release the temporary send buffer."""
        self.work.wait()
        self.send_buffer = None


class _EPAllToAllAsync(torch.autograd.Function):  # pylint: disable=abstract-method
    """Launch token exchange; the caller waits before consuming or unpadding it."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, send_counts: list[int], recv_counts: list[int],
                group: Any) -> tuple:  # pylint: disable=arguments-differ
        """Return the pending output and its storage-owning work handle.

        Args:
            ctx: Autograd context for reverse exchange metadata.
            x: Contiguous expert-major rows.
            send_counts: Rows sent to each rank.
            recv_counts: Rows received from each rank.
            group: Expert process group.

        Returns:
            Output storage and a pending exchange that must be waited.
        """
        ctx.send_counts, ctx.recv_counts, ctx.group = send_counts, recv_counts, group
        ctx.pad_to = 0
        if _backend_supports_uneven_a2a(group):
            out = x.new_empty((sum(recv_counts),) + tuple(x.shape[1:]))
            work = dist.all_to_all(list(out.split(recv_counts)), list(x.split(send_counts)),
                                   group=group, async_op=True)
            return out, _PendingEPExchange(work, x)
        pad_size = x.new_tensor([max([*send_counts, *recv_counts, 1])], dtype=torch.int64)
        dist.all_reduce(pad_size, op=dist.ReduceOp.MAX, group=group)
        pad_to = int(pad_size.item())
        send = _EPAllToAllPadded._pad(x, send_counts, pad_to)
        ctx.pad_to = pad_to
        out = torch.empty_like(send)
        work = dist.all_to_all_single(out, send, group=group, async_op=True)
        return out, _PendingEPExchange(work, send, pad_to)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor,
                 _grad_pending: Any) -> tuple:  # pylint: disable=arguments-differ
        """Reverse the exchange, excluding filler rows from the padded path.

        Args:
            ctx: Saved split counts and process group.
            grad_output: Gradient of the completed exchange.
            _grad_pending: No gradient for the Python work handle.

        Returns:
            Input gradient and placeholders for non-tensor inputs.
        """
        if ctx.pad_to:
            grad_output = _EPAllToAllPadded._unpad(grad_output, ctx.recv_counts, ctx.pad_to)
        grad = ep_all_to_all(grad_output.contiguous(), ctx.recv_counts, ctx.send_counts, ctx.group)
        return grad, None, None, None


def ep_all_to_all(
    x: torch.Tensor,
    send_counts: list[int],
    recv_counts: list[int],
    group: Any,
    *,
    overlap_fn: Callable[[], None] | None = None,
) -> torch.Tensor:
    """Unified entry for EP token exchange (autograd-differentiable).

    NCCL/HCCL -> ragged a2a (zero-padding); other backends (gloo test path) -> pad-to-max.

    When supplied, ``overlap_fn`` runs with the caller's autograd mode while
    the exchange is pending. The work is waited even if independent compute
    raises; no pending output is exposed to the callback or the caller.

    Args:
        x: Contiguous rows to exchange.
        send_counts: Row counts per destination rank.
        recv_counts: Row counts per source rank.
        group: Expert process group.
        overlap_fn: Optional independent compute executed before the combine wait.

    Returns:
        Completed exchanged rows with differentiable reverse communication.
    """
    if overlap_fn is not None:
        output, pending = _EPAllToAllAsync.apply(x, send_counts, recv_counts, group)
        try:
            overlap_fn()
        finally:
            pending.wait()
        if pending.pad_to:
            return _EPAllToAllPadded._unpad(output, recv_counts, pending.pad_to)
        return output
    if _backend_supports_uneven_a2a(group):
        return _EPAllToAllUneven.apply(x, send_counts, recv_counts, group)
    return _EPAllToAllPadded.apply(x, send_counts, recv_counts, group)
