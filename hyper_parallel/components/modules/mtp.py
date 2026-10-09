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
"""DeepSeek-style sequential multi-token prediction for the Torch runtime.

Depth ``k`` fuses the previous depth's output state with the embeddings of the
tokens shifted ``k`` positions left, runs its own decoder and output norm, and
feeds the result both to the shared output head and to depth ``k + 1``, as in
Megatron-LM and MindFormers. The objective lives in
:func:`hyper_parallel.components.losses.calculate_mtp_loss`.
"""

# This adapter uses the Torch/HF runtime, like the existing model and Trainer modules.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from torch import nn


class MultiTokenPredictionLayer(nn.Module):
    """Fuse next-token embeddings and the previous state using injected components.

    Norms, decoder and projection are supplied by the model. The enclosing
    :class:`MultiTokenPrediction` owns token shifting and depth chaining. This
    layer imposes no model-specific dtype or distributed defaults.
    """

    def __init__(self, *, embedding_norm: nn.Module, hidden_norm: nn.Module,
                 projection: nn.Module, decoder: nn.Module, output_norm: nn.Module) -> None:
        """Register caller-provided components without changing their parameters.

        Args:
            embedding_norm: Normalization for future-token embeddings.
            hidden_norm: Normalization for the previous state.
            projection: Embedding/hidden fusion projection.
            decoder: Model-provided decoder.
            output_norm: Normalization of this depth's output state.
        """
        super().__init__()
        self.enorm = embedding_norm
        self.hnorm = hidden_norm
        self.eh_proj = projection
        self.transformer_layer = decoder
        self.final_layernorm = output_norm

    def forward(self, hidden: torch.Tensor, embedding: torch.Tensor,
                **decoder_kwargs: Any) -> torch.Tensor:
        """Return this depth's output state, consumed by the head and the next depth.

        Args:
            hidden: Previous state: the trunk output or the previous depth's output.
            embedding: Embeddings of the shifted future tokens.
            **decoder_kwargs: Causal attention and position arguments of the decoder.
        """
        combined = torch.cat((self.hnorm(hidden), self.enorm(embedding)), dim=-1)
        return self.final_layernorm(self.transformer_layer(self.eh_proj(combined), **decoder_kwargs))


def shift_mtp_sequence(value: torch.Tensor, *, pad_value: int = 0) -> torch.Tensor:
    """Shift a global batch/sequence tensor left, padding with zero without wrapping.

    Args:
        value: Complete batch/sequence tensor to shift.
        pad_value: Fill value beyond the global sequence tail.
    """
    return torch.cat((value[:, 1:], torch.full_like(value[:, :1], pad_value)), dim=1)


@dataclass
class MultiTokenPredictionOutput:
    """Per-depth prediction logits and the last depth's output state."""

    logits: tuple[torch.Tensor, ...]
    hidden_states: torch.Tensor


class MultiTokenPrediction(nn.Module):
    """Own the prediction depths and chain them over shifted future tokens."""

    def __init__(self, layers: list[MultiTokenPredictionLayer]) -> None:
        """Register independent depths without duplicating the shared embedding/head."""
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden: torch.Tensor, input_ids: torch.Tensor, *,
                embedding: nn.Module, head: nn.Module,
                decoder_kwargs: Mapping[str, Any] | None = None,
                shift_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
                sequence_end_mask: torch.Tensor | None = None) -> MultiTokenPredictionOutput:
        """Shift future tokens and run the depths in order, returning each depth's logits.

        Args:
            hidden: Main trunk state, before its final output normalization.
            input_ids: Local or complete token IDs, shaped [batch, sequence].
            embedding: The main model's shared token embedding; not registered here.
            head: Shared output head.
            decoder_kwargs: Causal attention/position arguments for every decoder.
            shift_fn: One-token left shift, including partition halos when needed.
            sequence_end_mask: Document-tail positions whose future embeddings must be zero.

        Note:
            The default shift treats each row as one unpartitioned sequence. A custom
            shift and document-tail mask support partitioned or packed inputs.
            Depth ``k`` logits are aligned with the main targets and score the
            token ``k`` positions later; ``calculate_mtp_loss`` applies that shift.
        """
        if input_ids.ndim != 2 or input_ids.numel() == 0:
            raise ValueError("MTP requires nonempty [batch, sequence] global token IDs")
        decoder_kwargs = {} if decoder_kwargs is None else decoder_kwargs
        if sequence_end_mask is not None and sequence_end_mask.shape != input_ids.shape:
            raise ValueError("MTP document-tail mask must match input_ids")
        shift_fn = shift_mtp_sequence if shift_fn is None else shift_fn
        logits = []
        for layer in self.layers:
            input_ids = shift_fn(input_ids)
            if sequence_end_mask is not None:
                input_ids = input_ids.masked_fill(sequence_end_mask, 0)
            hidden = layer(hidden, embedding(input_ids), **decoder_kwargs)
            logits.append(head(hidden))
        return MultiTokenPredictionOutput(tuple(logits), hidden)


class DeepseekV3MTP(MultiTokenPrediction):
    """Construct reusable V3-style MTP with model-provided Transformer decoders.

    The decoder factory permits both V3 and V3.2 decoder implementations without
    duplicating the MTP algorithm. It must return a causal decoder whose forward
    returns a Tensor. Attention/position arguments are forwarded unchanged.
    """

    def __init__(self, *, hidden_size: int, num_layers: int,
                 decoder_factory: Callable[[int], nn.Module], rms_norm_eps: float = 1e-6,
                 norm_factory: Callable[[int], nn.Module] | None = None,
                 output_norm_factory: Callable[[int], nn.Module] | None = None) -> None:
        """Build independent fusion norms, projections and Transformer layers.

        Args:
            hidden_size: Hidden and embedding feature size.
            num_layers: Number of independent future prediction depths; zero is valid.
            decoder_factory: Creates one independent causal decoder for each depth.
            rms_norm_eps: Epsilon for the default Torch RMSNorm.
            norm_factory: Optional optimized or precision-specific fusion normalization.
            output_norm_factory: Optional per-depth output norm, applied before the
                head and the next depth; by default the depth output is not
                normalized, so the head passed to forward must normalize.
        """
        if hidden_size <= 0 or num_layers < 0:
            raise ValueError("MTP hidden_size must be positive and num_layers nonnegative")
        if norm_factory is None:
            norm_factory = partial(nn.RMSNorm, eps=rms_norm_eps)
        if output_norm_factory is None:
            output_norm_factory = nn.Identity
        super().__init__([
            MultiTokenPredictionLayer(
                embedding_norm=norm_factory(hidden_size), hidden_norm=norm_factory(hidden_size),
                projection=nn.Linear(2 * hidden_size, hidden_size, bias=False),
                decoder=decoder_factory(index), output_norm=output_norm_factory(hidden_size),
            ) for index in range(num_layers)
        ])
