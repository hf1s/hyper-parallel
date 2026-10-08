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
"""DeepSeek-style sequential multi-token prediction for the Torch runtime."""

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
    """Fuse next-token embeddings and trunk states using injected components.

    Norms, decoder and projection can be supplied by the model. The enclosing
    DeepseekV3MTP owns token shifting and ``components.losses.calculate_mtp_loss``
    owns the objective. This layer alone imposes no model-specific dtype or
    distributed defaults.
    """

    def __init__(self, *, embedding_norm: nn.Module, hidden_norm: nn.Module,
                 projection: nn.Module, decoder: nn.Module, output_norm: nn.Module) -> None:
        """Register caller-provided components without changing their parameters.

        Args:
            embedding_norm: Normalization for future-token embeddings.
            hidden_norm: Normalization for trunk states.
            projection: Embedding/hidden fusion projection.
            decoder: Model-provided decoder.
            output_norm: Prediction output normalization.
        """
        super().__init__()
        self.enorm = embedding_norm
        self.hnorm = hidden_norm
        self.eh_proj = projection
        self.transformer_layer = decoder
        self.final_layernorm = output_norm

    def forward(self, hidden: torch.Tensor, embedding: torch.Tensor,
                **decoder_kwargs: Any) -> torch.Tensor:
        """Return the next prediction state without creating logits or losses.

        Args:
            hidden: Trunk hidden states.
            embedding: Next-token embedding states.
        """
        combined = self.fuse_inputs(hidden, embedding)
        return self.final_layernorm(self.transformer_layer(self.eh_proj(combined), **decoder_kwargs))

    def fuse_inputs(self, hidden: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """Normalize and concatenate states without model-specific precision casts.

        Args:
            hidden: Trunk hidden states.
            embedding: Next-token embedding states.
        """
        return torch.cat((self.hnorm(hidden), self.enorm(embedding)), dim=-1)


def shift_mtp_sequence(value: torch.Tensor, *, sequence_ends: tuple[int, ...] | None = None,
                       pad_value: int = 0) -> torch.Tensor:
    """Shift a global batch/sequence tensor left, padding document ends without wrapping.

    Args:
        value: Complete batch/sequence tensor to shift.
        sequence_ends: Optional exclusive ends of independent packed documents.
        pad_value: Value replacing the last element of every document.
    """
    shifted = torch.cat((value[:, 1:], torch.full_like(value[:, :1], pad_value)), dim=1)
    if sequence_ends is not None:
        indices = torch.tensor(sequence_ends, device=value.device, dtype=torch.int64) - 1
        shifted = shifted.index_fill(1, indices, pad_value)
    return shifted


@dataclass
class MultiTokenPredictionOutput:
    """Per-depth prediction logits and the final raw recurrent state."""

    logits: tuple[torch.Tensor, ...]
    hidden_states: torch.Tensor


class DeepseekV3MTPExecution(nn.Module):
    """Run the DeepSeek V3 prediction depths and return their logits.

    Layers stay owned by the parent MTP. Keeping execution as a parameter-free
    sibling allows replacing its precision policy independently of accelerated
    Attention modules inside those layers.
    """

    @staticmethod
    def fuse_inputs(layer: MultiTokenPredictionLayer, hidden: torch.Tensor,
                    embedding: torch.Tensor) -> torch.Tensor:
        """Use the public layer's normalization and hidden-then-embedding fusion.

        Args:
            layer: Components for the current depth.
            hidden: Previous decoder state.
            embedding: Future-token embedding.
        """
        return layer.fuse_inputs(hidden, embedding)

    @staticmethod
    def recurrent_state(raw_hidden: torch.Tensor, prediction_hidden: torch.Tensor) -> torch.Tensor:
        """Pass the decoder output to the next depth, before output-head normalization.

        Args:
            raw_hidden: Decoder output before prediction normalization.
            prediction_hidden: Normalized state supplied to the output head.
        """
        del prediction_hidden
        return raw_hidden

    def forward(self, layers: nn.ModuleList, hidden: torch.Tensor, input_ids: torch.Tensor,
                embedding: nn.Module, head: nn.Module,
                decoder_kwargs: Mapping[str, Any] | None = None,
                sequence_ends: tuple[int, ...] | None = None) -> MultiTokenPredictionOutput:
        """Shift future tokens and run independent depths, returning each depth's logits.

        Args:
            layers: Independently parameterized MTP depths owned by the parent.
            hidden: Main trunk state, before its final output normalization.
            input_ids: Complete global token IDs, shaped [batch, sequence].
            embedding: The main model's shared token embedding; not registered here.
            head: Shared output head, including any shared output normalization.
            decoder_kwargs: Causal attention/position arguments for every decoder.
            sequence_ends: Optional exclusive document ends, validated by the model.

        Note:
            Each row must contain complete sequences. When sequence_ends is supplied,
            future-token shifts stop at each document boundary. Context-parallel
            token shifts are not implemented.
            Depth ``k`` logits are aligned with the main targets and score the
            token ``k`` positions later; ``calculate_mtp_loss`` applies that shift.
        """
        if input_ids.ndim != 2 or input_ids.numel() == 0:
            raise ValueError("MTP requires nonempty [batch, sequence] global token IDs")
        decoder_kwargs = {} if decoder_kwargs is None else decoder_kwargs
        logits = []
        for layer in layers:
            input_ids = shift_mtp_sequence(input_ids, sequence_ends=sequence_ends)
            combined = self.fuse_inputs(layer, hidden, embedding(input_ids))
            raw_hidden = layer.transformer_layer(layer.eh_proj(combined), **decoder_kwargs)
            prediction_hidden = layer.final_layernorm(raw_hidden)
            hidden = self.recurrent_state(raw_hidden, prediction_hidden)
            logits.append(head(prediction_hidden))
        return MultiTokenPredictionOutput(tuple(logits), hidden)


class MultiTokenPrediction(nn.Module):
    """Own prediction depths and run the complete DeepSeek MTP forward."""

    def __init__(self, layers: list[MultiTokenPredictionLayer]) -> None:
        """Register independent depths without duplicating the shared embedding/head."""
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.execution = DeepseekV3MTPExecution()

    def forward(self, hidden: torch.Tensor, input_ids: torch.Tensor,
                **kwargs: Any) -> MultiTokenPredictionOutput:
        """Run the public execution contract with this module's registered layers.

        Args:
            hidden: Main trunk state before its final output normalization.
            input_ids: Complete global input token IDs.
        """
        return self.execution(self.layers, hidden, input_ids, **kwargs)


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
            output_norm_factory: Optional per-depth output norm; by default the shared
                head passed to forward owns output normalization, as in DeepSeek V3.
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
