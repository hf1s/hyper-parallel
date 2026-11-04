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
"""Xing4.0 training model: the released causal LM with the training MoE, MTP and losses.

The backbone is the released ``Xing4_0Model`` of
:mod:`~hyper_parallel.models.xing40.modeling_xing4_0`, run unmodified. Added here is what the
inference-only release lacks: the training MoE, the MindFormers MTP depths around the released
decoder layer and a forward that returns the training losses.
"""

# This model uses the Torch/HF runtime, like the existing model families.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.utils import ModelOutput

from hyper_parallel.components.functional.aux_loss import aux_loss_auto_scale
from hyper_parallel.components.losses.chunked_cross_entropy import chunked_cross_entropy
from hyper_parallel.core.activation_memory.recompute_state import is_recomputing
from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.models.materialization import register_rebuildable_buffer
from hyper_parallel.models.xing40.configuration_xing4_0 import Xing4_0Config
from hyper_parallel.models.xing40.modeling_xing4_0 import (
    Xing4_0DecoderLayer,
    Xing4_0ForCausalLM,
    Xing4_0HyperConnection,
    Xing4_0MLP,
    Xing4_0MoE,
    Xing4_0RMSNorm,
    Xing4_0RotaryEmbedding,
    Xing4_0TopkRouter,
)

__all__ = [
    "Xing40ForCausalLM",
    "Xing40ModuleListExperts",
    "Xing40MoE",
    "Xing40MTP",
    "Xing40MTPLayer",
    "Xing40Output",
]

# The MindFormers mHC initialization: a projection much narrower than initializer_range, so the
# residual streams start close to an even mix; the gating scale is mhc_init_gating_factor.
_MHC_PROJECTION_INIT_STD = 1e-4


class Xing40ModuleListExperts(nn.ModuleList):
    """Callable ``nn.ModuleList`` of the released per-expert ``Xing4_0MLP``.

    It is the ``*.mlp.experts`` boundary the grouped-GEMM replacement and the EP executor attach
    to. ``forward`` runs the released ``Xing4_0MoE.moe`` with this container standing in for the
    MoE, so the routing loop stays the released one.
    """

    @property
    def experts(self) -> "Xing40ModuleListExperts":
        """Return this container, so the released ``moe()`` can treat it as the MoE."""
        return self

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        """Route each token through its selected experts and combine their weighted outputs.

        Args:
            hidden_states: Flattened tokens, ``[num_tokens, hidden_size]``.
            top_k_index: Selected expert indices per token, ``[num_tokens, top_k]``.
            top_k_weights: Combination weight per selection, ``[num_tokens, top_k]``.
        """
        return Xing4_0MoE.moe(self, hidden_states, top_k_index, top_k_weights)


class Xing40MoE(nn.Module):
    """The released router and MLPs, composed with the MindFormers training machinery.

    Each call prefixes every sequence with one zero pad token per expert, injects the
    sequence-wise balance loss through the routing weights and counts the routed tokens that
    :meth:`update_expert_bias` steps the expert-load bias with.

    Note:
        Composed, not a ``Xing4_0MoE`` subclass: this forward pads, injects a loss and mutates
        state, so it must not stand in where the released behaviour is expected.
    """

    def __init__(self, config: Xing4_0Config) -> None:
        """Build the released submodules, then the training state."""
        super().__init__()
        self.config = config
        self.experts = Xing40ModuleListExperts(
            Xing4_0MLP(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.n_routed_experts))
        self.gate = Xing4_0TopkRouter(config)
        self.shared_experts = Xing4_0MLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts)
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.padding = config.n_routed_experts if config.use_pad_tokens else 0
        self.aux_loss_coeff = config.moe_aux_loss_coeff
        self.enable_expert_bias = config.moe_router_enable_expert_bias
        self.expert_bias_update_rate = config.moe_router_bias_update_rate
        # A buffer, so the Trainer's cross-rank all-reduce finds it on the module's device;
        # rebuildable, because to_empty() leaves it uninitialized on a meta build; int64, because
        # model.to(dtype) reaches buffers and a BF16 count stops at 256.
        register_rebuildable_buffer(
            self, "tokens_per_expert",
            value=torch.zeros(self.num_experts, dtype=torch.int64), persistent=False)
        # FP32: steps of expert_bias_update_rate fall below the BF16 spacing, so a BF16 bias stops
        # moving. The released _keep_in_fp32_modules_strict applies only under from_pretrained.
        self.gate.e_score_correction_bias = self.gate.e_score_correction_bias.float()
        self.last_aux_loss: torch.Tensor | None = None
        # Installed by the EP adapter; None computes the routed experts locally.
        self.ep_executor = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Add the routed and shared experts' outputs for ``[batch, tokens, hidden]`` inputs.

        Args:
            hidden_states: Normalized MLP input.
        """
        hidden = hidden_states
        if self.padding:
            hidden = torch.cat((hidden.new_zeros(hidden.shape[0], self.padding, hidden.shape[-1]), hidden), dim=1)
        if self.ep_executor is None:
            indices, weights = self.route(hidden)
            routed = self.experts(hidden.reshape(-1, hidden.shape[-1]), indices,
                                  weights).view_as(hidden)
        else:
            routed = self.ep_executor(self, hidden)  # pylint: disable=not-callable
        return routed[:, self.padding:] + self.shared_experts(hidden_states)

    def route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route the real tokens of a pad-prefixed input, and serve as the EP router.

        Pad tokens are routed to every expert with zero weight.

        Args:
            hidden: Pad-prefixed router inputs, ``[batch, padding + tokens, hidden]``.

        Returns:
            Expert indices and routing weights, ``[batch * (padding + tokens), top_k]``.
        """
        # Removing each sequence's pad prefix leaves a strided batch; the gate uses view().
        router_input = hidden[:, self.padding:].reshape(-1, hidden.shape[-1])
        logits, weights, indices = self.gate(router_input)
        # Training only: evaluation must neither steer the expert bias nor replace the logged aux loss.
        if self.training:
            scores = logits.sigmoid()
            weights = self._attach_balance_loss(weights, scores, batch_size=hidden.shape[0])
            # A recomputed forward replays the same routing; count each token once.
            if not is_recomputing():
                counts = torch.zeros_like(scores[0]).scatter_add_(
                    0, indices.reshape(-1), torch.ones_like(indices.reshape(-1), dtype=scores.dtype))
                self.tokens_per_expert.add_(counts.to(self.tokens_per_expert.dtype))
        else:
            # Cleared, so the monitor does not report a stale training value for this pass.
            self.last_aux_loss = None
        if self.padding:
            batch = hidden.shape[0]
            pad_indices = torch.arange(self.padding * self.top_k, device=indices.device)
            pad_indices = (pad_indices.view(1, self.padding, self.top_k) % self.num_experts).expand(batch, -1, -1)
            indices = torch.cat((pad_indices, indices.reshape(batch, -1, self.top_k)), dim=1).reshape(-1, self.top_k)
            weights = torch.cat((weights.new_zeros(batch, self.padding, self.top_k),
                                 weights.reshape(batch, -1, self.top_k)), dim=1).reshape(-1, self.top_k)
        # Cast here, not at the local call site: EP uses this method as its router, and both paths
        # must combine in the activation dtype, as MindFormers does.
        return indices, weights.to(hidden.dtype)

    def _attach_balance_loss(self, weights: torch.Tensor, scores: torch.Tensor,
                             batch_size: int) -> torch.Tensor:
        """Inject MindFormers' sequence-wise ``coeff * E * mean_b(sum_i(f_bi * P_bi))`` via the weights.

        ``f_bi`` counts the top-k of the raw affinities, deliberately not the bias-shifted
        selection the router makes: MindFormers derives the auxiliary indices from the raw router
        logits (``pynative/transformers/moe/router.py``), since penalizing the load the expert bias
        produced would make the two balancers fight. ``P_bi`` averages the normalized affinities.

        Args:
            weights: Routing weights of the real tokens.
            scores: Router affinities of every expert.
            batch_size: Independent sequences before the tokens were flattened.
        """
        selected = scores.topk(self.top_k, dim=-1, sorted=False).indices.reshape(batch_size, -1)
        counts = scores.new_zeros(batch_size, self.num_experts).scatter_add_(
            1, selected, torch.ones_like(selected, dtype=scores.dtype))
        load = counts / selected.shape[1]
        affinity = (scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)).reshape(
            batch_size, -1, self.num_experts).mean(dim=1)
        loss = (affinity * load).sum(dim=-1).mean() * self.num_experts
        self.last_aux_loss = loss.detach()
        return aux_loss_auto_scale(weights, loss * self.aux_loss_coeff)

    @torch.no_grad()
    def update_expert_bias(self, lr: float, num_recomputations: int = 1) -> None:
        """Step the expert-load bias toward balance with the step's summed token counts.

        As in MindFormers, the sign step is not re-centered.

        Args:
            lr: Bias update rate.
            num_recomputations: Unused; recomputed forwards do not count tokens.
        """
        del num_recomputations
        bias = self.gate.e_score_correction_bias
        # An int64 histogram has no mean; take the deviation in the bias's dtype.
        counts = self.tokens_per_expert.to(bias.dtype)
        bias.add_((counts.mean() - counts).sign(), alpha=lr)
        self.tokens_per_expert.zero_()


class _Xing40MTPDecoderLayer(nn.Module):
    """The released ``Xing4_0DecoderLayer``'s submodules and forward, without its base class.

    HyperParallel's FSDP-unit and activation-checkpointing discovery reads a
    ``gradient_checkpointing`` attribute, which the released ``GradientCheckpointingLayer``
    carries, as the mark of a block container: nested in a depth, the released layer would split
    its projections, experts and mHC input norms into units of their own.
    """

    def __init__(self, config: Xing4_0Config, layer_idx: int) -> None:
        """Adopt the submodules the released ``__init__`` builds for ``layer_idx``.

        Args:
            config: Resolved model configuration.
            layer_idx: Index of the layer among the decoder layers and the MTP depths.
        """
        super().__init__()
        released = Xing4_0DecoderLayer(config, layer_idx)
        for name, module in released.named_children():
            self.add_module(name, module)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Run the released forward on ``[batch, sequence, num_stream, hidden]`` residual streams.

        Args:
            hidden_states: Residual streams.
            **kwargs: The released forward's keyword arguments.
        """
        return Xing4_0DecoderLayer.forward(self, hidden_states, **kwargs)


class Xing40MTPLayer(nn.Module):
    """One MindFormers MTP depth around the released mHC decoder layer.

    Note:
        mHC-wrapped because MindFormers' ``mtp_enable_hyper_connections`` follows the decoder
        when unset, as in the reference configuration; the public release has no mHC under
        ``mtp.``.
    """

    def __init__(self, config: Xing4_0Config, layer_idx: int) -> None:
        """Build the fusion norms and projection, the decoder layer and the output norm.

        Args:
            config: Resolved model configuration.
            layer_idx: Index of the depth's decoder layer, after the backbone's layers.
        """
        super().__init__()
        self.hc_mult = config.hc_mult
        self.enorm = Xing4_0RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = Xing4_0RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.transformer_layer = _Xing40MTPDecoderLayer(config, layer_idx)
        self.final_layernorm = Xing4_0RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden: torch.Tensor, embeddings: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                actual_seq_len: Sequence[int] | None = None) -> torch.Tensor:
        """Return this depth's normalized ``[batch, sequence, hidden]`` state.

        Args:
            hidden: Previous normalized state: the backbone output or the previous depth's state.
            embeddings: Embeddings of the tokens shifted to this depth.
            position_embeddings: The backbone's FP32 RoPE ``(cos, sin)`` tables.
            actual_seq_len: The backbone's cumulative packed document ends, or None.
        """
        fused = self.eh_proj(torch.cat((self.enorm(embeddings), self.hnorm(hidden)), dim=-1))
        # The released Xing4_0Model.forward's expansion into the residual streams and mean collapse.
        streams = fused.unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()
        streams = self.transformer_layer(streams, position_embeddings=position_embeddings,
                                         actual_seq_len=actual_seq_len)
        return self.final_layernorm(streams.mean(dim=2))


class Xing40MTP(nn.Module):
    """Container of the multi-token-prediction depths."""

    def __init__(self, config: Xing4_0Config) -> None:
        """Build ``num_nextn_predict_layers`` depths.

        Args:
            config: Resolved model configuration.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            Xing40MTPLayer(config, config.num_hidden_layers + depth)
            for depth in range(config.num_nextn_predict_layers))
        # Marks ``layers`` as the repeated block container for FSDP units and activation checkpointing.
        self.gradient_checkpointing = False


def _attention_reads_document_ends(attention: nn.Module) -> bool:
    """Return whether the installed attention consumes ``actual_seq_len`` rather than ignoring it.

    The framework ``MLAAttention`` declares it; the released ``Xing4_0Attention`` takes it only
    through ``**kwargs``, which every Transformers attention interface silently swallows.

    Args:
        attention: Attention module installed on a decoder layer.
    """
    return "actual_seq_len" in inspect.signature(type(attention).forward).parameters


@dataclass
class Xing40Output(ModelOutput):
    """Named training losses computed by the forward."""

    # ``logits`` stays first and empty so that ModelOutput keeps ``loss`` as one mapping.
    logits: torch.Tensor | None = None
    loss: dict[str, torch.Tensor] | None = None


class Xing40ForCausalLM(Xing4_0ForCausalLM):
    """The released causal LM with MTP and the training MoE; ``forward`` returns the batch's losses."""

    _no_split_modules = ["Xing4_0DecoderLayer", "Xing40MTPLayer"]
    # SDPA only (see _check_and_adjust_attn_implementation); these make Transformers refuse the rest.
    _supports_flash_attn = False
    _supports_flex_attn = False
    _supports_attention_backend = False
    # False keeps activation checkpointing on HyperParallel's wrapper, whose recompute Xing40MoE
    # detects through is_recomputing(): with no MTP depth every block is a released
    # GradientCheckpointingLayer, and True would hand them to Transformers' own checkpointing.
    supports_gradient_checkpointing = False

    def __init__(self, config: Xing4_0Config) -> None:
        """Build the released backbone and LM head, then the MTP depths and the training MoE.

        Raises:
            ValueError: If the configuration selects an unsupported variant.
        """
        if config.q_lora_rank is None:
            raise ValueError("Xing4.0 requires the low-rank query projection (q_lora_rank)")
        if config.attention_dropout:
            raise ValueError("Xing4.0 requires attention_dropout=0")
        # Built uninitialized so that post_init below applies this class's scheme once: the
        # backbone's own post_init would mark its modules initialized with the released one.
        with init.no_init_weights():
            super().__init__(config)
            self.mtp = Xing40MTP(config) if config.num_nextn_predict_layers else None
            # Every released layer builds the inference Xing4_0MoE, the MTP depths' included.
            for module in list(self.modules()):
                if isinstance(getattr(module, "mlp", None), Xing4_0MoE):
                    module.mlp = Xing40MoE(config)
        # Rebuilt from a CPU-built value: to_empty() leaves the live buffers uninitialized on a meta
        # build, and a meta buffer cannot be snapshotted.
        with torch.device("cpu"):
            rotary_reference = Xing4_0RotaryEmbedding(config)
        for name in ("inv_freq", "original_inv_freq"):
            register_rebuildable_buffer(
                self.model.rotary_emb, name, value=getattr(rotary_reference, name).detach().clone())
        self.post_init()

    @classmethod
    def from_config(cls, config: Xing4_0Config, **kwargs: object) -> Xing40ForCausalLM:
        """Build from a resolved configuration, as the shared model builder requires.

        Args:
            **kwargs: Construction options of the shared builder; unused.
        """
        del kwargs
        return cls(config)

    def _check_and_adjust_attn_implementation(self, attn_implementation: str | None, is_init_check: bool = False,
                                              allow_all_kernels: bool = False) -> str:
        """Resolve the requested attention implementation, accepting only SDPA.

        The MTP depths pass no mask and rely on SDPA's ``is_causal``; eager attention would attend
        bidirectionally there.

        Args:
            attn_implementation: Requested implementation; None selects SDPA.
            is_init_check: Whether the check runs at construction.
            allow_all_kernels: Whether hub kernels outside ``kernels-community`` may load.

        Raises:
            ValueError: If the request resolves to anything but ``"sdpa"``.
        """
        implementation = super()._check_and_adjust_attn_implementation(
            attn_implementation, is_init_check, allow_all_kernels)
        if implementation != "sdpa":
            raise ValueError("Xing4.0 attends causally through SDPA without attention masks; "
                             f"attn_implementation={implementation!r} is unsupported")
        return implementation

    @torch.no_grad()
    @init.guard_torch_init_functions()
    def initialize_weights(self) -> None:
        """Initialize every module with this class's ``_init_weights``, the backbone's included.

        The inherited method hands a nested ``PreTrainedModel`` its own ``_init_weights``, which
        for the released ``Xing4_0Model`` is the released hyper-connection scheme.
        """
        self.apply(self._initialize_weights)

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize one module, giving hyper-connections the MindFormers scheme.

        The gating scale starts at ``mhc_init_gating_factor`` with a narrow projection, not at the
        released scale of one, so the streams begin close to an even mix. Returning before
        delegating keeps the released branch from drawing from the RNG only to be overwritten; the
        guarded ``init`` helpers leave a tensor loaded from a checkpoint alone.

        Args:
            module: Module to initialize.
        """
        if isinstance(module, Xing4_0HyperConnection):
            init.normal_(module.hc_fn, mean=0.0, std=_MHC_PROJECTION_INIT_STD)
            init.zeros_(module.hc_base)
            init.constant_(module.hc_scale, self.config.mhc_init_gating_factor)
            return
        super()._init_weights(module)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None,
                shift_labels: torch.Tensor | None = None, position_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                actual_seq_len: Sequence[int] | None = None,
                loss_mask: torch.Tensor | None = None,
                use_cache: bool = False) -> Xing40Output:
        """Return ``foundation_loss`` and ``foundation_loss/mtp`` for one batch.

        The router balance loss reaches the gradients through the routing weights, not this mapping.

        Args:
            input_ids: Token IDs ``[batch, sequence]``.
            labels: Unused; ``shift_labels`` carries the same targets.
            shift_labels: Next-token targets aligned with ``input_ids``; negatives are ignored.
            position_ids: RoPE positions; defaults to ``0..sequence-1``.
            attention_mask: Must be None: the packed document ends, not a mask, bound attention.
            actual_seq_len: Cumulative ends of the documents packed into the batch, over its
                flattened ``[batch, sequence]`` tokens, as ``Xing40PackedRuntime`` publishes
                them. None trains the batch as one document per row.
            loss_mask: Positions the Trainer scores, ``[batch, sequence]``; zeros are ignored.
                The Trainer always computes one and weights this loss by the tokens it leaves,
                so ignoring it here would divide by more targets than the weighting assumes and
                would still send gradient through the positions it drops.
            use_cache: Must be False.

        Raises:
            ValueError: If the inputs are not a nonempty batch of causal training sequences, or if
                the batch packs several documents per sequence while the installed attention
                cannot separate them.
        """
        del labels
        if attention_mask is not None or use_cache:
            raise ValueError("Xing4.0 trains causal sequences without attention masks or KV cache")
        if shift_labels is None or input_ids.ndim != 2 or min(input_ids.shape) < 1 \
                or shift_labels.shape != input_ids.shape:
            raise ValueError("Xing4.0 requires nonempty [batch, sequence] input_ids with matching shift_labels")
        # More ends than rows packs several documents into a sequence; SDPA has no variable-length
        # causal mode and would silently attend across their boundaries.
        if actual_seq_len is not None and len(actual_seq_len) > input_ids.shape[0] \
                and not _attention_reads_document_ends(self.model.layers[0].self_attn):
            raise ValueError(
                "this batch packs several documents per sequence, but the installed attention "
                "ignores actual_seq_len and would attend across the document boundaries. "
                "Select the fused NPU attention replacement (plan_overrides '*.self_attn'), "
                "or train one document per sequence.")
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        # The released forward hands actual_seq_len through its decoder layers to self_attn.
        hidden = self.model(input_ids, position_ids=position_ids, use_cache=False,
                            actual_seq_len=actual_seq_len).last_hidden_state
        targets = shift_labels.masked_fill(shift_labels < 0, IGNORE_INDEX)
        if loss_mask is not None:
            if loss_mask.shape != shift_labels.shape:
                raise ValueError("loss_mask must have the same shape as shift_labels")
            targets = targets.masked_fill(loss_mask == 0, IGNORE_INDEX)
        losses = {"foundation_loss": self._cross_entropy(hidden, targets)}
        if self.mtp is not None:
            # The released forward's own rotary call, so every depth gets the backbone's FP32 tables.
            position_embeddings = self.model.rotary_emb(hidden, position_ids)
            losses["foundation_loss/mtp"] = self._mtp_loss(hidden, input_ids, targets,
                                                           position_embeddings, actual_seq_len)
        return Xing40Output(loss=losses)

    def _mtp_loss(self, hidden: torch.Tensor, input_ids: torch.Tensor, targets: torch.Tensor,
                  position_embeddings: tuple[torch.Tensor, torch.Tensor],
                  actual_seq_len: Sequence[int] | None = None) -> torch.Tensor:
        """Return ``mtp_loss_factor / D * sum_k CE_k``, where depth ``k`` predicts ``k`` tokens further.

        ``hidden`` is the trunk's post-final-norm state: MindFormers pynative's
        ``GPTModel.forward`` hands ``self.mtp`` what ``TransformerBlock.forward`` returns after its
        ``final_layernorm``. Do not "fix" it to pre-norm; the change is invisible at
        initialization, where the outer norm is a positive rescale ``hnorm`` undoes, so no parity
        gate would catch it.

        Each depth shifts inputs and targets left by one, padding with token 0 and an ignored
        target, so it keeps the trunk's RoPE tables and document ends. At depth ``k`` the last
        ``k`` positions of a packed document fuse in the ``k`` tokens after its end; attention
        itself still stops at the boundary.
        """
        total = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for layer in self.mtp.layers:
            input_ids = F.pad(input_ids[:, 1:], (0, 1), value=0)
            targets = F.pad(targets[:, 1:], (0, 1), value=IGNORE_INDEX)
            # The backbone's table, uncast: the reference embeds in the compute dtype, and a cast
            # would feed an FP32 model BF16 activations.
            hidden = layer(hidden, self.model.embed_tokens(input_ids),
                           position_embeddings=position_embeddings, actual_seq_len=actual_seq_len)
            total = total + self._cross_entropy(hidden, targets)
        return total * (self.config.mtp_loss_factor / len(self.mtp.layers))

    def _cross_entropy(self, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return the LM-head cross entropy averaged over the non-ignored targets.

        A full-vocabulary head uses the chunked cross entropy, which never materializes the
        logits; a tensor-parallel head runs as a module, so its sharding plan and the Trainer's
        ``loss_function`` handle the sharded logits.
        """
        count = targets.ne(IGNORE_INDEX).sum().clamp_min(1)
        weight = self.lm_head.weight
        if getattr(weight, "placements", None) is None and weight.shape[0] == self.config.vocab_size:
            loss_sum = chunked_cross_entropy(hidden, targets, weight,
                                             chunk_size=self.config.chunk_loss_size, ignore_index=IGNORE_INDEX)
            return loss_sum / count
        return self.loss_function(self.lm_head(hidden), None, self.config.vocab_size, num_items_in_batch=count,
                                  ignore_index=IGNORE_INDEX, shift_labels=targets)
