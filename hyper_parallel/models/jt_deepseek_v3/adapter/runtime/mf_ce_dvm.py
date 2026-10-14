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
"""MindFormers-parity DVM cross-entropy for the JT DeepSeek-V3 unchunked LM loss.

HP's unchunked causal-LM loss (``modeling_jt_deepseek_v3.JTDeepseekV3ForCausalLM._token_loss``,
used when ``config.loss_chunk_size`` is ``0``) reduces the softmax normalizer and the masked
loss with plain ``torch`` ops (``.exp().sum(-1)``, ``(losses * mask).sum()``). MindFormers
compiles the same full ``[4096, 152064]`` logits matrix
(``mindformers/parallel_core/training_graph/loss_func.py:175-340``'s ``CrossEntropyLoss``)
through its Ascend 910B3 graph, which lowers the vocabulary-axis reduction to a measured
SIMD64 tree instead of one linear accumulation. FP32 addition is not associative, so the
tree's op order rounds differently than ``.sum(-1)``; the schedule itself has no separate
readable MindFormers Python source -- it is the compiled graph's op order, reverse-engineered
and validated against MF's logged reference values in ``precision_probe.py`` (SHA256
``57a655514521b4c81cf04eaf945035f3c48ee5ea6265cb618e803d01fe3d805c``, matching issue #179's
citation). Issue #179's ``ce-dvm`` factor measures this gap (200-step MAE 0.047 vs MF,
first-step delta 0, 200th-step delta 0.058, judged reference-compatible/P2, not a
default-changing fix).

This module ports that measured schedule (``_dvm_sumx``, ``_sequential_sum``, ``_dvm_ce_sum``,
``MFGraphCrossEntropy``, ``MFDVMCrossEntropy``, ``mf_dvm_cross_entropy``) essentially verbatim
from ``precision_probe.py``, plus a ``build_jt_model`` wrapper that installs it as
``model.loss_function``, as an alternative loss for alignment experiments; it is not selected
by default and is not a product change.

Requires ``config.loss_chunk_size: 0`` in the experiment YAML. HP's model routes
``self.loss_function`` through the chunked ``_projection_loss``/``projected_cross_entropy``
path whenever ``loss_chunk_size`` is nonzero, and through the unchunked ``_token_loss`` path
(one call with the full logits matrix) when it is ``0``. ``_dvm_ce_sum`` hard-checks the exact
``(4096, 152064)`` shape of one full, unchunked logits matrix -- HP's fixed
``vocab_size=152064`` (``examples/training_demo/jt_deepseek_v3/jt_deepseek_v3.yaml``'s
``model.config.vocab_size``) and ``sequence_length=4096`` (that YAML's 4096-token-packed
dataset, ``dataset.source_dataset.dataset_path: .../sft_4k_full``, and its baseline
``model.config.loss_chunk_size: 4096``, which chunks the loss into exactly one full-sequence
chunk) -- and raises if given a chunked view instead.
"""

# This adapter uses the Torch runtime, like the existing model and Trainer modules.
# pylint: disable=forbidden-backend-import

from typing import Any, Optional

import torch


def _dvm_sumx(values: torch.Tensor) -> torch.Tensor:
    """Reproduce measured SIMD64 SumX ordering for an aligned final axis."""
    if values.shape[-1] % 64:
        raise ValueError("DVM SumX replay requires a final axis divisible by 64")
    work = values.reshape(*values.shape[:-1], -1, 64)
    while work.shape[-2] > 1:
        length = work.shape[-2]
        half = length // 2
        reduced = work[..., :half, :] + work[..., half:2 * half, :]
        if length % 2:
            reduced[..., 0, :] = reduced[..., 0, :] + work[..., -1, :]
        work = reduced
    work = work.squeeze(-2)
    while work.shape[-1] > 1:
        work = work[..., ::2] + work[..., 1::2]
    return work.squeeze(-1)


def _sequential_sum(values: torch.Tensor) -> torch.Tensor:
    """Accumulate the last axis in its observed iteration order."""
    result = values[..., 0].clone()
    for index in range(1, values.shape[-1]):
        result = result + values[..., index]
    return result


def _dvm_ce_sum(exponents: torch.Tensor) -> torch.Tensor:
    """Reproduce the measured 910B3 DVM schedule for the reference CE shape."""
    if exponents.shape != (4096, 152064):
        raise ValueError("Measured DVM CE schedule requires 4096 rows and vocabulary 152064")
    partials = _dvm_sumx(exponents.reshape(4096, 22, 6912))
    result = _sequential_sum(partials)
    # Tail rows match the measured sum of even-index and odd-index block accumulations.
    result[4080:] = _sequential_sum(partials[4080:, ::2]) + _sequential_sum(partials[4080:, 1::2])
    return result[:, None]


class MFGraphCrossEntropy(torch.autograd.Function):
    """Compute the stable log-softmax/NLL loss shared by the plain and DVM MF variants.

    :class:`MFDVMCrossEntropy` overrides only the two vocabulary-axis reductions in
    ``forward``; both share this ``backward``, whose softmax-minus-onehot derivative does
    not depend on reduction order.
    """

    @staticmethod
    def forward(ctx: Any, logits: torch.Tensor, labels: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        """Return the masked mean NLL loss over flattened ``logits``/``labels``."""
        ctx.input_shape, ctx.input_dtype = logits.shape, logits.dtype
        values = logits.reshape(-1, logits.shape[-1]).float()
        targets = labels.reshape(-1)
        mask = targets >= 0
        targets = targets.clamp_min(0)
        centered = values - values.amax(-1, keepdim=True)
        log_probabilities = centered - centered.exp().sum(-1, keepdim=True).log()
        denominator = denominator.float()
        ctx.save_for_backward(log_probabilities, targets, mask, denominator)
        losses = -log_probabilities.gather(1, targets[:, None]).squeeze(-1)
        return (losses * mask).sum() / denominator

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Return the softmax-minus-onehot gradient scaled by the saved denominator."""
        log_probabilities, targets, mask, denominator = ctx.saved_tensors
        derivative = log_probabilities.exp()
        derivative.scatter_add_(1, targets[:, None], -torch.ones_like(targets[:, None], dtype=torch.float32))
        sensitivity = (gradient / denominator) * mask.float()
        result = derivative * sensitivity[:, None]
        return result.to(ctx.input_dtype).reshape(ctx.input_shape), None, None


class MFDVMCrossEntropy(MFGraphCrossEntropy):
    """Evaluate HP's unchunked logits with MF's measured 910B3 DVM reduction order."""

    @staticmethod
    def forward(ctx: Any, logits: torch.Tensor, labels: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        """Return the masked mean NLL loss using the measured DVM reduction order."""
        ctx.input_shape, ctx.input_dtype = logits.shape, logits.dtype
        values = logits.reshape(-1, logits.shape[-1]).float()
        targets = labels.reshape(-1)
        mask = targets >= 0
        targets = targets.clamp_min(0)
        centered = values - values.amax(-1, keepdim=True)
        log_probabilities = centered - _dvm_ce_sum(centered.exp()).log()
        denominator = denominator.float()
        ctx.save_for_backward(log_probabilities, targets, mask, denominator)
        losses = -log_probabilities.gather(1, targets[:, None]).squeeze(-1)
        return _dvm_sumx(losses * mask) / denominator


def mf_dvm_cross_entropy(*, logits: torch.Tensor, labels: Optional[torch.Tensor], vocab_size: int,
                         shift_labels: torch.Tensor, num_items_in_batch: torch.Tensor,
                         ignore_index: int = -100, **kwargs: Any) -> torch.Tensor:
    """Keep the masked mean-loss ``loss_function`` contract while using the measured DVM schedule.

    Only valid for one full, unchunked ``(sequence_length, vocab_size)`` logits matrix, as
    HP's ``_token_loss`` passes when ``config.loss_chunk_size`` is ``0``; see the module
    docstring for why the chunked ``_projection_loss`` path cannot use this.

    Raises:
        ValueError: If ``ignore_index`` is not ``-100`` or ``logits``'s vocabulary dimension
            does not match ``vocab_size``.
    """
    del labels, kwargs
    if ignore_index != -100 or logits.shape[-1] != vocab_size:
        raise ValueError("Unexpected CE vocabulary or ignore index")
    return MFDVMCrossEntropy.apply(logits, shift_labels, num_items_in_batch)


def build_jt_model(**kwargs: Any) -> Any:
    """Build the JT model with the MF-parity DVM cross-entropy as its loss function.

    Builds the ordinary JT model exactly as ``jt_builder.build_jt_model`` does, then
    replaces its ``loss_function`` with :func:`mf_dvm_cross_entropy` so the unchunked LM loss
    uses MindFormers' measured DVM reduction order instead of a plain ``.sum(-1)``. Requires
    ``config.loss_chunk_size: 0``; see the module docstring.

    Args:
        **kwargs: Forwarded verbatim to ``jt_builder.build_jt_model`` (``config``,
            ``reference_weights``, ``distributed_setup`` and infrastructure options).

    Returns:
        The sharded JT model with ``loss_function`` set to :func:`mf_dvm_cross_entropy`.
    """
    from hyper_parallel.models.jt_deepseek_v3.adapter.jt_builder import (  # pylint: disable=C0415
        build_jt_model as _build_jt_model,
    )

    model = _build_jt_model(**kwargs)
    model.loss_function = mf_dvm_cross_entropy
    return model
