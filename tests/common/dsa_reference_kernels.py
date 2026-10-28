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
"""CPU reference kernels standing in for the NPU DeepSeek sparse-attention operators.

``DeepseekV32DSAAttention`` and its CP wrapper call three device operators
through the HyperParallel functional wrappers: the Lightning indexer, the
sparse flash attention and ``npu_rotary_mul``. This helper implements their
TND contracts with plain PyTorch so the replacement, its weight conversion,
and its TP/CP orchestration run on CPU against the Transformers attention.
The DSA KL-loss operator is deliberately absent: tests keep the indexer loss
disabled, and calling it raises.
"""
# Test helper for CPU workers; torch is required here by design.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

import contextlib
import sys
import types
from collections.abc import Iterator

import torch

_STUB_MODULES = ("torch_npu", "omni_training_custom_ops")
_DSA_MODULES = (
    "hyper_parallel.components.modules.dsa_attention",
    "hyper_parallel.models.deepseek_v32.adapter.distributed.context_parallel",
)
# Lazy export packages cache resolved symbols in their globals.
_LAZY_PACKAGES = ("hyper_parallel.components.functional", "hyper_parallel.components.modules")


def _sequence_spans(cumulative: torch.Tensor) -> list[tuple[int, int]]:
    """Convert cumulative ends without a leading zero into half-open spans."""
    spans, start = [], 0
    for end in cumulative.tolist():
        spans.append((start, int(end)))
        start = int(end)
    return spans


def _causal_allowed(query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
    """Bottom-right aligned causal mask (sparse_mode=3): the last query sees the last key."""
    query = torch.arange(query_length, device=device).unsqueeze(1) + (key_length - query_length)
    key = torch.arange(key_length, device=device).unsqueeze(0)
    return key <= query


def lightning_indexer(
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        *,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        sparse_count: int,
        sparse_mode: int = 3,
        layout_query: str = "TND",
        layout_key: str = "TND",
        **_: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select ``sparse_count`` causal keys per query by ``sum_h w_h * relu(q_h . k)``.

    Indices are relative to the start of each packed key sequence; slots
    without a causal candidate hold ``-1``.
    """
    if sparse_mode != 3 or layout_query != "TND" or layout_key != "TND":
        raise NotImplementedError("reference indexer supports causal TND inputs only")
    tokens = query.shape[0]
    indices = torch.full((tokens, 1, sparse_count), -1, dtype=torch.int32, device=query.device)
    values = torch.full((tokens, 1, sparse_count), float("-inf"), dtype=torch.float32, device=query.device)
    query_spans = _sequence_spans(actual_seq_lengths_query)
    key_spans = _sequence_spans(actual_seq_lengths_key)
    for (q_start, q_end), (k_start, k_end) in zip(query_spans, key_spans):
        query_length, key_length = q_end - q_start, k_end - k_start
        if query_length == 0 or key_length == 0:
            continue
        scores = torch.relu(torch.einsum(
            "qnd,kd->qnk", query[q_start:q_end].float(), key[k_start:k_end, 0].float()
        ))
        scores = torch.einsum("qn,qnk->qk", weights[q_start:q_end].float(), scores)
        scores = scores.masked_fill(~_causal_allowed(query_length, key_length, query.device), float("-inf"))
        count = min(sparse_count, key_length)
        top_values, top_indices = scores.topk(count, dim=-1)
        top_indices = top_indices.masked_fill(torch.isinf(top_values), -1)
        indices[q_start:q_end, 0, :count] = top_indices.to(torch.int32)
        values[q_start:q_end, 0, :count] = top_values
    return indices, values


def sparse_flash_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        sparse_indices: torch.Tensor,
        scale_value: float,
        *,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
        query_rope: torch.Tensor,
        key_rope: torch.Tensor,
        sparse_mode: int = 3,
        layout_query: str = "TND",
        layout_kv: str = "TND",
        **_: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attend each query to its selected latent keys (MLA absorbed form).

    Scores are ``(q . k + q_rope . k_rope) * scale`` over the selected keys and
    the output is the softmax-weighted latent value. Returns the output plus
    the per-head softmax maximum and sum.
    """
    if sparse_mode != 3 or layout_query != "TND" or layout_kv != "TND":
        raise NotImplementedError("reference sparse attention supports causal TND inputs only")
    outputs, maxima, sums = [], [], []
    query_spans = _sequence_spans(actual_seq_lengths_query)
    key_spans = _sequence_spans(actual_seq_lengths_kv)
    for (q_start, q_end), (k_start, _) in zip(query_spans, key_spans):
        if q_end == q_start:
            continue
        selected = sparse_indices[q_start:q_end, 0].long()
        valid = selected >= 0
        rows = k_start + selected.clamp_min(0)
        scores = torch.einsum("qnd,qkd->qnk", query[q_start:q_end], key[rows, 0])
        scores = scores + torch.einsum("qnd,qkd->qnk", query_rope[q_start:q_end], key_rope[rows, 0])
        scores = (scores * scale_value).float().masked_fill(~valid.unsqueeze(1), float("-inf"))
        maximum = scores.amax(dim=-1)
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("qnk,qkd->qnd", probabilities.to(value.dtype), value[rows, 0]))
        maxima.append(maximum)
        sums.append(torch.exp(scores - maximum.unsqueeze(-1)).sum(dim=-1))
    return torch.cat(outputs), torch.cat(maxima), torch.cat(sums)


def _unsupported_indexer_loss(*args: object, **kwargs: object) -> None:
    """Reject the KL-loss operator, whose reference is out of scope."""
    del args, kwargs
    raise NotImplementedError("the CPU DSA reference does not implement the indexer KL loss")


def rotary_mul(
        tensor: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rotary_mode: str = "half",
) -> torch.Tensor:
    """Reference ``torch_npu.npu_rotary_mul`` for the half and interleave layouts."""
    if rotary_mode == "half":
        first, second = tensor.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
    elif rotary_mode == "interleave":
        rotated = torch.stack((-tensor[..., 1::2], tensor[..., 0::2]), dim=-1).flatten(-2)
    else:
        raise NotImplementedError(f"unsupported rotary_mode {rotary_mode!r}")
    return tensor * cos + rotated * sin


def _stub_modules() -> dict[str, types.ModuleType]:
    """Build import stand-ins exposing only what the DSA path calls."""
    torch_npu = types.ModuleType("torch_npu")
    torch_npu.npu_rotary_mul = rotary_mul
    return {"torch_npu": torch_npu, "omni_training_custom_ops": types.ModuleType("omni_training_custom_ops")}


def _binds_stub(module: types.ModuleType, stubs: dict[str, types.ModuleType]) -> bool:
    """Return whether a module captured one of the stand-in modules at import."""
    stub_ids = {id(stub) for stub in stubs.values()}
    return any(id(value) in stub_ids for value in vars(module).values())


def _is_refreshed(name: str) -> bool:
    """Return whether a module must be imported afresh against the stand-ins."""
    return name in _STUB_MODULES or name in _DSA_MODULES or (
        name.startswith(_LAZY_PACKAGES) and name not in _LAZY_PACKAGES
    )


def _forget_cached_dsa_modules(packages: dict[str, dict]) -> None:
    """Drop cached NPU-dependent modules and the symbols lazy packages resolved from them."""
    for name in [name for name in sys.modules if _is_refreshed(name)]:
        del sys.modules[name]
    for name in packages:
        package = vars(sys.modules[name])
        for attribute, value in list(package.items()):
            origin = value.__name__ if isinstance(value, types.ModuleType) else getattr(value, "__module__", "")
            if isinstance(origin, str) and origin.startswith(f"{name}."):
                del package[attribute]


def _restore_modules(
        previous_modules: dict[str, types.ModuleType],
        previous_packages: dict[str, dict],
        stubs: dict[str, types.ModuleType],
) -> None:
    """Evict modules bound to the stand-ins and restore the previous import state."""
    for name, module in list(sys.modules.items()):
        if name not in previous_modules and (name in _DSA_MODULES or _binds_stub(module, stubs)):
            del sys.modules[name]
            parent, _, child = name.rpartition(".")
            if getattr(sys.modules.get(parent), child, None) is module:
                delattr(sys.modules[parent], child)
    for name, module in previous_modules.items():
        if _is_refreshed(name) and sys.modules.get(name) is not module:
            sys.modules[name] = module
            parent, _, child = name.rpartition(".")
            if parent in sys.modules:
                setattr(sys.modules[parent], child, module)
    for name, contents in previous_packages.items():
        package = vars(sys.modules[name])
        package.clear()
        package.update(contents)


@contextlib.contextmanager
def reference_dsa_kernels() -> Iterator[None]:
    """Run the DSA replacement and its CP wrapper on CPU reference kernels.

    The NPU-dependent modules are imported afresh against stand-in
    ``torch_npu`` and custom-operator modules, so a stand-in left behind by
    another test cannot leak into them. The stand-ins leave ``sys.modules`` as
    soon as the imports finish and ``torch.ops.custom`` resolves to the
    reference kernels until the context exits. On exit every module that
    captured a stand-in is evicted and the previous modules and lazily cached
    package symbols are restored.
    """
    previous_modules = dict(sys.modules)
    previous_packages = {
        name: dict(vars(sys.modules[name])) for name in _LAZY_PACKAGES if name in sys.modules
    }
    previous_custom = torch.ops.__dict__.get("custom")
    stubs = _stub_modules()
    _forget_cached_dsa_modules(previous_packages)
    sys.modules.update(stubs)
    try:
        for module_name in _DSA_MODULES:
            __import__(module_name)
        setattr(torch.ops, "custom", types.SimpleNamespace(
            npu_lightning_indexer_enhance=lightning_indexer,
            npu_sparse_flash_attention_enhance=sparse_flash_attention,
            npu_sparse_lightning_indexer_grad_kl_loss_enhance=_unsupported_indexer_loss,
        ))
    finally:
        for name in _STUB_MODULES:
            sys.modules.pop(name, None)
    try:
        yield
    finally:
        if previous_custom is None:
            delattr(torch.ops, "custom")
        else:
            setattr(torch.ops, "custom", previous_custom)
        _restore_modules(previous_modules, previous_packages, stubs)


__all__ = [
    "lightning_indexer",
    "reference_dsa_kernels",
    "rotary_mul",
    "sparse_flash_attention",
]
