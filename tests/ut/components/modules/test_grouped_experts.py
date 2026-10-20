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
"""CPU unit tests for the real GroupedExperts forward mathematics."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import torch
from torch import nn

from hyper_parallel.components import functional


def _cpu_grouped_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    group_list: torch.Tensor,
    group_type: int = 0,
    group_list_type: int = 0,
) -> torch.Tensor:
    """Run a CPU grouped matmul using cumulative expert boundaries."""
    del group_type, group_list_type
    outputs: list[torch.Tensor] = []
    start = 0
    for expert, end_tensor in enumerate(group_list.tolist()):
        end = int(end_tensor)
        if end > start:
            out = torch.matmul(x[start:end], w[expert])
            if bias is not None:
                out = out + bias[expert]
            outputs.append(out)
        start = end
    if outputs:
        return torch.cat(outputs, dim=0)
    return x.new_empty((0, w.shape[-1]))


def _cpu_swiglu(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply SwiGLU on CPU."""
    gate, up = x.chunk(2, dim=dim)
    return torch.nn.functional.silu(gate) * up


def _cpu_moe_token_permute(
    x: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Permute tokens by expert and return the original-route to permuted-row inverse."""
    top_k = indices.shape[1]
    order = indices.reshape(-1).argsort(stable=True)
    output = x[order // top_k]
    inverse = order.argsort()
    return output, inverse


def _cpu_moe_token_unpermute(
    permuted: torch.Tensor,
    inverse: torch.Tensor,
    probs: torch.Tensor,
) -> torch.Tensor:
    """Restore token order and apply routing probabilities on CPU."""
    num_tokens, top_k = probs.shape
    restored = permuted.index_select(0, inverse).reshape(
        num_tokens,
        top_k,
        permuted.shape[-1],
    )
    return (restored * probs.unsqueeze(-1)).sum(dim=1)


_CPU_HELPERS = {
    "grouped_matmul": _cpu_grouped_matmul,
    "swiglu": _cpu_swiglu,
    "moe_token_permute": _cpu_moe_token_permute,
    "moe_token_unpermute": _cpu_moe_token_unpermute,
}


def _load_grouped_experts_module() -> Any:
    """Load the real grouped_experts module while binding CPU functional helpers."""
    module_path = (
        Path(__file__).resolve().parents[4]
        / "hyper_parallel"
        / "components"
        / "modules"
        / "grouped_experts.py"
    )
    spec = importlib.util.spec_from_file_location("_test_grouped_experts_real", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load grouped experts module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(functional.__dict__, _CPU_HELPERS):
        spec.loader.exec_module(module)
    return module


def _make_gate_up() -> torch.Tensor:
    """Return deterministic gate/up expert weights."""
    return torch.tensor(
        [
            [[0.11, -0.23, 0.37, 0.41], [-0.52, 0.63, -0.74, 0.85]],
            [[-0.19, 0.29, -0.31, 0.43], [0.53, -0.61, 0.71, -0.83]],
        ],
        dtype=torch.float64,
    )


def _make_down() -> torch.Tensor:
    """Return deterministic down-projection expert weights."""
    return torch.tensor(
        [
            [[0.17, -0.27], [0.37, 0.47]],
            [[-0.57, 0.67], [-0.77, 0.87]],
        ],
        dtype=torch.float64,
    )


def _make_bias1() -> torch.Tensor:
    """Return deterministic non-zero first-projection expert biases."""
    return torch.tensor(
        [
            [0.13, -0.21, 0.35, -0.47],
            [-0.59, 0.61, -0.73, 0.83],
        ],
        dtype=torch.float64,
    )


def _make_bias2() -> torch.Tensor:
    """Return deterministic non-zero second-projection expert biases."""
    return torch.tensor(
        [
            [0.19, -0.29],
            [0.39, 0.49],
        ],
        dtype=torch.float64,
    )


def _make_hidden_states() -> torch.Tensor:
    """Return mixed-sign token states."""
    return torch.tensor(
        [
            [0.25, -0.75],
            [-0.50, 0.90],
            [1.10, -0.30],
        ],
        dtype=torch.float64,
    )


def _make_top_k_index(top_k: int) -> torch.Tensor:
    """Return mixed-expert routing indices for three tokens."""
    if top_k == 1:
        return torch.tensor([[0], [1], [0]], dtype=torch.long)
    return torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.long)


def _make_top_k_weights(top_k: int) -> torch.Tensor:
    """Return non-uniform routing weights for three tokens."""
    if top_k == 1:
        return torch.tensor([[0.35], [0.65], [0.80]], dtype=torch.float64)
    return torch.tensor(
        [
            [0.20, 0.70],
            [0.55, 0.40],
            [0.90, 0.10],
        ],
        dtype=torch.float64,
    )


def _make_dy() -> torch.Tensor:
    """Return a non-uniform upstream gradient."""
    return torch.tensor(
        [
            [0.30, -0.70],
            [1.10, 0.20],
            [-0.50, 0.90],
        ],
        dtype=torch.float64,
    )


def _make_expert_major_input() -> torch.Tensor:
    """Return expert-major mixed-sign token states."""
    return torch.tensor(
        [
            [0.25, -0.75],
            [-0.50, 0.90],
            [1.10, -0.30],
        ],
        dtype=torch.float64,
    )


def _make_expert_major_scores() -> torch.Tensor:
    """Return non-uniform expert-major routing scores."""
    return torch.tensor([0.25, 0.75, 0.60], dtype=torch.float64)


class _FakeSourceExperts(nn.Module):
    """Minimal source experts module consumed by the real GroupedExperts constructor."""

    def __init__(self, *, with_bias: bool, router_gating_in_fp32: bool) -> None:
        """Initialize deterministic CPU parameters and source attributes."""
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=2,
            moe_intermediate_size=2,
            n_routed_experts=2,
            hidden_act="silu",
            initializer_range=0.02,
        )
        self.num_experts = 2
        self.hidden_size = 2
        self.intermediate_size = 2
        self.is_transposed = True
        self.router_gating_in_fp32 = router_gating_in_fp32
        self.gate_up_proj = nn.Parameter(_make_gate_up())
        self.down_proj = nn.Parameter(_make_down())
        if with_bias:
            self.bias1 = nn.Parameter(_make_bias1())
            self.bias2 = nn.Parameter(_make_bias2())


def _reference_forward(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    bias1: torch.Tensor | None,
    bias2: torch.Tensor | None,
) -> torch.Tensor:
    """Compute the token-route reference MLP output."""
    num_tokens = hidden_states.shape[0]
    top_k = top_k_index.shape[1]
    token_outputs: list[torch.Tensor] = []
    for token in range(num_tokens):
        token_output = hidden_states.new_zeros(hidden_states.shape[1])
        for route in range(top_k):
            expert = int(top_k_index[token, route].item())
            fc1 = hidden_states[token] @ gate_up_proj[expert]
            if bias1 is not None:
                fc1 = fc1 + bias1[expert]
            gate, up = fc1.chunk(2, dim=-1)
            activated = torch.nn.functional.silu(gate) * up
            expert_output = activated @ down_proj[expert]
            if bias2 is not None:
                expert_output = expert_output + bias2[expert]
            token_output = token_output + expert_output * top_k_weights[token, route]
        token_outputs.append(token_output)
    return torch.stack(token_outputs, dim=0)


def _reference_expert_major(
    x: torch.Tensor,
    counts: torch.Tensor,
    scores: torch.Tensor | None,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    bias1: torch.Tensor | None,
    bias2: torch.Tensor | None,
    router_gating_in_fp32: bool,
) -> torch.Tensor:
    """Compute the expert-major reference with interface-specific score placement."""
    outputs: list[torch.Tensor] = []
    start = 0
    for expert, count_tensor in enumerate(counts.tolist()):
        count = int(count_tensor)
        end = start + count
        if end > start:
            fc1 = x[start:end] @ gate_up_proj[expert]
            if bias1 is not None:
                fc1 = fc1 + bias1[expert]
            gate, up = fc1.chunk(2, dim=-1)
            activated = torch.nn.functional.silu(gate) * up
            if not router_gating_in_fp32 and scores is not None:
                activated = activated * scores[start:end].reshape(-1, 1)
            expert_output = activated @ down_proj[expert]
            if bias2 is not None:
                expert_output = expert_output + bias2[expert]
            outputs.append(expert_output)
        start = end
    if outputs:
        return torch.cat(outputs, dim=0)
    return x.new_empty((0, down_proj.shape[-1]))


class TestGroupedExpertsForward(unittest.TestCase):
    """Validate GroupedExperts forward mathematics on CPU."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the real grouped_experts module once for all tests."""
        cls._module = _load_grouped_experts_module()
        cls._grouped_experts_cls = cls._module.GroupedExperts

    def _make_grouped_experts(self, router_gating_in_fp32: bool, with_bias: bool) -> nn.Module:
        """Instantiate the real GroupedExperts with deterministic CPU parameters."""
        source = _FakeSourceExperts(
            with_bias=with_bias,
            router_gating_in_fp32=router_gating_in_fp32,
        )
        module = self._grouped_experts_cls(module=source)
        return module

    def test_forward_applies_each_route_probability_once(self) -> None:
        """Check forward outputs and gradients against a token-route reference."""
        for router_gating_in_fp32 in (True, False):
            for with_bias in (False, True):
                for top_k in (1, 2):
                    with self.subTest(
                        router_gating_in_fp32=router_gating_in_fp32,
                        with_bias=with_bias,
                        top_k=top_k,
                    ):
                        self._check_forward_contract(
                            router_gating_in_fp32=router_gating_in_fp32,
                            with_bias=with_bias,
                            top_k=top_k,
                        )

    def _check_forward_contract(
        self,
        *,
        router_gating_in_fp32: bool,
        with_bias: bool,
        top_k: int,
    ) -> None:
        """Compare forward values and gradients for one parameterization."""
        module = self._make_grouped_experts(router_gating_in_fp32, with_bias)
        hidden_states = _make_hidden_states().requires_grad_(True)
        top_k_index = _make_top_k_index(top_k)
        top_k_weights = _make_top_k_weights(top_k).requires_grad_(True)
        reference_hidden = hidden_states.detach().clone().requires_grad_(True)
        reference_weights = top_k_weights.detach().clone().requires_grad_(True)

        actual_params = [module.gate_up_proj, module.down_proj]
        if with_bias:
            actual_params.extend([module.bias1, module.bias2])
        reference_params = [
            parameter.detach().clone().requires_grad_(True) for parameter in actual_params
        ]

        actual_y = module(hidden_states, top_k_index, top_k_weights)
        reference_bias1 = reference_params[2] if with_bias else None
        reference_bias2 = reference_params[3] if with_bias else None
        reference_y = _reference_forward(
            reference_hidden,
            top_k_index,
            reference_weights,
            reference_params[0],
            reference_params[1],
            reference_bias1,
            reference_bias2,
        )
        torch.testing.assert_close(
            actual_y,
            reference_y,
            rtol=1e-12,
            atol=1e-12,
            msg=(
                f"flag={router_gating_in_fp32}, bias={with_bias}, K={top_k}: "
                f"actual={actual_y}, expected={reference_y}"
            ),
        )

        dy = _make_dy()
        actual_inputs = [hidden_states, top_k_weights, *actual_params]
        reference_inputs = [reference_hidden, reference_weights, *reference_params]
        actual_grads = torch.autograd.grad(actual_y, actual_inputs, grad_outputs=dy)
        reference_grads = torch.autograd.grad(reference_y, reference_inputs, grad_outputs=dy)

        labels = ["dx", "dprob", "dgate_up_proj", "ddown_proj"]
        if with_bias:
            labels.extend(["dbias1", "dbias2"])
        for label, actual_grad, reference_grad in zip(labels, actual_grads, reference_grads):
            torch.testing.assert_close(
                actual_grad,
                reference_grad,
                rtol=1e-12,
                atol=1e-12,
                msg=(
                    f"{label} mismatch: "
                    f"flag={router_gating_in_fp32}, bias={with_bias}, K={top_k}, "
                    f"actual={actual_grad}, expected={reference_grad}"
                ),
            )

    def test_forward_expert_major_score_placement(self) -> None:
        """Check forward_expert_major keeps its original score-placement contract."""
        x = _make_expert_major_input()
        counts = torch.tensor([2, 1], dtype=torch.long)
        scores = _make_expert_major_scores()
        for router_gating_in_fp32 in (True, False):
            with self.subTest(router_gating_in_fp32=router_gating_in_fp32):
                module = self._make_grouped_experts(router_gating_in_fp32, with_bias=True)
                actual = module.forward_expert_major(x, counts, scores)
                expected = _reference_expert_major(
                    x,
                    counts,
                    scores,
                    module.gate_up_proj,
                    module.down_proj,
                    module.bias1,
                    module.bias2,
                    router_gating_in_fp32,
                )
                torch.testing.assert_close(
                    actual, expected, rtol=1e-12, atol=1e-12,
                    msg=f"flag={router_gating_in_fp32}: actual={actual}, expected={expected}",
                )

                baseline = module.forward_expert_major(x, counts, None)
                if router_gating_in_fp32:
                    torch.testing.assert_close(
                        actual, baseline, rtol=1e-12, atol=1e-12,
                        msg=f"actual={actual}, unweighted={baseline}",
                    )
                else:
                    self.assertFalse(
                        torch.allclose(actual, baseline, rtol=1e-12, atol=1e-12),
                        msg=f"weighted={actual}, unweighted={baseline}",
                    )


if __name__ == "__main__":
    unittest.main()
