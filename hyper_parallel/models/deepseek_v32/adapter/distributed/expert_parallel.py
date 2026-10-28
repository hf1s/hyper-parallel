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
"""DeepSeek-V3.2 expert-parallel compute factory.

``DeepseekV32TopkRouter`` already returns the complete DeepSeek routing
decision ``(router_logits, topk_weights, topk_indices)``: sigmoid scores, the
``e_score_correction_bias`` used only for selection, group-limited top-k,
normalisation and ``routed_scaling_factor``. The factory consumes that decision
through the generic ``topk_router_module`` adapter instead of re-deriving it, so
the EP block evaluates the router once and selects exactly the experts and
weights of the non-EP block. Token dispatch, the local expert GEMMs and the
combine come from the generic EP skeleton; the shared experts are added to the
routed output as ``DeepseekV32MoE.forward`` does.
"""

from typing import Any, Callable, Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed.expert_parallel.recipes import build_ep_compute
from hyper_parallel.distributed.recipe_spec import local_compute


@local_compute
def deepseek_v32_ep_compute_fn(
        *,
        module: Any,
        mesh: Any,
        tp_mesh: Any,
        cp_mesh: Any,
        ep_mesh: Any,
        use_grouped_gemm: bool = False,
        overlap_shared_expert: bool = False,
        fix_router: bool = False,
        capacity_factor: Optional[float] = None,
) -> Callable:
    """Build the EP forward of one ``DeepseekV32MoE`` block.

    Expected module interface: ``gate`` returning ``(logits, weights,
    indices)``, batched ``experts`` and ``shared_experts``.

    Args:
        module: DeepSeek-V3.2 MoE block.
        mesh: Unused; part of the local-compute factory signature.
        tp_mesh: Unused.
        cp_mesh: Unused.
        ep_mesh: Expert mesh whose ``ep`` axis owns the routed experts.
        use_grouped_gemm: Run local experts through the grouped-GEMM path.
            Required when ``experts`` is the ``GroupedExperts`` replacement,
            whose weights use the transposed grouped-GEMM layout.
        overlap_shared_expert: Evaluate ``shared_experts`` while the dispatch
            exchange is in flight. The shared branch reads only
            ``hidden_states``, so the result is unchanged.
        fix_router: Benchmark-only balanced routing. Loss, gradient norm and
            expert-load statistics are meaningless in that mode.
        capacity_factor: Per-expert slot bound of the overlapped dispatch path;
            ``None`` keeps dropless routing.

    Returns:
        ``compute_fn(module, hidden_states)`` with the block forward signature.
    """
    del mesh, tp_mesh, cp_mesh

    def combine(
            module: Any,
            hidden_states: torch.Tensor,
            routed: torch.Tensor,
            overlapped: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add the shared-experts branch to the routed output."""
        shared = module.shared_experts(hidden_states) if overlapped is None else overlapped
        return routed + shared

    def overlap_shared(module: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the shared-experts branch behind the dispatch exchange."""
        return module.shared_experts(hidden_states)

    return build_ep_compute(
        module,
        ep_mesh,
        router_fn="topk_router_module",
        fix_router=fix_router,
        archetype_key="deepseek_v32_topk_router_shared",
        expected_attrs=["gate", "experts", "shared_experts"],
        combine=combine,
        use_grouped_gemm=use_grouped_gemm,
        capacity_factor=capacity_factor,
        overlap_fn=overlap_shared if overlap_shared_expert else None,
    )


__all__ = ["deepseek_v32_ep_compute_fn"]
