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
"""CPU/Gloo DeepSeek-V3.2 TP/CP/EP/FSDP parity against a single-process reference.

Each case builds one small DeepSeek-V3.2 training model with fixed weights,
one MTP depth and the sequence-wise balance loss, converts a copy through the
Trainer chain (recipe ``plan_overrides`` -> module replacement ->
``ShardingPlanner`` -> ``apply_sharding_plan`` -> FSDP2) and runs one
forward/backward on the same tokens. DSA runs the NPU sparse-attention
replacement on the CPU reference kernels of ``tests.common.dsa_reference_kernels``;
the NPU-only RMSNorm and grouped-expert kernels stay native.

FSDP2 owns the Trainer's gradient reduction (SUM over the DP x CP domain and
over TP replicas, scaled by 1 / (dp * cp * (1 if loss_parallel else tp))), so
the raw losses are backpropagated and every gradient must equal the
single-process gradient. Expert loads summed over the DP x CP domain must equal
the single-process loads.
"""
# Distributed worker: torch and hyper_parallel imports are required here.
# pylint: disable=forbidden-backend-import

import contextlib
import copy
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import yaml
from transformers import DeepseekV32Config

from tests.common.dsa_reference_kernels import reference_dsa_kernels

_ADAPTER = "hyper_parallel.models.deepseek_v32.adapter"
_SEQ_LEN = 16
_BATCH = 2
# Packed documents over the flattened [batch, sequence] stream; two of them
# straddle the CP2 split of their row (positions 8 and 24).
_PACKED_BOUNDARIES = (0, 5, 16, 19, 27, 32)


class _GlooAllGather(torch.autograd.Function):
    """Differentiable all-gather whose backward is valid on every Gloo subgroup.

    ``torch.distributed.nn.functional.all_gather`` emulates its backward
    reduce-scatter on Gloo with ``scatter`` calls that address global rank 0,
    which fails on TP or CP subgroups that do not contain it.
    """

    @staticmethod
    def forward(ctx: Any, group: Any, tensor: torch.Tensor) -> tuple:  # pylint: disable=arguments-differ
        """Gather every rank's shard."""
        tensor = tensor.contiguous()
        ctx.group = group
        gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
        dist.all_gather(gathered, tensor, group=group)
        return tuple(gathered)

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor) -> tuple:  # pylint: disable=arguments-differ
        """Sum every rank's output gradient and keep this rank's slot."""
        stacked = torch.stack([grad.contiguous() for grad in grads])
        dist.all_reduce(stacked, group=ctx.group)
        return None, stacked[dist.get_rank(ctx.group)]


def _config() -> DeepseekV32Config:
    """Return one dense layer followed by two MoE layers and one MoE MTP depth."""
    config = DeepseekV32Config.from_dict({
        "vocab_size": 128, "hidden_size": 64, "intermediate_size": 96, "moe_intermediate_size": 32,
        "num_hidden_layers": 3, "first_k_dense_replace": 1, "num_attention_heads": 4,
        "num_key_value_heads": 4, "n_shared_experts": 1, "n_routed_experts": 8, "n_group": 2,
        "topk_group": 1, "num_experts_per_tok": 2, "kv_lora_rank": 16, "q_lora_rank": 24,
        "qk_rope_head_dim": 8, "v_head_dim": 16, "qk_nope_head_dim": 16, "index_topk": 6,
        "index_head_dim": 16, "index_n_heads": 4, "max_position_embeddings": 64,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0}, "attn_implementation": "sdpa",
        "num_nextn_predict_layers": 1,
    })
    config.architectures = ["DeepseekV32ForCausalLM"]
    return config


def _reference_model():
    """Build the unsharded training model with nonzero expert-selection biases."""
    from hyper_parallel.models.deepseek_v32.modeling_deepseek_v32 import (  # pylint: disable=import-outside-toplevel
        DeepseekV32ForCausalLM,
    )

    torch.manual_seed(7)
    model = DeepseekV32ForCausalLM(
        _config(), moe_aux_loss_coeff=0.1, moe_router_enable_expert_bias=True,
    ).float().train()
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                buffer.copy_(torch.randn_like(buffer) * 0.05)
    return model


def _plan_overrides() -> list:
    """Return the recipe's DSA, MTP, CP and EP entries; NPU-only norm and expert kernels stay native."""
    return [
        {"match": "mtp.layers.*.eh_proj", "params": {"weight": {"tp": "replicate"}}},
        {
            "match": "*.self_attn",
            "replace": {"kind": "module", "implementation": {
                "_target_": f"{_ADAPTER}.conversion.module_replacement.replace_deepseek_v32_dsa_attention",
            }},
        },
        {
            "match": "*.self_attn.forward", "when": "cp",
            "replace": {"kind": "method", "implementation": {
                "_target_": f"{_ADAPTER}.distributed.context_parallel.deepseek_v32_dsa_attention_cp_wrapper",
            }},
        },
        {
            "match": "model.layers.[1-9].mlp.forward", "when": "ep",
            "replace": {"kind": "method", "implementation": {
                "_target_": "hyper_parallel.distributed.expert_parallel.recipes.deepseekv3_ep_compute_fn",
                "router_fn": "topk_router_module", "overlap_shared_expert": True,
            }},
        },
        {
            "match": "mtp.layers.*.mtp_block.mlp.forward", "when": "ep",
            "replace": {"kind": "method", "implementation": {
                "_target_": "hyper_parallel.distributed.expert_parallel.recipes.deepseekv3_ep_compute_fn",
                "router_fn": "topk_router_module", "overlap_shared_expert": True,
            }},
        },
    ]


def _parse_trainer_config(tp, cp, ep, loss_parallel):
    """Resolve a minimal Trainer YAML so plan_overrides follow the recipe path."""
    from hyper_parallel.trainer.config.parser import parse_training_args  # pylint: disable=import-outside-toplevel

    recipe = {
        "model": {"_target_": "transformers.AutoConfig.from_pretrained"},
        "optimizer": {"_target_": "hyper_parallel.components.optim.AdamW", "adamw_config": {"lr": 1.0e-4}},
        "training": {"backend": "gloo", "global_batch_size": 1, "micro_batch_size": 1},
        "accelerator": {"tp_size": tp, "cp_size": cp, "ep_size": ep, "pp_size": 1,
                        "sequence_parallel": False, "loss_parallel": loss_parallel},
        "fsdp_config": {
            "dp_shard_size": 1, "edp_shard_size": 1,
            "mix_precision": {"param_dtype": "float32", "reduce_dtype": "float32"},
        },
        "plan_overrides": _plan_overrides(),
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "recipe.yaml"
        path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
        return parse_training_args([str(path)])


def _build_parallel_model(reference, config):
    """Run replacement, planning, sharding and FSDP2 on a copy of the reference weights."""
    # pylint: disable=import-outside-toplevel
    from hyper_parallel.components.checkpoint.weight_conversion import get_model_conversion_mapping
    from hyper_parallel.distributed.apply import apply_sharding_plan
    from hyper_parallel.models._transformers.model_builder import (
        _apply_fsdp2_with_external,
        _maybe_apply_module_replacements,
        instantiate_infrastructure,
    )
    from hyper_parallel.models.deepseek_v32.modeling_deepseek_v32 import bind_parallel_groups
    from hyper_parallel.trainer.config import normalize_distributed_setup_overrides
    from hyper_parallel.trainer.runtime.distributed import create_distributed_setup_from_config

    setup = create_distributed_setup_from_config(config)
    normalize_distributed_setup_overrides(setup, config)
    mesh = setup.mesh_context
    model = copy.deepcopy(reference)
    model, _ = _maybe_apply_module_replacements(
        model, None, setup, None, False, get_model_conversion_mapping(model)
    )
    replaced_layers = [*model.model.layers, *(layer.mtp_block for layer in model.mtp.layers)]
    source_layers = [*reference.model.layers, *(layer.mtp_block for layer in reference.mtp.layers)]
    for replaced, source in zip(replaced_layers, source_layers):
        with torch.no_grad():
            replaced.self_attn.linear_qkv.weight.copy_(torch.cat([
                source.self_attn.q_a_proj.weight, source.self_attn.kv_a_proj_with_mqa.weight,
            ]))
    planner, fsdp_manager = instantiate_infrastructure(distributed_setup=setup, device=torch.device("cpu"))
    plan = planner.plan(
        model, mesh.device_mesh, tp_size=mesh.tp_size, cp_size=mesh.cp_size, ep_size=mesh.ep_size,
        sequence_parallel=False, loss_parallel=mesh.loss_parallel,
    )
    model, source_shard_info = apply_sharding_plan(model, plan, mesh)
    model = _apply_fsdp2_with_external(model, fsdp_manager, source_shard_info, None)
    bind_parallel_groups(model, mesh)
    return model, mesh, source_shard_info or {}


def _reference_gradient(reference, name):
    """Return the reference gradient of one parallel-model parameter name."""
    named = dict(reference.named_parameters())
    if name.endswith("linear_qkv.weight"):
        prefix = name[: -len("linear_qkv.weight")]
        return torch.cat([named[prefix + "q_a_proj.weight"].grad, named[prefix + "kv_a_proj_with_mqa.weight"].grad])
    return named[name].grad


def _compared_gradients(model, reference, source_shard_info):
    """Yield (name, local gradient, expected local gradient) for every parameter with a gradient."""
    from hyper_parallel import distribute_tensor  # pylint: disable=import-outside-toplevel

    for name, parameter in model.named_parameters():
        expected = _reference_gradient(reference, name)
        assert (parameter.grad is None) == (expected is None), f"{name}: gradient presence differs"
        if expected is None:
            continue
        grad = parameter.grad
        grad = grad.to_local() if hasattr(grad, "to_local") else grad
        placements, source_mesh = source_shard_info.get(name, ((), None))
        if any(not placement.is_replicate() for placement in placements):
            expected = distribute_tensor(expected, source_mesh, placements, src_data_rank=None).to_local()
        yield name, grad, expected


def _block_diagonal_mask(boundaries):
    """Return the causal per-document mask of packed boundaries over [batch, sequence]."""
    mask = torch.zeros(_BATCH, 1, _SEQ_LEN, _SEQ_LEN, dtype=torch.bool)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        row, first = divmod(start, _SEQ_LEN)
        last = end - row * _SEQ_LEN
        mask[row, 0, first:last, first:last] = torch.ones(last - first, last - first, dtype=torch.bool).tril()
    return mask


def _expert_loads(model):
    """Return each training router's routed-token counts by module name."""
    from hyper_parallel.models.deepseek_v32.modeling_deepseek_v32 import (  # pylint: disable=import-outside-toplevel
        DeepseekV32Router,
    )

    return {name: module.tokens_per_expert for name, module in model.named_modules()
            if isinstance(module, DeepseekV32Router)}


def _run_case(*, tp=1, cp=1, ep=1, loss_parallel=False, packed=False):
    """Compare one parallel topology against the single-process model."""
    # pylint: disable=import-outside-toplevel
    dist.init_process_group("gloo")
    original_all_gather = dist_nn.all_gather
    dist_nn.all_gather = lambda tensor, group=dist.group.WORLD: _GlooAllGather.apply(group, tensor)
    try:
        # FSDP2 resolves its device from the mesh; the Gloo mesh lives on CPU.
        with reference_dsa_kernels(), patch(
                "hyper_parallel.core.fully_shard.api._get_device_from_mesh", return_value=torch.device("cpu")):
            from hyper_parallel.core.tensor_parallel import loss_parallel as loss_parallel_context
            from hyper_parallel.models._transformers.loss_parallel import causal_lm_loss_parallel
            from hyper_parallel.models.deepseek_v32.adapter.data.runtime import DeepseekV32PackedSequence

            reference = _reference_model()
            config = _parse_trainer_config(tp, cp, ep, loss_parallel)
            model, mesh, source_shard_info = _build_parallel_model(reference, config)
            if loss_parallel:
                model.loss_function = causal_lm_loss_parallel

            torch.manual_seed(11)
            tokens = torch.randint(1, 128, (_BATCH, _SEQ_LEN + 1))
            inputs, labels = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
            positions = torch.arange(_SEQ_LEN).unsqueeze(0).expand(_BATCH, -1)
            reference_mask = {"attention_mask": _block_diagonal_mask(_PACKED_BOUNDARIES)} if packed else {}
            expected = reference(input_ids=inputs, labels=labels, shift_labels=labels,
                                 position_ids=positions, use_cache=False, **reference_mask)
            sum(expected.loss.values()).backward()

            local = slice(mesh.cp_rank * _SEQ_LEN // cp, (mesh.cp_rank + 1) * _SEQ_LEN // cp)
            boundaries = {}
            if packed:
                boundaries = DeepseekV32PackedSequence().build_runtime_inputs(
                    batch={"cu_seq_lens": torch.tensor(_PACKED_BOUNDARIES, dtype=torch.int32)}, context=None,
                )
            forward_context = (
                loss_parallel_context(mesh=mesh.device_mesh["tp"]) if loss_parallel else contextlib.nullcontext()
            )
            with forward_context:
                output = model(
                    input_ids=inputs[:, local].contiguous(), labels=labels[:, local].contiguous(),
                    shift_labels=labels[:, local].contiguous(), position_ids=positions[:, local],
                    use_cache=False, **boundaries,
                )
                sum(output.loss.values()).backward()

            same_keys = set(output.loss) == set(expected.loss) == {"foundation_loss", "foundation_loss/mtp"}
            assert same_keys, f"loss keys differ: {set(output.loss)} vs {set(expected.loss)}"
            for key, value in output.loss.items():
                loss = value.detach().reshape(()).clone()
                if cp > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.AVG, group=mesh.device_mesh["cp"].get_group())
                torch.testing.assert_close(loss, expected.loss[key].detach(), rtol=1e-5, atol=1e-6, msg=key)
            compared = 0
            for name, grad, reference_grad in _compared_gradients(model, reference, source_shard_info):
                torch.testing.assert_close(grad, reference_grad, rtol=1e-4, atol=1e-6, msg=name)
                compared += 1
            assert compared, "no parallel gradient was compared"
            # The Trainer's MoE monitor sums expert loads over DP x CP before the bias update.
            dp_cp_mesh = mesh.dp_cp_mesh
            expected_loads = _expert_loads(reference)
            loads = _expert_loads(model)
            assert loads.keys() == expected_loads.keys() and loads, "router sets differ"
            for name, counts in loads.items():
                counts = counts.clone()
                if dp_cp_mesh is not None and dp_cp_mesh.size() > 1:
                    dist.all_reduce(counts, group=dp_cp_mesh.get_group())
                torch.testing.assert_close(counts, expected_loads[name], msg=name)
    finally:
        dist_nn.all_gather = original_all_gather
        dist.destroy_process_group()


def test_cp2_dsa():
    """CP2 through the DSA all-gather wrapper."""
    _run_case(cp=2)


def test_cp2_dsa_packed():
    """CP2 with packed documents straddling the CP split, boundaries from the runtime adapter."""
    _run_case(cp=2, packed=True)


def test_tp2_ep2_dsa_loss_parallel():
    """TP2 + loss parallel + EP2 at cp_size=1, so the plain DSA forward runs."""
    _run_case(tp=2, ep=2, loss_parallel=True)


def test_tp2_cp2_ep4_dsa():
    """TP2 x CP2 with EP4 over the DSA replacement on four ranks."""
    _run_case(tp=2, cp=2, ep=4)
