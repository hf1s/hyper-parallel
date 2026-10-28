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
"""CPU/Gloo DeepSeek-V3.2 TP/SP/CP/EP parity against a single-process reference.

Each case builds one small DeepSeek-V3.2 with fixed weights, converts a copy
through the Trainer chain (recipe ``plan_overrides`` -> module replacement ->
``ShardingPlanner`` -> ``apply_sharding_plan``) and runs one forward/backward
on the same tokens. The DSA cases run the NPU sparse-attention replacement on
the CPU reference kernels of ``tests.common.dsa_reference_kernels``.

FSDP is NPU/GPU-only, so the gradient reductions it owns are emulated: the
dense DP+CP domain averages, and only parameters whose final value needs no
TP-replicate reduction are compared (TP-sharded weights under sequence
parallelism and EP-sharded experts when no other rank feeds them tokens).
"""
# Distributed worker: torch and hyper_parallel imports are required here.
# pylint: disable=forbidden-backend-import

import contextlib
import copy
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import yaml
from transformers import DeepseekV32Config, DeepseekV32ForCausalLM

from tests.common.dsa_reference_kernels import reference_dsa_kernels

_ADAPTER = "hyper_parallel.models.deepseek_v32.adapter"
_SEQ_LEN = 16
_BATCH = 2


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
    """Return one dense layer followed by two MoE layers."""
    config = DeepseekV32Config.from_dict({
        "vocab_size": 128, "hidden_size": 64, "intermediate_size": 96, "moe_intermediate_size": 32,
        "num_hidden_layers": 3, "first_k_dense_replace": 1, "num_attention_heads": 4,
        "num_key_value_heads": 4, "n_shared_experts": 1, "n_routed_experts": 8, "n_group": 2,
        "topk_group": 1, "num_experts_per_tok": 2, "kv_lora_rank": 16, "q_lora_rank": 24,
        "qk_rope_head_dim": 8, "v_head_dim": 16, "qk_nope_head_dim": 16, "index_topk": 6,
        "index_head_dim": 16, "index_n_heads": 4, "max_position_embeddings": 64,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0}, "attn_implementation": "sdpa",
    })
    config.architectures = ["DeepseekV32ForCausalLM"]
    return config


def _reference_model() -> DeepseekV32ForCausalLM:
    """Build the unsharded model with nonzero expert-selection biases."""
    torch.manual_seed(7)
    model = DeepseekV32ForCausalLM(_config()).float().train()
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                buffer.copy_(torch.randn_like(buffer) * 0.05)
    return model


def _plan_overrides(dsa: bool) -> list:
    """Return the recipe parallel entries; NPU-only norm and expert kernels stay native."""
    entries = []
    if dsa:
        entries.append({
            "match": "*.self_attn",
            "module_type": "transformers.models.deepseek_v32.modeling_deepseek_v32.DeepseekV32Attention",
            "replace_module": {
                "_target_": f"{_ADAPTER}.conversion.module_replacement.replace_deepseek_v32_dsa_attention",
            },
        })
        entries.append({
            "match": "*.self_attn", "when": "cp", "region_dispatch": False, "inner_target": "self",
            "inner_wrapper": {
                "_target_": f"{_ADAPTER}.distributed.context_parallel.deepseek_v32_dsa_attention_cp_wrapper",
            },
        })
    entries.append({
        "match": "model.layers.[1-9].mlp", "when": "ep", "region_dispatch": False,
        "local_compute_fn": {"_target_": f"{_ADAPTER}.distributed.expert_parallel.deepseek_v32_ep_compute_fn"},
    })
    return entries


def _parse_trainer_config(tp, cp, ep, sequence_parallel, loss_parallel, dsa):
    """Resolve a minimal Trainer YAML so plan_overrides follow the recipe path."""
    from hyper_parallel.trainer.config.parser import parse_training_args  # pylint: disable=import-outside-toplevel

    recipe = {
        "model": {"_target_": "transformers.AutoConfig.from_pretrained"},
        "optimizer": {"_target_": "hyper_parallel.components.optim.AdamW", "adamw_config": {"lr": 1.0e-4}},
        "training": {"backend": "gloo", "global_batch_size": 1, "micro_batch_size": 1},
        "accelerator": {"tp_size": tp, "cp_size": cp, "ep_size": ep, "pp_size": 1,
                        "sequence_parallel": sequence_parallel, "loss_parallel": loss_parallel},
        "fsdp_config": {"dp_shard_size": 1, "edp_shard_size": 1},
        "plan_overrides": _plan_overrides(dsa),
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "recipe.yaml"
        path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
        return parse_training_args([str(path)])


def _build_parallel_model(reference, config, dsa):
    """Run replacement, planning and sharding on a copy of the reference weights."""
    # pylint: disable=import-outside-toplevel
    from hyper_parallel.components.checkpoint.weight_conversion import get_model_conversion_mapping
    from hyper_parallel.distributed.apply import apply_sharding_plan
    from hyper_parallel.models._transformers.model_builder import (
        _maybe_apply_module_replacements,
        instantiate_infrastructure,
    )
    from hyper_parallel.trainer.config import normalize_distributed_setup_overrides
    from hyper_parallel.trainer.runtime.distributed import create_distributed_setup_from_config

    setup = create_distributed_setup_from_config(config)
    normalize_distributed_setup_overrides(setup, config)
    mesh = setup.mesh_context
    model = copy.deepcopy(reference)
    model, _ = _maybe_apply_module_replacements(
        model, None, setup, None, False, get_model_conversion_mapping(model)
    )
    if dsa:
        for replaced, source in zip(model.model.layers, reference.model.layers):
            with torch.no_grad():
                replaced.self_attn.linear_qkv.weight.copy_(torch.cat([
                    source.self_attn.q_a_proj.weight, source.self_attn.kv_a_proj_with_mqa.weight,
                ]))
    planner, _ = instantiate_infrastructure(distributed_setup=setup, device=torch.device("cpu"))
    plan = planner.plan(
        model, mesh.device_mesh, tp_size=mesh.tp_size, cp_size=mesh.cp_size, ep_size=mesh.ep_size,
        sequence_parallel=mesh.sequence_parallel, loss_parallel=mesh.loss_parallel,
    )
    model, source_shard_info = apply_sharding_plan(model, plan, mesh)
    return model, mesh, source_shard_info or {}


def _reference_gradient(reference, name):
    """Return the reference gradient of one parallel-model parameter name."""
    named = dict(reference.named_parameters())
    if name.endswith("linear_qkv.weight"):
        prefix = name[: -len("linear_qkv.weight")]
        return torch.cat([named[prefix + "q_a_proj.weight"].grad, named[prefix + "kv_a_proj_with_mqa.weight"].grad])
    return named[name].grad


def _compared_gradients(model, reference, mesh, source_shard_info, *, check_experts):
    """Yield (name, local gradient, expected local gradient) for locally exact parameters.

    TP-replicated weights are compared only without TP, where the emulated
    DP+CP average is their complete reduction.
    """
    from hyper_parallel import distribute_tensor  # pylint: disable=import-outside-toplevel

    dense_group = mesh.dp_cp_mesh.get_group() if mesh.dp_cp_mesh is not None and mesh.dp_cp_mesh.size() > 1 else None
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        placements, source_mesh = source_shard_info.get(name, ((), None))
        sharded = any(not placement.is_replicate() for placement in placements)
        is_expert = source_mesh is not None and "ep" in (source_mesh.mesh_dim_names or ())
        if (not sharded and mesh.tp_size > 1) or (is_expert and not check_experts):
            continue
        grad = parameter.grad.clone()
        if not is_expert and dense_group is not None:
            dist.all_reduce(grad, op=dist.ReduceOp.AVG, group=dense_group)
        expected = _reference_gradient(reference, name)
        if sharded:
            expected = distribute_tensor(expected, source_mesh, placements, src_data_rank=None).to_local()
        yield name, grad, expected


def _run_case(*, tp=1, cp=1, ep=1, sequence_parallel=False, loss_parallel=False, dsa=False,
              check_gradients=True, check_experts=True):
    """Compare one parallel topology against the single-process model."""
    # pylint: disable=import-outside-toplevel
    dist.init_process_group("gloo")
    original_all_gather = dist_nn.all_gather
    dist_nn.all_gather = lambda tensor, group=dist.group.WORLD: _GlooAllGather.apply(group, tensor)
    kernels = reference_dsa_kernels() if dsa else contextlib.nullcontext()
    try:
        with kernels:
            from hyper_parallel.core.tensor_parallel import loss_parallel as loss_parallel_context
            from hyper_parallel.models._transformers.loss_parallel import causal_lm_loss_parallel

            reference = _reference_model()
            config = _parse_trainer_config(tp, cp, ep, sequence_parallel, loss_parallel, dsa)
            model, mesh, source_shard_info = _build_parallel_model(reference, config, dsa)
            if loss_parallel:
                model.loss_function = causal_lm_loss_parallel

            torch.manual_seed(11)
            tokens = torch.randint(1, 128, (_BATCH, _SEQ_LEN + 1))
            inputs, labels = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
            positions = torch.arange(_SEQ_LEN).unsqueeze(0).expand(_BATCH, -1)
            expected = reference(input_ids=inputs, labels=labels, shift_labels=labels,
                                 position_ids=positions, use_cache=False)
            expected.loss.backward()

            local = slice(mesh.cp_rank * _SEQ_LEN // cp, (mesh.cp_rank + 1) * _SEQ_LEN // cp)
            extra = {} if dsa else {"attention_mask": torch.ones(1, 1, _SEQ_LEN, _SEQ_LEN, dtype=torch.bool).tril()}
            forward_context = (
                loss_parallel_context(mesh=mesh.device_mesh["tp"]) if loss_parallel else contextlib.nullcontext()
            )
            with forward_context:
                output = model(
                    input_ids=inputs[:, local].contiguous(), labels=labels[:, local].contiguous(),
                    shift_labels=labels[:, local].contiguous(), position_ids=positions[:, local],
                    use_cache=False, **extra,
                )
                # Trainer loss rule: equal token counts cancel the DP*CP factor; SP divides by TP.
                (output.loss / (tp if sequence_parallel else 1)).backward()

            loss = output.loss.detach().reshape(()).clone()
            if cp > 1:
                dist.all_reduce(loss, op=dist.ReduceOp.AVG, group=mesh.device_mesh["cp"].get_group())
            torch.testing.assert_close(loss, expected.loss.detach(), rtol=1e-5, atol=1e-6)
            if check_gradients:
                compared = 0
                for name, grad, reference_grad in _compared_gradients(
                        model, reference, mesh, source_shard_info, check_experts=check_experts):
                    torch.testing.assert_close(grad, reference_grad, rtol=1e-4, atol=1e-6, msg=name)
                    compared += 1
                assert compared, "no parallel gradient was compared"
    finally:
        dist_nn.all_gather = original_all_gather
        dist.destroy_process_group()


def test_tp2_sp_ep2_transformers_attention():
    """TP2 + sequence parallel + EP2 on the Transformers attention and experts."""
    _run_case(tp=2, ep=2, sequence_parallel=True)


def test_cp2_dsa():
    """CP2 through the DSA all-gather wrapper; every gradient is compared."""
    _run_case(cp=2, dsa=True)


def test_tp2_ep2_dsa_loss_parallel():
    """TP2 + loss parallel + EP2 on the DSA replacement (forward parity)."""
    _run_case(tp=2, ep=2, loss_parallel=True, dsa=True, check_gradients=False)


def test_tp2_cp2_ep4_dsa():
    """TP2 x CP2 with EP4 over the DSA replacement on four ranks (forward parity)."""
    _run_case(tp=2, cp=2, ep=4, dsa=True, check_gradients=False)
