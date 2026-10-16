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
"""JT model and sharding fixtures shared by UT and CP workers."""

from hyper_parallel.trainer.config.resolver import resolve_config
from hyper_parallel.trainer.config.trainer import TrainerConfig


def jt_test_config() -> TrainerConfig:
    """Build independent model, MLA/EP/SP and optimizer settings without example YAMLs."""
    model_path = "hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3"
    adapter_path = "hyper_parallel.models.jt_deepseek_v3.adapter"
    config = {
        "architectures": ["JTDeepseekV3ForCausalLM"], "vocab_size": 512, "hidden_size": 256,
        "intermediate_size": 512, "moe_intermediate_size": 128, "num_hidden_layers": 2,
        "num_attention_heads": 8, "num_key_value_heads": 8, "n_shared_experts": 1, "n_routed_experts": 8,
        "routed_scaling_factor": 2.827, "kv_lora_rank": 512, "q_lora_rank": 128, "qk_rope_head_dim": 64,
        "v_head_dim": 128, "qk_nope_head_dim": 128, "n_group": 1, "topk_group": 1, "num_experts_per_tok": 2,
        "norm_topk_prob": True, "hidden_act": "silu", "max_position_embeddings": 262400,
        "initializer_range": 0.02, "rms_norm_eps": 1e-6, "first_k_dense_replace": 1,
        "mlp_layer_types": ["dense", "sparse"], "layer_types": ["full_attention", "full_attention"],
        "attention_dropout": 0.0, "rope_parameters": {"rope_type": "default", "rope_theta": 5000000},
        "rope_interleave": True, "tie_word_embeddings": False, "use_cache": False, "attention_bias": False,
        "mlp_bias": False, "num_nextn_predict_layers": 1, "mtp_loss_factor": 0.3, "use_pad_tokens": True,
        "moe_aux_loss_coeff": 0.0001, "moe_router_enable_expert_bias": True, "moe_router_bias_update_rate": 0.001,
    }
    overrides = [
        {"match": ["model.layers.*.self_attn", "mtp.layers.*.transformer_layer.self_attn"],
             "module_type": f"{model_path}.JTDeepseekV3Attention", "exact_type": True,
             "replace_module": {"_target_": f"{model_path}.JTDeepseekV3MLAAttention"}},
        {"match": "*.self_attn", "region_dispatch": False, "params": {
            "linear_qkv.weight": {"tp": "replicate"}, "q_b_proj.weight": {"tp": "shard(0)"},
            "kv_b_proj.weight": {"tp": "shard(0)"}, "o_proj.weight": {"tp": "shard(1)"}}},
        {"match": "*.self_attn", "when": "cp", "inner_target": "self", "inner_wrapper": {
            "_target_": f"{adapter_path}.distributed.context_parallel.mla_cp_wrapper",
            "strategy": "expanded_ulysses"}},
        {"match": "*.self_attn", "when": "sequence_parallel",
             "in_src": {"hidden_states": {"tp": "shard(1)"}}, "in_dst": {"hidden_states": {"tp": "shard(1)"}}},
        {"match": "*.self_attn.*_layernorm", "when": "sequence_parallel",
             "in_src": {"hidden_states": {"tp": "shard(1)"}}, "in_dst": {"hidden_states": {"tp": "shard(1)"}},
             "out_src": {"tp": "shard(1)"}, "out_dst": {"tp": "replicate"}},
        {"match": "*.self_attn.key_rope_gather", "when": "sequence_parallel",
             "in_src": {"input": {"tp": "shard(1)"}}, "in_dst": {"input": {"tp": "shard(1)"}},
             "out_src": {"tp": "shard(1)"}, "out_dst": {"tp": "replicate"}},
        {"match": "*.mlp.shared_experts", "params": {
            "gate_proj.weight": {"tp": "replicate"}, "up_proj.weight": {"tp": "replicate"},
            "down_proj.weight": {"tp": "replicate"}}, "out_src": {"tp": "replicate"}, "out_dst": {"tp": "replicate"}},
        {"match": "*.mlp.shared_experts", "when": "sequence_parallel",
             "in_src": {"x": {"tp": "shard(1)"}}, "in_dst": {"x": {"tp": "shard(1)"}},
             "out_src": {"tp": "shard(1)"}, "out_dst": {"tp": "shard(1)"}},
        {"match": "mtp.layers.*.eh_proj", "params": {"weight": {"tp": "replicate"}}},
        {"match": "mtp.layers.*.eh_proj", "when": "sequence_parallel",
             "in_src": {"input": {"tp": "shard(1)"}}, "in_dst": {"input": {"tp": "shard(1)"}},
             "out_src": {"tp": "shard(1)"}, "out_dst": {"tp": "shard(1)"}},
    ]
    for pattern in ("model.layers.[123456789]*.mlp", "mtp.layers.*.transformer_layer.mlp"):
        overrides.append({"match": pattern, "when": "ep", "region_dispatch": False, "local_compute_fn": {
            "_target_": f"{adapter_path}.distributed.ep_compute.jt_deepseek_v3_ep_compute_fn"}})
    return resolve_config({
        "model": {"_target_": f"{adapter_path}.jt_builder.build_jt_model", "config": config},
        "plan_overrides": overrides,
        "fsdp_config": {"mix_precision": {"reduce_dtype": "float32"}},
        "optimizer": {
            "_target_": f"{adapter_path}.runtime.jt_optimizer.build_optimizer", "fp32_main_params": True,
            "muon_config": {"lr": 2.4e-5, "weight_decay": 0.1, "momentum": 0.95, "nesterov": True,
                                "ns_steps": 5, "ns_variant": "legacy", "ns_epsilon": 1e-7, "matched_adamw_rms": 0.2},
            "adamw_config": {"adamw_lr": 2.4e-5, "adamw_weight_decay": 0.1,
                                 "adamw_betas": [0.9, 0.95], "adamw_eps": 1e-8},
            "qk_clip_threshold": 100.0,
        },
    })
