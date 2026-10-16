"""Map JT parameter layouts to the native MindFormers norm schedule."""


def mapping_for(name: str) -> tuple[list[str], str]:
    """Map HP model coordinates to native MF coordinates."""
    exact = {
        "model.embed_tokens.weight": "model.embedding.word_embeddings.weight",
        "model.norm.weight": "model.decoder.final_layernorm.weight",
        "lm_head.weight": "model.output_layer.weight",
    }
    if name in exact:
        return [exact[name]], "identity"
    if name.startswith("mtp.") and ".transformer_layer." not in name:
        return ["model." + name], "identity"
    prefix = name.split(".self_attn.")[0] if ".self_attn." in name else name.split(".mlp.")[0]
    if ".self_attn." not in name and ".mlp." not in name:
        prefix = name.rsplit(".", 2)[0]
    native = prefix.replace("model.layers.", "model.decoder.layers.") if prefix.startswith("model.") else "model." + prefix
    suffix = name[len(prefix) + 1:]
    replacements = {
        "input_layernorm.weight": "input_layernorm.weight",
        "post_attention_layernorm.weight": "pre_mlp_layernorm.weight",
        "self_attn.q_a_layernorm.weight": "self_attention.q_layernorm.weight",
        "self_attn.kv_a_layernorm.weight": "self_attention.kv_layernorm.weight",
        "self_attn.q_b_proj.weight": "self_attention.linear_q_up_proj.weight",
        "self_attn.kv_b_proj.weight": "self_attention.linear_kv_up_proj.weight",
        "self_attn.o_proj.weight": "self_attention.linear_proj.weight",
        "mlp.gate.weight": "mlp.router.weight",
        "mlp.gate.e_score_correction_bias": "mlp.router.expert_bias",
    }
    if suffix in replacements:
        return [native + "." + replacements[suffix]], "identity"
    if suffix == "self_attn.linear_qkv.weight":
        return [
            native + ".self_attention.linear_q_down_proj.weight",
            native + ".self_attention.linear_kv_down_proj.weight",
        ], "concat"
    if suffix in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
        index = 1 if suffix.endswith("gate_up_proj") else 2
        return [native + f".mlp.experts.weight{index}"], "expert_transpose"
    for projection, transform in [("gate_proj", "even"), ("up_proj", "odd"), ("down_proj", "identity")]:
        if suffix.endswith(projection + ".weight"):
            return [native + "." + suffix.replace(
                projection, "linear_fc2" if projection == "down_proj" else "linear_fc1"
            )], transform
    raise ValueError(f"Unmapped HP parameter: {name}")
