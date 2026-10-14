# JT DeepSeek-V3 fixed router

Set the native model configuration option `fix_router: true` (default:
`false`) for deterministic, balanced expert assignments during performance
tuning. This changes the model's routing semantics and is intended for synthetic
benchmarking.

```yaml
model:
  config:
    fix_router: true
```

For `E` routed experts, `K` experts per token and real token index `t`, slot
`s` selects expert `(t * K + s) % E`. Each token receives distinct experts,
and real-token counts per expert differ by at most one, including partial
cycles. Each local sequence partition starts a new cycle; changing partition
sizes can change synthetic assignments. The configuration rejects non-boolean
flags and requires `1 <= K <= E`.

The gate still computes its normal FP32 logits and sigmoid probabilities.
Weights are gathered at the fixed indices, optionally normalized and multiplied
by `routed_scaling_factor`. Router gradients, sequence auxiliary loss and
real-token statistics remain active. Correction bias and RNG state do not
change the assignments. Pad-prefixed tokens retain zero weight and do not
contribute to real-token statistics.

Both standalone and EP execution call the same model-owned route function.
Trunk and MTP configurations inherit the option. With `fix_router: false`,
the existing learned top-k router is unchanged. The option can be used together
with `shared_expert_overlap: true` after both feature branches are applied.

## Validation

```bash
python -m pytest -q tests/ut/models/jt_deepseek_v3/test_fix_router.py
python -m tests.torch.expert_parallel._test_jt_moe_cpu
```

The unit tests cover default routing, configuration roundtrip and invalid
inputs, deterministic unique assignments, exact/remainder/empty cycles,
normalization/scaling, BF16 activations with FP32 routing weights, padding,
router gradients, EP parity and trunk/MTP training.

The second command compares actual JT local and two-rank Gloo EP execution,
including output and input/router/expert/shared gradients, learned/fixed routes
and padding on/off. Expert math uses the Torch reference and the shared CPU UT
NPU-import fixture; this does not validate NPU grouped kernels or device
performance. When the overlap feature is present, the EP candidate also enables
it, providing joint-feature acceptance.

## References

The performance-only fixed-routing contract follows the official
[MindSpeed-LLM argument guide](https://github.com/Ascend/MindSpeed-LLM/blob/master/docs/zh/pytorch/features/fsdp2/arguments.md).
The deterministic cycle is JT's implementation choice, rather than the random
logits used by
[Megatron's benchmark balancing option](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py).
The retained sigmoid weights, normalization and scaling were checked against
[Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6/blob/main/modeling_deepseek.py)
and [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py).
