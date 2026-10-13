# JT DeepSeek-V3 shared expert overlap

Set the native model configuration option `shared_expert_overlap: true`
(default: `false`). With the existing JT expert-parallel recipe, the final
routed-expert all-to-all is launched asynchronously, the shared MLP runs on the
original unpadded local tokens, and the communication handle is waited before
weighted aggregation and branch addition.

```yaml
model:
  config:
    shared_expert_overlap: true
```

The option applies to trunk and MTP MoE layers. Local execution without EP uses
the same branch semantics and runs serially. There is no module-level shared
output or work-handle cache, so repeated forwards and activation recomputation
do not reuse stale outputs. The shared MLP is still called as a module, preserving
its FSDP hooks and planned TP entry/exit communication.

Only forward combine communication is overlapped. Dispatch and reverse exchanges
remain synchronous. The shared branch keeps the caller's autograd mode; a shared
expert exception still joins the outstanding communication. The padded Gloo path
also supports the asynchronous schedule for numerical acceptance, including
empty ranks. NCCL/HCCL retain the existing ragged exchange format.

## Validation

```bash
python -m pytest -q tests/ut/expert_parallel/test_collectives.py tests/ut/models/jt_deepseek_v3
python tests/torch/expert_parallel/_test_ep_combine_overlap.py
```

The first command checks callback ordering, output and input/router/expert/shared
gradients, padding, repeated steps, inference and exception cleanup. CPU JT tests
mock NPU entry points and reject unmocked NPU kernel calls. The second command
runs two real CPU/Gloo ranks and checks serial/async roundtrip parity, shared
weight gradients and empty splits.

For real accelerator acceptance, use the lightweight launcher with
`HYPER_PARALLEL_EP_TEST_BACKEND=hccl` or `nccl`:

```bash
python -m pytest -q tests/torch/expert_parallel/test_ep_combine_overlap.py
```

Measure the timeline and throughput on the target training configuration before
reporting a performance gain; CPU and mocked kernels cannot establish device
overlap or speedup.

## References

Branch semantics and sigmoid weighting were checked against the official
[Kimi-K2.6 MoE](https://huggingface.co/moonshotai/Kimi-K2.6/blob/main/modeling_deepseek.py)
and [DeepSeek-V3 implementation](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py).
Scheduling follows the independent-compute and stream-join principle used by
[Megatron shared experts](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/shared_experts.py).
JT implements forward combine overlap through its own public EP skeleton.
