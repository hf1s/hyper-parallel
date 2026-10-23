# FP32 expert weight gradients

The ordinary grouped SwiGLU kernel rounds each local expert weight gradient to
the BF16/FP16 weight dtype before FSDP reduces it. Different token partitions can
therefore produce different gradients even with FP32 reduction enabled. This
effect can be amplified by Muon's orthogonalization.

For the JT example, explicitly enable:

```yaml
model:
  config:
    expert_fp32_weight_gradients: true
optimizer:
  fp32_main_params: true
fsdp_config:
  mix_precision:
    param_dtype: bfloat16
    reduce_dtype: float32
```

The setting defaults to `false`. The forward and input-gradient kernels remain
unchanged. The grouped weight-gradient kernel accumulates BF16/FP16 products
directly into FP32 storage owned by the unsharded parameter. FSDP transfers this
buffer into its existing accumulation and reduction path, including delayed
synchronization and ordinary autograd contributions to the same parameter.

This requires an Ascend runtime exposing `torch_npu.npu_grouped_matmul_add_` and
FSDP's FP32 main-gradient policy. An incompatible owner or precision policy fails
instead of silently rounding the requested gradient back to BF16. Only local,
contiguous expert parameters are supported by this fused kernel path; native
DTensor inputs must first pass through their declared local expert region.

The option changes numerical rounding and increases unsharded gradient storage.
Use a matching FSDP baseline when comparing parallel strategies. It does not
promise bitwise equality across different reduction orders, change Muon's own
precision, or resolve upstream TP forward/input-gradient differences. Measure
the complete model's loss, peak memory and step time for the intended topology.
