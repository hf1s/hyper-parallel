# Muon Newton–Schulz precision

Muon defaults to BF16 Newton–Schulz inputs, arithmetic and relayed updates.
FP32 main parameters and FP32 FSDP gradient reduction do not change that default.
Rounding before and after orthogonalization can amplify small gradient differences
between otherwise equivalent parallel layouts.

Use an explicit setting to preserve FP32 throughout this path:

```yaml
optimizer:
  muon_config:
    ns_dtype: float32
```

The public `Muon` optimizer and individual parameter groups accept `ns_dtype`
as either the string `"bfloat16"` or `"float32"`. The choice covers the output
of the momentum stage, the built-in NS arithmetic, and packed update relay
buffers, including empty local shards and ranks without gradients. Custom momentum
callback outputs are copied into this dtype so NS cannot overwrite their live
gradients or momentum buffers; custom NS callbacks
remain responsible for their own arithmetic. Momentum-state and parameter dtypes,
NS coefficients, iteration count, learning rate and backend matmul precision
settings are unchanged.

The setting is saved in optimizer parameter groups. Loading a checkpoint without
this field uses the historical BF16 behavior, even if the new optimizer was
constructed with an FP32 default. Resume a comparison with the same saved recipe.
The trainer uses an optimizer load skeleton with these legacy defaults before
DCP reads persisted fields, including flattened chained and mixed-precision
optimizers. A weights-only restore keeps the new optimizer configuration.

FP32 doubles the element storage of NS inputs, intermediates and relay buffers,
and may be substantially slower on the target accelerator. The existing soft
batch limit now accounts for element size. It still preserves compute-owner
filling and never splits an individual matrix, so it is not an absolute memory
limit. Measure end-to-end memory and throughput before selecting the setting.

This option does not correct upstream forward/input-gradient rounding, guarantee
bitwise equality across parallel layouts, or replace a full training comparison
against a matching FSDP baseline.
