# JT Canonical V4.1 Refactor：Hyper 与 Reference 对齐记录

> 目标：在保持 canonical 参数树、YAML-selected execution operator 和 no-JT-specific-Muon-split 设计的前提下，记录 Hyper 与 MindFormers Reference 的 10-step 对齐结果，并区分模型执行差异、Muon 差异和系统级差异。

## 1. 当前基线与结论

### Canonical production baseline

| field | value |
|---|---|
| branch | `refactor/jt-canonical-v41-final` |
| commit | `86246c8b279b` |
| design | canonical parameter tree + YAML fused operators + no JT-specific Muon logical split |
| validation | [Issue #33](https://github.com/hf1s/hyper-parallel/issues/33) |
| status | runtime stable; 10 steps completed |

Step 10 baseline delta（Hyper - Reference）：

```text
LM loss   -2.3472e-3
MTP loss  -2.1692e-4
total     -2.5643e-3
```

### Current attribution

已确认：

1. step 1 raw model loss 与 Reference 基本一致；偏差在第一次 optimizer update 后出现。
2. `JT_RAW_LOSS` 已经出现 LM 偏差，`mean_global_loss` 不是主要来源。
3. QK clipping 没有触发：`qk_clip_delta = 0`。
4. Hyper 与 Reference 的首次 gradient / Muon input 基本一致。
5. Hyper 与 Reference 的 NS output / scaled update 不一致；差异在 Muon update path 中出现。
6. canonical/no-split 相比旧 logical-split 额外改变了 Muon update direction，但旧 logical-split 本身仍有约 `1.2e-3` 级 LM 偏差。
7. MLP packing、latent projection unfused、expert orientation 均未解释主要剩余偏差。

因此当前问题分成两层：

```text
A. Hyper canonical/no-split 与旧 logical-split 的 Muon update 差异
B. Hyper Muon core 与 MindFormers Reference Muon update 实现的系统差异
```

当前尚未把任何实验 patch 合并到 production baseline。

## 2. 主要验证结果

| Issue | branch | commit | 实验 | step 10 LM delta | step 10 MTP delta | step 10 total delta |
|---:|---|---|---|---:|---:|---:|
| [#21](https://github.com/hf1s/hyper-parallel/issues/21) | `fix/jt-muon-logical-split` | `2dbac0a85f73` | initial Muon logical split | `-6.58e-4` | `-4.67e-5` | `-7.06e-4` |
| [#22](https://github.com/hf1s/hyper-parallel/issues/22) | `fix/jt-muon-logical-split` | `9fced3bcb7d3` | KV/RoPE split refinement | `-1.376e-3` | `-6.17e-5` | `-1.439e-3` |
| [#23](https://github.com/hf1s/hyper-parallel/issues/23) | `fix/jt-muon-logical-split` | `88ed184b323b` | loss reduction cleanup | `-1.375e-3` | `-6.18e-5` | `-1.438e-3` |
| [#25](https://github.com/hf1s/hyper-parallel/issues/25) | `fix/jt-muon-logical-split` | `9ea4a9fa6692` | hidden dtype boundary | `-1.225e-3` | `-8.33e-5` | `-1.309e-3` |
| [#26](https://github.com/hf1s/hyper-parallel/issues/26) | `codex/deepseek-v32-sft-trainer-without-muon-splits` | `f71024da851e` | remove Muon logical split | `-2.583e-3` | `-2.659e-4` | `-2.849e-3` |
| [#32](https://github.com/hf1s/hyper-parallel/issues/32) | `refactor/jt-canonical-v41` | `86246c8b279b` | canonical/no-split | `-2.347e-3` | `-2.169e-4` | `-2.564e-3` |
| [#33](https://github.com/hf1s/hyper-parallel/issues/33) | `refactor/jt-canonical-v41` | `86246c8b279b` | final baseline repeat | `-2.347e-3` | `-2.169e-4` | `-2.564e-3` |
| [#34](https://github.com/hf1s/hyper-parallel/issues/34) | `experiment/jt-canonical-muon-logical-split` | `7891e46915c3` | canonical + logical split | `-1.331e-3` | `-1.139e-4` | `-1.445e-3` |
| [#35](https://github.com/hf1s/hyper-parallel/issues/35) | `experiment/jt-canonical-forward-alignment` | `c65ce9e71484` | forward-only MLP packing | `-1.581e-3` | `-8.30e-5` | `-1.665e-3` |
| [#36](https://github.com/hf1s/hyper-parallel/issues/36) | `experiment/jt-canonical-muon-expert-shape` | `46bf6ddac5a7` | expert orientation, first attempt | same as #34 | same as #34 | same as #34 |
| [#37](https://github.com/hf1s/hyper-parallel/issues/37) | `experiment/jt-attention-latent-unfused` | `5be580bab94e` | separate q/kv child GEMMs | `-1.517e-3` | `-9.76e-5` | `-1.614e-3` |
| [#38](https://github.com/hf1s/hyper-parallel/issues/38) | `experiment/jt-canonical-muon-expert-shape-corrected` | `f5deaeaa0572` | corrected expert orientation | same as #34 | same as #34 | same as #34 |
| [#41](https://github.com/hf1s/hyper-parallel/issues/41) | `experiment/jt-attention-latent-two-gemm` | `809c9a3d4471` | legacy-style split latent GEMMs | `-1.517e-3` | `-9.76e-5` | `-1.614e-3` |
| [#50](https://github.com/hf1s/hyper-parallel/issues/50) | `experiment/jt-muon-ns-fp32-compare` | `5f9a754f` | FP32 internal NS experiment | `-3.015e-3` | `-2.570e-4` | `-3.272e-3` |

### Interpretation

- Restoring logical Muon split is the only tested change that removes roughly `1e-3` of LM drift.
- MLP packing does not improve LM and makes total loss worse.
- Latent projection fusion changes the result only slightly; unfused variants are worse.
- Expert orientation produced no observable change in the validation run.
- FP32 internal NS did not improve the trajectory; it made the final loss worse.

## 3. Engineering/runtime fixes before the stable baseline

| Issue | commit | fix |
|---:|---|---|
| [#29](https://github.com/hf1s/hyper-parallel/issues/29) | `1175b80d` | import `torch.nn as nn` in grouped expert replacement |
| [#30](https://github.com/hf1s/hyper-parallel/issues/30) | `1bad895f` | restore fused attention `forward` call contract and `**kwargs` |
| [#31](https://github.com/hf1s/hyper-parallel/issues/31) | `cd632147` | derive TP-local attention head count from projection output |
| [#32](https://github.com/hf1s/hyper-parallel/issues/32) | `86246c8b` | use gathered query sequence length before `o_proj` reshape |

The final stable baseline is `86246c8b`.

## 4. Diagnostic branches and their purpose

| branch | commit | base / purpose |
|---|---|---|
| `experiment/jt-canonical-muon-logical-split` | `7891e469` | canonical branch plus logical Muon split; best alignment experiment |
| `experiment/jt-canonical-forward-alignment` | `c65ce9e` | MLP gate/up forward packing; rejected by #35 |
| `experiment/jt-canonical-muon-expert-shape` | `46bf6dda` | initial expert reshape experiment; name match was ineffective |
| `experiment/jt-canonical-muon-expert-shape-corrected` | `f5deaeaa` | corrected expert parameter matching; no effect in #38 |
| `experiment/jt-attention-latent-unfused` | `5be580ba` | q/kv child linear A/B; not the main source |
| `experiment/jt-attention-latent-two-gemm` | `809c9a3d` | exact legacy-style split latent GEMMs; not the main source |
| `experiment/jt-muon-first-step-trace` | `89dfd2f` | first-step Hyper Muon norm trace |
| `experiment/jt-old-muon-first-step-trace` | `19184249` | same trace on old `9ea4a9` trajectory |
| `experiment/jt-system-loss-trace` | `66ad0a29` | raw model loss vs Trainer reduction trace |
| `experiment/jt-reference-vector-diagnostics` | `29f1bbfa` | Reference instrumentation bundle imported into Hyper branch |
| `experiment/jt-canonical-vector-trace` | `30533061` | canonical first-step gradient/update tensor dump, base `86246c8b` |
| `experiment/jt-old-vector-trace` | `efe45104` | old first-step gradient/update tensor dump, base `9ea4a9` |
| `experiment/jt-hyper-muon-ns-trace` | `45ae7c67` | initial Hyper NS trace; global-shape dump, superseded |
| `experiment/jt-hyper-muon-ns-local-trace` | `af547a85` | local-slice Hyper NS trace |
| `experiment/jt-muon-ns-fp32-compare` | `5f9a754f` | FP32 internal NS A/B; rejected by #50 |
| `experiment/jt-muon-reference-scale` | `0865b2c7` | Reference local logical-dimension scale A/B; pending validation |

## 5. Diagnostic run index

| Issue | branch | commit | result |
|---:|---|---|---|
| [#39](https://github.com/hf1s/hyper-parallel/issues/39) | `experiment/jt-muon-first-step-trace` | `89dfd2f2` | current Hyper first-step norm trace; QK clip zero |
| [#40](https://github.com/hf1s/hyper-parallel/issues/40) | `experiment/jt-old-muon-first-step-trace` | `19184249` | old trajectory trace for comparison |
| [#41](https://github.com/hf1s/hyper-parallel/issues/41) | `experiment/jt-attention-latent-two-gemm` | `809c9a3d` | exact split latent GEMMs; LM step 10 `-1.517e-3` |
| [#42](https://github.com/hf1s/hyper-parallel/issues/42) | `experiment/jt-system-loss-trace` | `98ae222f` | trace import failure: missing `ABC` |
| [#43](https://github.com/hf1s/hyper-parallel/issues/43) | `experiment/jt-system-loss-trace` | `98ae222f` | trace model failure: missing `mtp_loss` assignment |
| [#45](https://github.com/hf1s/hyper-parallel/issues/45) | `experiment/jt-system-loss-trace` | `98ae222f` + dirty worktree | raw model loss and Trainer-reduced loss trace; run passed |
| [#46](https://github.com/hf1s/hyper-parallel/issues/46) | `experiment/jt-canonical-vector-trace` | `30533061` | canonical gradient/update tensor dump |
| [#47](https://github.com/hf1s/hyper-parallel/issues/47) | `experiment/jt-old-vector-trace` | `efe45104` | old trajectory gradient/update tensor dump |
| [#48](https://github.com/hf1s/hyper-parallel/issues/48) | `experiment/jt-hyper-muon-ns-trace` | `45ae7c67` | initial global-shape NS dump; superseded |
| [#49](https://github.com/hf1s/hyper-parallel/issues/49) | `experiment/jt-hyper-muon-ns-local-trace` | `af547a85` | TP-local NS dump; valid shape comparison |
| [#50](https://github.com/hf1s/hyper-parallel/issues/50) | `experiment/jt-muon-ns-fp32-compare` | `5f9a754f` | FP32 NS internal A/B; final LM delta `-3.015e-3`, rejected |

## 6. Reference-side diagnostics

Reference checkout:

```text
/Users/touk0/project/Distributed-and-Parallel-Computing/reference-code-10steps
```

Local artifacts:

```text
/Users/touk0/project/Distributed-and-Parallel-Computing/vector-trace-01
```

Reference diagnostic branch imported into Hyper:

```text
experiment/jt-reference-vector-diagnostics
commit 29f1bbfa
```

Reference vector output:

```text
vectors/rank_0 ... rank_7/
```

Dump kinds:

```text
muon_input_fp32
momentum_fp32
ns_output
scaled_update
```

The Reference trace and Hyper vector trace use different parameter layouts. The comparison mapping is:

```text
Reference linear_q_down_proj      ↔ Hyper q_a_proj
Reference linear_kv_down_proj     ↔ Hyper kv_a_proj_with_mqa
Reference linear_q_up_proj        ↔ Hyper q_b_proj
Reference linear_kv_up_proj       ↔ Hyper kv_b_proj
Reference linear_proj             ↔ Hyper o_proj
```

The Hyper vector trace must be compared after TP-local slicing. The initial global-shape Hyper NS trace was invalid for this comparison and was superseded by `af547a85`.

## 7. First-step attribution

The clean canonical baseline and old trajectory traces show:

```text
gradient / Muon input:
  cosine >= 0.999995 in the compared attention slices

QK clipping:
  zero; threshold 100 was not reached
```

The remaining difference appears after entering the Muon Newton–Schulz/update path:

```text
Reference ns_output != Hyper ns_output
Reference scaled_update != Hyper update
```

The current evidence separates two effects:

1. **Common Hyper-vs-Reference Muon core difference**
   - present even when comparing against the old logical-split trajectory;
   - likely involves distributed NS input assembly, local/global logical dimensions, NS numerical path, or update scaling.

2. **Canonical/no-split-specific difference**
   - strongest in the `kv/rope`, `q_b`, and `kv_b` logical slices;
   - explains why `86246c8b` is worse than `7891e469`.

The raw model loss and Trainer loss trace from [#45](https://github.com/hf1s/hyper-parallel/issues/45) showed that the LM delta already exists in raw model loss; Trainer reduction is not the primary source.


### Reference vector comparison

After TP-local slicing and first-step mapping:

```text
Reference muon_input vs Hyper gradient (with Nesterov factor):
  cosine ≈ 0.999995–0.999999

Reference ns_output vs Hyper BF16 ns_output:
  q_a   ≈ 0.96–0.97
  kv_a  ≈ 0.46–0.69
  q_b   ≈ 0.78–0.83
  kv_b  ≈ 0.47–0.64
  o_proj≈ 0.44–0.83
```

This places the first clear divergence after the gradient and inside the Hyper
Muon NS/update path. The FP32-internal-NS experiment [#50](https://github.com/hf1s/hyper-parallel/issues/50)
did not improve the 10-step trajectory; the next candidate is local/global
logical-shape scaling and shard/update ordering.
## 8. Current pending experiment

```text
branch: experiment/jt-muon-reference-scale
commit: 0865b2c7
base: 86246c8b + diagnostic local-slice trace
```

It enables the Reference-style local logical-dimension scale with:

```bash
JT_MUON_REFERENCE_SCALE=1
```

This is an attribution experiment only. It has not been validated yet and is not part of the production baseline.

## 9. Production decision

Accepted production baseline:

```text
refactor/jt-canonical-v41-final @ 86246c8b
```

Properties:

- canonical trainable parameter tree;
- YAML-selected fused attention and expert operators;
- no JT-specific Muon logical split;
- no forced FP32 path;
- no hard-coded computation-order workaround;
- runtime validation passes 10 steps.

The experimental branches above are diagnostic/attribution branches and should not be merged into the production branch without a separate acceptance decision.
