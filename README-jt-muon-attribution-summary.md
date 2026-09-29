# JT Canonical V4.1：Hyper / Reference 对齐归因摘要

## 1. 当前 `refactor/jt-canonical-v41-final` 对齐问题

| field | value |
|---|---|
| branch | `refactor/jt-canonical-v41-final` |
| commit | `86246c8b279b` |
| validation | [Issue #33](https://github.com/hf1s/hyper-parallel/issues/33) |
| runtime | 10 steps completed, exit 0 |

Step 10（Hyper - Reference）：

```text
LM loss   -2.347e-3
MTP loss  -2.169e-4
total     -2.564e-3
```

Step 1 raw loss 基本一致；偏差在 optimizer update 后出现。

## 2. 已排除的其他偏差来源

| Issue | branch / commit | 结论 |
|---:|---|---|
| [#35](https://github.com/hf1s/hyper-parallel/issues/35) | `experiment/jt-canonical-forward-alignment` / `c65ce9e7` | MLP gate/up forward packing 未改善 LM，反而变差 |
| [#37](https://github.com/hf1s/hyper-parallel/issues/37) | `experiment/jt-attention-latent-unfused` / `5be580ba` | q/kv child GEMM 不是主要来源 |
| [#38](https://github.com/hf1s/hyper-parallel/issues/38) | `experiment/jt-canonical-muon-expert-shape-corrected` / `f5deaeaa` | expert orientation 无可观测影响 |
| [#41](https://github.com/hf1s/hyper-parallel/issues/41) | `experiment/jt-attention-latent-two-gemm` / `809c9a3d` | legacy-style 两次 latent GEMM 未改善 LM |
| [#45](https://github.com/hf1s/hyper-parallel/issues/45) | `experiment/jt-system-loss-trace` / `98ae222f` | raw model LM 已出现偏差，Trainer loss reduction 不是主要来源 |
| [#51](https://github.com/hf1s/hyper-parallel/issues/51) | `experiment/jt-muon-reference-scale` / `0865b2c7` | local logical-dimension scale 使 LM delta 翻为正，拒绝 |
| [#53](https://github.com/hf1s/hyper-parallel/issues/53) | `experiment/jt-muon-ns-unbatched` / `eca1fee9` | unbatched NS 与 baseline 逐字节一致 |
| [#54](https://github.com/hf1s/hyper-parallel/issues/54) | `experiment/jt-muon-ns-reference-arithmetic` / `6d0a1597` | Reference-style 函数式 NS arithmetic 无变化 |
| [#55](https://github.com/hf1s/hyper-parallel/issues/55) | `experiment/jt-muon-ns-reference-norm` / `ec27f8fc` | Reference 2D norm 写法无变化 |
| [#57](https://github.com/hf1s/hyper-parallel/issues/57) | `experiment/jt-muon-parameter-apply-trace` / `b94a44dc` | chunk / weight decay / lr / parameter write-back 无变化 |
| [#58](https://github.com/hf1s/hyper-parallel/issues/58) | `experiment/jt-muon-final-attribution` / `8a21566a` | 完整阶段 trace；差异仍停在 Muon NS/update |

另外：

```text
QK clipping: qk_clip_delta = 0
step 1 raw loss: 与 Reference 约 1e-6 内
gradient / Muon input: cosine 约 0.999995–0.999999
```

结论：forward、backward gradient、loss aggregation、QK clip、parameter apply 不是主要偏差来源。

## 3. Muon 优化器的语义差异

Reference / 旧 JT 实现对逻辑矩阵进行 split：

```text
linear_qkv:
  q / kv / rope 分开做 NS

q_b_proj:
  每个 head 的 qk_nope / qk_rope 分开做 NS

kv_b_proj:
  每个 head 的 qk_nope / value 分开做 NS
```

当前 canonical/no-split 实现：

```text
q_a_proj                  独立参数，但 q 已独立
kv_a_proj_with_mqa       kv latent + rope 仍在一个矩阵
q_b_proj                 整体矩阵
kv_b_proj                整体矩阵
```

证据：

- [#39](https://github.com/hf1s/hyper-parallel/issues/39) canonical 首步 Muon trace；
- [#40](https://github.com/hf1s/hyper-parallel/issues/40) 旧 trajectory Muon trace；
- [#46](https://github.com/hf1s/hyper-parallel/issues/46) canonical gradient/update tensor；
- [#47](https://github.com/hf1s/hyper-parallel/issues/47) old gradient/update tensor。

对比显示：

```text
gradient 基本一致
Muon ns_output / update direction 不一致
```

所以 logical split 会减少一部分 LM drift，但它不是全部差异的来源。

## 4. 消除部分语义差异后的结果

| branch | commit | validation | step 10 LM | step 10 MTP | step 10 total |
|---|---|---|---:|---:|---:|
| canonical/no-split | `86246c8b` | [#33](https://github.com/hf1s/hyper-parallel/issues/33) | `-2.347e-3` | `-2.169e-4` | `-2.564e-3` |
| canonical + logical split | `7891e469` | [#34](https://github.com/hf1s/hyper-parallel/issues/34) | `-1.331e-3` | `-1.139e-4` | `-1.445e-3` |
| old logical-split baseline | `9ea4a9fa` | [#25](https://github.com/hf1s/hyper-parallel/issues/25) | `-1.225e-3` | `-8.33e-5` | `-1.309e-3` |

结果：

```text
恢复 logical split 消除了约 1e-3 的额外 LM drift。
但旧 logical-split Hyper 与 Reference 仍有约 1.2e-3 LM 偏差。
```

因此当前最终归因是：

```text
除 Muon core/update 实现外，其他已测试路径都基本对齐。
剩余问题不是 loss 公式或当前 model forward 的单个融合算子，
而是 Hyper Muon 与 Reference Muon 在 NS/update 执行语义上的系统差异。
```

Production baseline 保持：

```text
refactor/jt-canonical-v41-final @ 86246c8b
```

实验分支均未合并。
