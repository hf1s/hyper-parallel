# JT #217 exact archive code — step300 continuation

唯一 canonical reproduction bundle. 不使用 rN/vN 目录，不依赖散落的旧实验目录。

## Goal

用 #217 更新后的 exact archive source，在我们的 step300 model/optimizer state 上续训 step301–600，并与 MF reference 逐步比较。

这不是 #217 原始 MF200→HP201–400 验收；step200 checkpoint 不在公开 evidence 中。

## Frozen source/config

- canonical source head: `fe16c866cba6f40e5fea432e8147e7446fbf1e7f`;
- refactor source: branch `repro/jt217-auditable-archive-20261018`;
- source is the parent repository on the Git branch;
- in the Huawei runtime bundle, the same source is at the sibling `source/` directory;
- config: `config/hp-archive200-r2.yaml`;
- source profile: `mf-r180-cann85-910b3-v1`;
- no P12 experimental factors;
- no audit/checksum callbacks.
The YAML selects the archive entry point; it is not the whole implementation. The full source composition is documented in [`SOURCE_COMPOSITION.md`](SOURCE_COMPOSITION.md).

## Refactor rerun on Huawei-129

| 核查项 | 原 canonical archive | auditable archive branch | 结论 |
|---|---|---|---|
| 活动实现位置 | `models/jt_deepseek_v3/mf_archive/*` | JT model、loss、optimizer、trainer 标准模块 | 实现已移出 `mf_archive`；旧目录只保留说明和兼容标记 |
| 配置 target | `mf_archive.model/layers/optimizer` | `archived_model`、`archived_layers`、`adapter.runtime.archived_optimizer` | 活动配置不再导入旧实现路径 |
| 模型、数据、恢复状态 | 相同 `paths.env` | 相同 `paths.env` | 使用同一 step300 锚点和数据集 |
| HP continuation | step301–600 | step301–600 | 两次均成功完成 300 steps |
| metrics 对齐 | 300 行 | 300 行 | step、component keys 全部一致 |
| 数值逐值比较 | 旧 HP metrics | 新 HP metrics | 1500/1500 数值完全相等；最大绝对/相对差均为 0 |
| MF 重跑 | 未运行 | 未运行 | 本次只验证 HP 侧 |

新 bundle 输出：`/workspace/zth-jt-precision-scripts/repro/jt217-auditable-archive-20261018-overlay/output/hp/`。

## Inputs

`inputs/paths.env` is the only input contract. It names the exact large inputs:

```text
JT217_MODEL_STATE=/workspace/mf-weights/full-state-step300
JT217_NATIVE_STATE=/workspace/mf-weights/native-step300/global_step_300
JT217_DATASET=/workspace/datasets/sft_data_4k_from_256k
```

The original inputs remain untouched. The MF dataset must be the same Arrow dataset used by the MF reference; do not substitute `sft_4k_from_256k`. `reference/mf-reference.info.log` is the fixed MF comparison log.

## Run

Inside `hp-910b-arrow`:

```bash
bash run-hp.sh
```

Output:

```text
output/hp/metrics.jsonl
output/hp/SUCCESS.json
```

## Plot/compare

```bash
bash make-plot.sh
```

Outputs:

```text
output/compare/summary.md
output/compare/loss-delta.png
output/compare/loss-smoothed.png
output/compare/early.png
```

## Expected contract

- HP CANN9.2.0-beta.2
- single NPU / TP=EP=CP=PP=1
- `CLOSE_MATMUL_K_SHIFT=0`
- step301 total error is approximately `2.13e-6`
- the comparison uses the MF reference log and reports every step 301–600
- no hashes/checksums are required or calculated
