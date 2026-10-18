# JT #217 exact archive code — step300 continuation

唯一 canonical reproduction bundle. 不使用 rN/vN 目录，不依赖散落的旧实验目录。

## Goal

用 #217 更新后的 exact archive source，在我们的 step300 model/optimizer state 上续训 step301–600，并与 MF reference 逐步比较。

这不是 #217 原始 MF200→HP201–400 验收；step200 checkpoint 不在公开 evidence 中。

## Frozen source/config

- source head: `fe16c866cba6f40e5fea432e8147e7446fbf1e7f`
- source is the parent repository on the Git branch;
- in the Huawei runtime bundle, the same source is at the sibling `source/` directory;
- config: `config/hp-archive200-r2.yaml`;
- source profile: `mf-r180-cann85-910b3-v1`;
- no P12 experimental factors;
- no audit/checksum callbacks.

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
