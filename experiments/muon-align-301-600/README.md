# Muon 对齐实验包（301→600）

**这是什么**：一段可复现的 Muon 优化器对齐实验。HP（hyper-parallel）在 JT DeepSeek-V3 小模型上
从 MF（MindFormers）第 300 步的权重续训 301→600，逐步记录 `loss / lm_loss / mtp_loss / aux_loss /
grad_norm`，与 MF 同一区间的日志逐点比较，用来判断 HP 的 Muon 实现是否与 MF 逐位一致。

**这份包给你什么**：跑出这个实验所需的**全部东西**（入口脚本、补丁、框架 overlay、配置、PYTHONPATH
上必须存在的目录、环境脚本），以及**用它跑出来的结果**与**自检锚点**。不需要看任何其它分支或实验。

| 项 | 值 |
| --- | --- |
| 基线 commit | `57c35de0f58ae30cba8aa80cf0b62db35e1f51b7` |
| 工作树（容器内） | `/workspace/worktree/align-l2-f`（内容 == 上面的 commit）|
| 入口 | `harness/run.sh` → `harness/entry_mfs.py <config>`（torchrun 单卡）|
| 实验机型 | 910B3 / CANN 9.2.0-beta.2，Python 用 `/workspace/venv` |

## 1. 包内容

| 路径 | 一句话 |
| --- | --- |
| `harness/` | 运行入口与诊断脚本。**跑本实验只需** `run.sh`、`entry_mfs.py`、`pr180.py`；其余（`fix5.py`/`grad_native.py`/`param_native.py`/`dy_tap.py`/`cmp_states.py`/`make_configs.py`/`run_*.sh`）是同一实验目录里的诊断工具，留着备查 |
| `configs/cfg-z.yaml` | **本实验用的配置**（Muon 两个对齐开关打开）|
| `configs/cfg-base.yaml` | 对照配置（两个开关关闭），只跑第 301 步用来自检 |
| `overlay/` | 要覆盖到工作树的框架改动（6 个文件整文件 + 一份 patch）。改动内容：Muon 两个通用开关（`fp32_update`/`ns_per_param`，默认关）、MF 对齐的 NS 适配层（`mf_muon_ns.py`+`mf_muon_parity.py`）、构建器与参数分组的两处小改、包内 `pr180.py` |
| `pypath/` | 实验运行期间 PYTHONPATH 上必须存在的三个非仓库目录（见第 3 步）|
| `env/env.sh` | 容器环境脚本（初始化 CANN、venv、离线标志）|
| `results/run-301-600/` | **本包跑出的主结果**：`metrics-pr180.jsonl`（301→600 共 300 步，每步一行 JSON）、**原始运行日志 `launcher.log`**（逐步打印 `[PR180] step N loss=… grad_norm=…`，并含第 2 节每条补丁的生效行）、`command.sh`（实际命令行）、`SUCCESS.json`、最后一步快照 |
| `results/plots/` | 由现成脚本 `/workspace/tools/compare_hp_mf.py` 生成的图与汇总表：`loss-delta.png`、`loss-smoothed.png`、`early.png`、`summary.md`（命令见第 5 节）|
| `results/anchor-step301.jsonl` | 自检锚点：只跑第 301 步（`cfg-base`）的输出 |
| `results/mf_reference_301_600.csv` | MF 侧同区间参考值（300 点，来自 `info.log`），用于算差值 |

## 2. 实验实际"带上"的补丁（对照审阅用，每条都有运行日志输出）

| 作用 | 实现 | 运行时会打印 |
| --- | --- | --- |
| 关闭 CANN matmul K-shift tiling（#187）| `entry_mfs.py` | `[PR180] CLOSE_MATMUL_K_SHIFT=None popped, later writes suppressed` |
| loss 三项相加顺序改成 MF 的 `(lm+aux)+mtp`（#188）| `entry_mfs.py` | `[PR180] loss association overridden to (lm + aux) + mtp` |
| 反向标量按 token 计数先成（#180）| `pr180.py` | `[PR180] foundation: … backward_scale=1.0 bits=0x3f800000` |
| MTP 丢弃文档尾部 mask（#150）| `entry_mfs.py` | `[PR180] Fix 4: dropping sequence_end_mask on DeepseekV3MTP` |
| MF 单卡实测的 DVM norm clipping | `cfg-z.yaml::model.factors` | `[PR180] attached the measured single-card MF DVM norm clipping` |
| Muon NS 适配层（MF 分块 + 舍入点 + fp32 动量/更新）| `overlay/.../mf_muon_ns.py`、`mf_muon_parity.py` | `mf_muon_parity] MindFormers Newton-Schulz blocks (last-block scale: True) for 11 parameters` |
| 旧的 Fix 5 monkey patch | `fix5.py`（`run.sh` 里固定关闭）| `[PR180] Fix 5 DISABLED: the Muon scale stays in bfloat16` |

## 3. 复现步骤

```bash
# 0) 取 base 工作树
git fetch <你的远端> 57c35de0f58ae30cba8aa80cf0b62db35e1f51b7
git worktree add /workspace/worktree/align-l2-f 57c35de0f58ae30cba8aa80cf0b62db35e1f51b7

# 1) 实验目录（入口 + 配置）
EXP=/workspace/align-work/muon-align-repro
mkdir -p "$EXP" && cp -r <本包>/harness/* <本包>/configs/* "$EXP"/

# 2) 框架 overlay 覆盖到工作树
cp -r <本包>/overlay/components <本包>/overlay/core <本包>/overlay/models \
      <本包>/overlay/pr180.py /workspace/worktree/align-l2-f/hyper_parallel/
# 也可以用 patch（等价）：git -C /workspace/worktree/align-l2-f apply -p1 \
#     <本包>/overlay/framework-overlay-vs-57c35de0.patch

# 3) PYTHONPATH 上的三个目录
mkdir -p /workspace/worktree/align-l2-f/examples/training_demo/align-l2-f
cp -r <本包>/pypath/align-l2-f/*      /workspace/worktree/align-l2-f/examples/training_demo/align-l2-f/
mkdir -p /alignment/tools && cp -r <本包>/pypath/alignment-tools/* /alignment/tools/
mkdir -p /workspace/align-work/MTPChainHP && cp -r <本包>/pypath/MTPChainHP/* /workspace/align-work/MTPChainHP/
cp <本包>/env/env.sh /workspace/env.sh

# 4) 跑主实验（301→600，输出逐行写 $OUTPUT_ROOT/<RUN_NAME>/metrics-pr180.jsonl）
cd "$EXP"
ASCEND_RT_VISIBLE_DEVICES=5 OUTPUT_ROOT=/home/zth/mfs-runs RUN_NAME=repro600 \
  CONFIG="$EXP/cfg-z.yaml" STEPS=600 MASTER_PORT=31883 ./run.sh

# 5) 自检（只跑第 301 步，应与锚点完全一致）
ASCEND_RT_VISIBLE_DEVICES=5 OUTPUT_ROOT=/home/zth/mfs-runs RUN_NAME=anchor \
  CONFIG="$EXP/cfg-base.yaml" STEPS=301 MASTER_PORT=31884 ./run.sh
```

移植到别的机器/路径时，需要改 `harness/run.sh` 里这几处硬编码：`WORKTREE`、PYTHONPATH 中的
`/alignment/tools` 与 `/workspace/align-work/MTPChainHP`、`source /workspace/env.sh`、
`/workspace/venv/bin/torchrun`。

**外部输入**（本包不含，需环境里已有；路径与配置里的字段对应）：

| 输入 | 路径 | 说明 |
| --- | --- | --- |
| 数据集 | `/workspace/datasets/sft_4k_from_256k` | 配置 `dataset.source_dataset.dataset_path`（`datasets.load_from_disk`）|
| 起点权重 | `/workspace/mf-weights/full-state-step300` | 配置 `model.reference_weights`：MF 第 300 步全量状态 ⇒ 步号从 301 起 |
| MF 参考日志 | `/workspace/reference/mf-4k-from-256k/info.log` | `results/mf_reference_301_600.csv` 是它的解析结果 |

## 4. 怎么判断跑对了

1. **第 301 步必须精确等于**：`loss=8.1421117782592773`、`grad_norm=10.284420013427734`
   （与 `results/anchor-step301.jsonl` 和 MF 的 `8.142112 / 10.28442` 一致）。不等就说明环境或补丁没对上。
2. **主结果的第 301/302 步**应为：`8.1421117782592773 / 10.284420013427734`、
   `7.8258428573608398 / 12.831367492675781`；MF 参考对应为 `8.142112 / 10.284420`、
   `7.802559 / 12.751572`。
3. 每行 JSON 的字段：`step, loss, loss_17g, grad_norm, grad_norm_17g, norm_fp64*`（`*_17g` 是
   17 位有效数字打印，`loss`/`grad_norm` 是 float64 值）。
   **注意**：MF 参考只打印 6 位小数，所以 `|Δ| < 5e-7` 属于打印量化噪声，不是差异。

## 5. 本包结果怎么读

`results/run-301-600/`：`metrics-pr180.jsonl` 300 行（301→600），同目录 `launcher.log` 是原始运行日志
（每步一行 `[PR180] step N loss=… grad_norm=…`，并含第 2 节各补丁的生效行）。

**出图/出汇总表**——用环境里现成的脚本，命令照抄（本包 `results/plots/` 就是这条命令的输出）：

```bash
/workspace/venv/bin/python /workspace/tools/compare_hp_mf.py \
  --mf /workspace/reference/mf-4k-from-256k/info.log \
  --run "HP cfg-z=<run-dir>/metrics-pr180.jsonl" \
  --steps 301-600 --out results/plots
```

`results/plots/summary.md` 的结论（301–600 全窗口）：

| 指标 | 值 |
| --- | --- |
| 首步 `Δtotal` | `-0.000000`（0 ULP）|
| `leading steps \|d\| ≤ 1e-6` | 1（只有第 301 步）|
| mean`\|Δtotal\|` / `\|ΔLM\|` / `\|ΔMTP\|` | 0.0948 / 0.0868 / 0.0169 |
| max`\|Δtotal\|` | 0.6717 |
| `\|Δtotal\| > 5e-3` 的步数 | 285 / 300 |
| grad-norm ratio 中位数 | 1.0084 |
| 第 302 步 `\|Δloss\|` | 2.23e-2（MF 7.802559 vs HP 7.825843）|
| 第 600 步 | MF `5.407374/20.364977` vs HP `5.508638/25.752897` |

结论口径（与上游 issue #190 一致）：**300 步曲线不能用来判断实现对不对**——两侧都是混沌放大，
差异在第 302 步就跳到 1e-2 量级；有效判据是单步逐位、前 10–30 步、以及模块级逐位对比
（Muon NS 中间量在 910B 上 7 类 100%、router `ns2–ns5`+`x_ret` 40960/40960 全等）。

## 6. 本包不包含 / 未完成

1. 参数级逐字节对比**没做**（当时三处输出盘满）；本包不声明参数级一致。
2. `lm_head` 权重梯度（0.229 ULP）、attention 反向 GEMM、共享 `dy` 累加边界 —— 都没收尾。
3. 2026-10-14 17:41–18:05（容器时间）那段污染期的数字无效，不要引用。
