# 固定 JT 配方的 MF → HP 续训状态转换

本工具处理旧 MF 两层 JT 实验归档：第 0 层为 dense、第 1 层为 MoE、一个 MTP 层。
它不是通用 MF 转换器，也不改变训练算子、规约顺序或优化器更新公式。

## 处理流程

1. `convert_mf_ckpt_to_hp.py` 将模型与优化器动量执行相同的坐标转换，输出 `model.npz`、`optimizer.npz`、`state.json` 和 `CONVERSION.json`。
2. `materialize_hp_native_checkpoint.py` 构造 HP 配方，验证目标优化器参数的全部必需状态，从无损 FP32 源恢复主权重与动量，重建受支持的步数、常数 LR 计数及顺序数据游标。
3. 使用 HP checkpoint callback 写入 `<output>/global_step_<N>/` 原生 DCP；检查 `.metadata` 已生成，拒绝覆盖已有输出。

## 命令与配置

```bash
python scripts/checkpoint_conversion/convert_mf_ckpt_to_hp.py \
  --ckpt /path/to/mf-step200.safetensors \
  --mlp-layout swiglu \
  --out-dir /path/to/hp-offline-state

python scripts/checkpoint_conversion/materialize_hp_native_checkpoint.py \
  /path/to/train-4k.yaml \
  --state-dir /path/to/hp-offline-state \
  --output /path/to/hp-checkpoints \
  --step 200
```

`--mlp-layout swiglu` 用于 dense/shared-expert 已替换成公共 `SwiGLUMLP` 的配置：MF 交错 gate/up 行转换为 HP `linear_fc1` 的先 gate 后 up 布局，动量执行相同转换。默认 `separate` 对应独立的 `gate_proj/up_proj/down_proj`。配置与布局必须匹配，不得用随机初始化补齐缺失参数。

### 固定 MF 原生 train-cell safetensors

默认 `--source-format auto` 根据 safetensors 的 `__metadata__.format=ms` 选择原生字段校验；无该标记的旧 family-prefix 文件保持原入口。可显式指定 `--source-format mf-native` 或 `family-prefix`。本适配不读取 MindSpore `.ckpt`，也不重组移除冗余后的多分片文件。

固定 `CheckpointMonitor` 保存 `MFTrainOneStepCell` 时，文件还包括优化器/包装器计数、日志和运行状态。读取器只处理已知生产者字段：

- `global_step` 必须与 `step_num` 同为相等的非负整数；`scale_sense` 必须与单位 `loss_scale` 一致。不用其中一个值覆盖另一个不一致值。
- 固定配方各 dropout 的 `seed/offset` 仅在整数零值时接受；不恢复 RNG 状态。非零时拒绝转换。
- 两处已知 router 的 `fi_accu/expert_load` 必须形状匹配专家数、为已消费的零统计。非零时拒绝，避免漏掉尚待执行的 bias 更新。
- `lm_loss/mtp_loss/aux_loss` 仅作为有限 FP32 单元素日志累加器处理，允许非零旧日志。固定 wrapper 的训练 loss/梯度由当步网络返回值计算；日志回调读出后清零，这三个累加器不参与模型或优化器更新，本适配不恢复历史日志。

这些字段的名称、shape、dtype、处理理由和标量值记录到 `CONVERSION.json` 的 `source_input`，模型与动量数组不被修改。未知键继续报错，不按张量大小或泛化的名字后缀静默丢弃。该 profile 仍限于本文的确定性固定配方；header 标记本身不证明模型完整、保存时点正确或数据内容相同，后续物化仍须检查全部状态与真实下一批。

`model.reference_weights` 必须指向所选离线目录，其余必要条件：

- 单设备，TP/EP/CP/PP 均为 1，global/micro batch 均为 1。
- 原生 `MultiLRScheduler`，`lr_decay_style: constant`，无 warmup。
- 原生 `FixedBatchDataLoader`、`num_workers: 0`，顺序 `single` sampler，无样本重映射；数据是与 MF 一致的**完整原始序列**。源已完成 200 次更新时下一条逻辑索引为 200，不能再传入裁掉前 200 条的数据集。
- LR、betas、epsilon、weight decay、模型结构、tokenization/packing 与源实验一致。这些信息不能从 m/v 推导，需调用者核实配方。
- 确定性续训配方；不恢复源 RNG、随机 sampler、动态 loss scaler 或非恒定 LR 历史。

物化器启用 optimizer/train-state 保存，将训练长度设置为 `max(1, 2*N)`，随后只保存状态，不执行训练。游标限于首个数据遍历中的逻辑位置；不从旧 MF 的 `epoch_num` 推导 HP epoch，源值只作为 `source_epoch_num` 保留。

### 已有 manifest 命名的 AdamW 归档

另一种已保存的实验格式用 `<HP参数名>::exp_avg` / `::exp_avg_sq` 保存动量，由 `CONVERSION.json` 的 `optimizer_manifest` 明确映射至 HP 参数名。可直接读取这些 NPZ，不必为了改键名重复生成大文件：

```bash
python scripts/checkpoint_conversion/materialize_hp_native_checkpoint.py \
  /path/to/train-4k.yaml \
  --state-dir /path/to/existing-conversion \
  --optimizer-key-format manifest \
  --resume-state /path/to/verified-resume-state.json \
  --output /path/to/hp-checkpoints \
  --step 200
```

此格式仅支持纯 AdamW。读取器在构造 Trainer 前核实 `format_version=1`、`optimizer_type=AdamW`、`completed_steps` / `optimizer_global_step` 与请求步数一致，逐文件验证原清单中的 SHA256，并检查模型数量、每个 optimizer 参数的模型键、全部 m/v 槽的唯一完整映射。按需读取数组，不重写源文件。Hash 只检测文件是否与清单一致，不能证明此前的坐标转换正确。

该清单没有 batch、数据位置和 loss scale，因而必须额外提供经过源配方核实的 `--resume-state`，不能按工具支持范围猜测缺失事实。例如源确为 batch1、单位 loss scale、完整顺序数据，且已完成 200 次更新时：

```json
{"global_step": 200, "global_batch_size": 1, "loss_scale": 1.0, "consumed_samples": 200, "data_order": "sequential"}
```

这个例子不是缺失元数据的默认值，也不能用于已切片的 suffix 数据。默认 `family-prefix` 输入始终使用目录自身的 `state.json`，拒绝外部 `--resume-state` 覆盖其计数。

## 状态校验和精度边界

- 模型与动量源必须为有限 FP32 值、shape 匹配；不静默转换 FP64、BF16 或损坏状态。
- 目标优化器的参数及 family 决定必需槽数量。AdamW 必须有 m/v，Muon 必须有 momentum。缺失、未知参数、未知 family、负 Adam v 均拒绝；真实标量状态不会跳过。
- 只允许同一目标参数的另一 family 的有限 FP32 单元素占位，不是所有小张量都忽略。融合参数不能由一半真实动量、一半占位拼成。
- 支持 Torch AdamW、HP AdamW、HP Muon，以及原生 `ChainedOptimizer` 和 `Float16OptimizerWithFloat16Params` 组合。SGD、AMSGrad、未知 wrapper 拒绝。
- 用 wrapper 的真实参数对象映射定位 FP32 main parameter，源值来自 `model.npz`；不将已舍入的 BF16 模型值升回 FP32 当作主权重。计算参数按自身 dtype 从主权重同步。
- Torch AdamW 恢复参数级 step；HP AdamW/Muon 恢复原生 group step。Torch FP32 step 无法逐次精确递增时拒绝恢复。
- 全部输入验证及目标状态分配完成后才写入参数、动量和计数；输入错误不产生部分更新。最后写入阶段的设备故障不提供事务回滚，磁盘故障也不提供多文件原子提交。
- 未知源键或不完整融合映射报错。默认转换器的 `CONVERSION.json` 记录计数、`state.json` 记录真实源路径，不生成文件 hash；`manifest` 输入会核验已有清单的文件 hash。

主权重存储精度与前向计算精度是不同配置；该转换不要求 HP 采用旧 MF 的逐算子实现。Muon 恢复只是此格式的条件支持项，不属于纯 AdamW 对齐实验的必要补丁。

## 回归范围

```bash
python -m pytest -q tests/ut/scripts
```

测试覆盖输入拒绝后的参数/动量/计数不变、真实 BF16/FP32-master wrapper、标量状态、dense/shared/expert/QKV 布局及元数据约束。

小模型测试实际执行 safetensors → NPZ → 原生 DCP 保存/读取 → 新对象恢复，核对全部主权重/m/v、step/LR、真实 `FixedBatchDataLoader` 的下一条样本及第 201 步 AdamW 独立 FP64 数学参考。该 CPU 测试仅 mock rank 通信，规划器、序列化、磁盘和优化器使用真实代码。

同一重载及下一步检查也覆盖 manifest 命名输入；另有损坏文件、计数不符、缺槽/重映射、错误优化器、缺少显式续训元数据的拒绝测试。

这些测试不等于固定完整 MF 归档的 NPU Trainer 物化通过，也不等于 200 步 loss 对齐通过。完整实验仍需验证真实导入路径、CANN/框架版本、模型配置、数据内容及每步观测结果。
