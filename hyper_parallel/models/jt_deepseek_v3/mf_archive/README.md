# 固定 MF 归档的显式续训兼容配方

本包用于从固定 MF r1.8.0 归档生成的检查点复现单卡续训。它不是新的 JT 默认实现，
也不把“loss 更接近旧 MF”当作数学正确性的证明。当前 JT 模型、Muon、AdamW 和 Trainer
默认入口不启用本包。总问题及逐项实验证据见 [#217](https://github.com/hf1s/hyper-parallel/issues/217)。

## 适用范围

- HP：Torch/TorchNPU 2.12、CANN 9.2.X、单张 Ascend 910B3；MF 参考使用原归档和 CANN 8.5.0。
- 4K Arrow，序列长度 4096，batch=1；所有并行度为 1，关闭重计算和交换。
- hidden=5120、vocab=152064、主干两层（dense、MoE）及一个 MTP 层；8 routed、2 shared、top-2。
- FP32 主参数、BF16 主计算；Muon/AdamW、QK clip、学习率、状态恢复沿用配方配置。
- 固定形状的归约调度仅在上述范围实测。模型工厂会拒绝未验证的设备、并行度及关键形状。
  不据此承诺其它硬件、CANN 版本或形状的逐位结果及性能。

本包没有参考 loss、未来梯度、逐步修正值、训练步号或元素坐标分支；`gradient_layout.json`
只有参数名称、形状和归约顺序。所有计算输入来自本次训练。没有运行时改写源代码、
全局替换框架函数、改写对象 `__class__` 或加载旧实验脚本。

## 必要配置与入口

保留完整 JT YAML 的网络、4K Arrow、学习率、单卡设置，在以下位置显式选择归档配方：

```yaml
model:
  _target_: hyper_parallel.models.jt_deepseek_v3.mf_archive.model.build_archived_model
  archive_profile: mf-r180-cann85-910b3-v1
  reference_weights: /your/converted-mf-step200
  # config: 完整网络配置，见上述范围；不由本包替换训练数据或标签。

optimizer:
  _target_: hyper_parallel.models.jt_deepseek_v3.mf_archive.optimizer.ArchivedOptimizerBuilder
  # muon_config / adamw_config / qk_clip_threshold: 保留原实验值。

dataloader:
  get_batch:
    reset_position_ids: false

checkpoint:
  restore_from: /your/native-hp-step200/global_step_200
  restore_optimizer: true
  restore_train_state: true
  restore_dataloader_state: true
  save_ckpt: false

training:
  train_iters: 400
```

原 `plan_overrides` 中将 `JTDeepseekV3Attention` 替换为 MLA 的那一项，目标改为
`hyper_parallel.models.jt_deepseek_v3.mf_archive.layers.ArchivedMLAAttention`，
其余配置保持。以上是覆盖项说明，不是可单独运行的完整 YAML。

```bash
# 使用目标环境自己的 CANN set_env.sh，保留其 Python 编译组件路径。
source /usr/local/Ascend/cann/set_env.sh
export CLOSE_MATMUL_K_SHIFT=0
export ASCEND_RT_VISIBLE_DEVICES=0
python -m torch.distributed.run --nproc_per_node=1 --master_port=31821 \
  -m hyper_parallel.models.jt_deepseek_v3.mf_archive.run your_complete_recipe.yaml
```

完整配置模板为本目录 [recipe.yaml](recipe.yaml)，先替换数据、转换权重和原生续训检查点路径。
正式验收还应记录源码提交/哈希、实际导入路径、CANN 版本、有效配置、恢复状态及每步输入。
离线转换必须包含 AdamW 一、二阶矩、Muon 动量、step、学习率调度器及数据游标；只载模型不构成续训对照。
`apply_lr_in_update` 是更新中折入 LR 的执行表示，本配方恢复全部状态后显式选择 true，
不会替换 moment、学习率或恢复步数。

## 正确性和旧实现兼容必须分别判断

| 行为与实现位置 | 分类及正确性依据 |
|---|---|
| Q/latent-KV、expert gate/up 存储融合后的逻辑矩阵边界 | 原生产品问题 [#242 / PR #243](https://github.com/hf1s/hyper-parallel/pull/243)。独立参数与融合存储的更新应等价。归档配方另外按旧 MF schema 拆 Q/KV/RoPE、Q-up/KV-up，不能把这部分都归给存储融合缺陷。 |
| 单分片 FP32 更新写回 | 产品问题 [#215 / PR #213](https://github.com/hf1s/hyper-parallel/pull/213)。归档配方依赖该修复保留 FP32 更新，不能借本包掩盖公共优化器缺陷。 |
| `model.py` 的 packed MTP 跨文档 shift | 明确的旧目标兼容。当前文档内 shift 对独立 packed 文档语义正确；旧归档跨文档行为不应成为默认。本包仅为了比较原检查点保留它，attention 仍按文档隔离。 |
| `model.py` 的绝对 RoPE 位置 | 旧配方选择；绝对位置与文档重置位置须在输入审计中明确，不能暗改后再称同配置。 |
| `optimizer.py` 的 Q-up 共用末分片 shape scale | 旧归档行为。现代逐矩阵算法应使用各自逻辑形状；共用末片缩放不推荐用于新训练，只在显式归档配方保留。 |
| `layers.py` 的 FP32 norm/router/embedding/head、残差和分支梯度合并 | 可选混合精度配方。较晚舍入保留更多信息，但不能仅以接近 MF 证明另一种 dtype 配置是代码错误。 |
| `losses.py`、`reductions.py` 的归约顺序、epsilon 字面量转换 | 旧编译图的数值兼容。FP32 加法不满足结合律，BF16 舍入边界可放大微小差异；固定 tile/core 不是通用正确性规则。 |
| `legacy_gemm.py` 的输出头权重梯度 MatMulV2/NZ | 旧 GEMM 累加路线兼容。结果的数学表达式仍为 `dy.T @ x`；仅改变此导数的内核路线，作用域内设置并恢复编译及格式开关。 |
| `optimizer.py` 的 sum-form 动量、NS 标量和 AdamW 运算顺序 | 归档优化配方/舍入兼容；完整恢复已有 moment 是前提。不能将零状态重启包装成恢复成功。 |

最新 MF PyNative、Megatron 和归档实现是三个不同参考，不应互相替代：

- [MF master PyNative 的 Muon schema](https://gitcode.com/mindspore/mindformers/blob/1c9258ca845e21ddd66fde83e262d5635359bd30/mindformers/pynative/base_models/gpt/gpt_model.py)：明确模型逻辑矩阵分组。
- [MF master PyNative Muon](https://gitcode.com/mindspore/mindformers/blob/1c9258ca845e21ddd66fde83e262d5635359bd30/mindformers/pynative/optimizer/muon.py)：FP32 norm、clamp 下限、随后 BF16；不要求旧固定归约调度。
- [Megatron 的 Emerging Optimizers 接口](https://github.com/NVIDIA/Megatron-LM/blob/c6e38c6c3fc308108486add972974c4c74d1d860/megatron/core/optimizer/emerging_optimizers.py)：可配置逻辑 QKV 拆分，分别正交化和缩放；普通 QKV 布局不能直接套用 JT MLA 的边界。

归档共享专家 down 的已核实样本中，MF norm 标量更近 FP64，但 92,955 个不同的 BF16
normalized 元素全部是普通 HP 结果更近 FP64。因此这里可证明归档兼容差异，不能证明
普通 HP norm 有数学错误，也不能推出任一实现对所有输入都更准确。

## 验证与资源限制

UT 检查路由导数的独立 FP64 数学参考、AdamW 状态意义、共享 key 的前反向布局、
归约尾块覆盖、状态恢复、Trainer 默认构造行为及代码边界。
完整验收必须由独立训练结果决定：从 MF200 续训至 400，total 首步 ≤0.0005，
未加权 MTP/aux 首步各 ≤0.001，三项 200 步每一步各 ≤0.005。
不得只比较最后一步、平均误差或乘权后数值。

本包含固定硬件归约的设备/主机同步，目标为可审计复现，没有吞吐优化承诺。
只允许一个训练线程使用编译开关作用域；不要在同一进程并发改变 TorchNPU 编译设置。
默认不保存大张量 dump；需要定位时优先保留统计、哈希和最小因果证据，再按需采集完整值。
