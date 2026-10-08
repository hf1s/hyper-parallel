# JT 与 MF Graph O1 的 4K 单卡数值对齐

此配置用于复现 MF 原生 Graph O1 的数值路径。它依赖相同的初始 FP32 权重、
相同训练步实际消费的 tokens / labels / mask / 文档边界，以及一致的学习率和优化器参数。
相同随机种子不保证两个框架的数据采样顺序相同。

## 使用

复制 `jt_deepseek_v3.mf_o1_single_card.yaml`，修改两处路径：

- `model.reference_weights`：含 `model.npz` 的目录，权重须匹配当前结构及投影布局。
- `dataset.data_path`：4K indexed 数据公共前缀，包含 tokens、labels、loss_mask、cu_seqlens 四组 bin/idx。

激活已经安装的 CANN / PyTorch / PTA 环境后执行：

```bash
cd /workspace/hyper-parallel
export ASCEND_RT_VISIBLE_DEVICES=0
set -o pipefail
mkdir -p /workspace/logs
torchrun --master_addr=127.0.0.1 --master_port=29972 --nproc_per_node=1 \
  --module examples.training_demo.train_text \
  examples/training_demo/jt_deepseek_v3/jt_deepseek_v3.mf_o1_single_card.yaml \
  2>&1 | tee /workspace/logs/jt-mf-o1-4k.log
```

配置为 0 Dense + 1 MoE + 1 MTP、hidden 5120、8 路由专家、2 共享专家、4K、单卡、100 步。
权重和数据不随源码分发；更换自己的数据可正常训练，但不能据此复现指定参考的逐步 loss。
构建器会校验权重名称、形状、dtype 和加载后的值，禁止缺失参数被随机初始化。

## 对齐策略

| 配置或实现 | 作用 |
| --- | --- |
| Router / RMSNorm 的 `fsdp_fp32_modules` | FSDP 保留实时 FP32 参数及计算，梯度仍回传，不保存冻结参数副本 |
| FP32 RoPE、融合 SwiGLU、FP32 MoE 概率与残差合并 | 与参考的转换边界一致，减少额外 BF16 舍入 |
| 原生 Q/KV 下投影 | 分开的 GEMM；此配置不启用 MLA 融合投影替换 |
| `reset_position_ids: false` | 匹配参考的全局位置；attention 仍按 packed 边界隔离 |
| `mtp_reset_on_document: false` | 匹配参考的全局 MTP 移位；默认仍在文档内移位 |
| `moe_combine_num_partitions: 40` | 复现此次 O1 融合归约的分块求和顺序；默认 1 |
| `optimizer.reference_muon: true` | 按逻辑矩阵执行 Muon，匹配 O1 BF16 系数、归一化、乘加顺序及共享缩放；默认关闭 |
| Muon `batch_ns: false` | 保留每个 NS 回调输入的原始维数，区分二维投影与三维专家；公共优化器默认仍批量执行 |

`reference_muon` 使用 legacy 五轮 NS。二维归一化保留 FP32 中间值，三维专家归一化
保留 BF16 范数边界；二维多项式计算 `(c * A) @ A`，三维计算 `c * (A @ A)`。
系数在融合乘加前舍入到 BF16，矩阵乘法输出仍为 BF16。

这是一组显式的数值复现选项，不是跨硬件、跨编译器的逐位一致性保证。
40 个分块来自已观测的参考内核，其他设备或编译配置需要重新核验。
本轮仅验证 4K、单卡、0 Dense + 1 MoE + 1 MTP；不据此声称 256K 或所有并行组合通过。

## 已完成验证

在 910B 上，MF Graph O1 / MindSpore 2.7.2 / CANN 8.5.0 与
HP / torch、PTA 2.12 / CANN 9.2.0-beta.2 比较：

- 每步校验 tokens、按 loss_mask 折叠的监督 targets 及 packed 边界，100 步全部相同。
- 首步总 loss 均为 `16.920595169067383`；首步各分量不要求逐位一致。
- 100 步最大总 loss 绝对差为 `0.0043659210205078125`，全部小于 `0.005`。
- 二维及真实专家尺寸 `[8, 1536, 5120]` 的五轮 NS 独立复算与原生参考逐元素一致。

这是已完成的一组训练对比，不承诺其他初始化、数据顺序或环境也达到相同误差。
参考数据下沉模式下须在模型入口记录实际训练输入，不能把 Dataset 初始化预读当成训练步。
