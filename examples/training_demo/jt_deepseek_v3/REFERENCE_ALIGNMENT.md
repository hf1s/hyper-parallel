# JT 与 MF Graph O1 的 4K 单卡数值对齐

此配置用于复现 MF 原生 Graph O1 的数值路径。它依赖相同的初始 FP32 权重、
相同训练步实际消费的 tokens / labels / mask / 文档边界，以及一致的学习率和优化器参数。
相同随机种子不保证两个框架的数据采样顺序相同。

## 使用

选择并复制对应 YAML，修改两处路径：

| YAML | 结构 | hidden |
| --- | --- | --- |
| `jt_deepseek_v3.mf_o1_single_card.yaml` | 0 Dense + 1 MoE + 1 MTP | 5120 |
| `jt_deepseek_v3.mf_o1_dense_single_card.yaml` | 1 Dense + 1 MoE + 1 MTP | 3072 |

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

两份配置均为 8 路由专家、2 共享专家、4K、单卡、100 步。运行 Dense 版本时替换上述 YAML 路径。
Dense 版本为容纳新增层只缩小 hidden；vocab、FFN intermediate、MLA 维度、head 数及优化器参数保持一致。
Dense 的 `plan_overrides` 额外匹配 `model.layers.0.mlp`，复用已有融合 SwiGLU。
重新初始化或转换对应结构的完整权重，不能直接复用另一份配置的 `model.npz`。
权重和数据不随源码分发；更换自己的数据可正常训练，但不能据此复现指定参考的逐步 loss。
构建器会校验权重名称、形状、dtype 和加载后的值，禁止缺失参数被随机初始化。

## 对齐策略

| 配置或实现 | 作用 |
| --- | --- |
| JT 的 `JTFSDP2Manager` | 局部选择 Router / RMSNorm 的 FP32 子单元；复用公共 FSDP 的分片、梯度缩放和预取，不保存冻结参数副本 |
| FP32 RoPE、融合 SwiGLU、FP32 MoE 概率与残差合并 | 与参考的转换边界一致，减少额外 BF16 舍入 |
| 原生 Q/KV 下投影 | 分开的 GEMM；此配置不启用 MLA 融合投影替换 |
| `reset_position_ids: false` | 匹配参考的全局位置；attention 仍按 packed 边界隔离 |
| `mtp_reset_on_document: false` | 匹配参考的全局 MTP 移位；默认仍在文档内移位 |
| `moe_combine_num_partitions: 40` | 复现此次 O1 融合归约的分块求和顺序；默认 1 |
| `moe_combine_group_size` | 相邻专家先组成 FP32 部分和；默认 1，hidden 3072 的参考配置设为 3，且须整除 top-k |
| `optimizer.reference_muon: true` | 按逻辑矩阵执行 Muon，匹配 O1 BF16 系数、归一化、乘加顺序及共享缩放；默认关闭 |
| JT 的 `_JTReferenceMuon` | 在参考模式下保留 NS 输入原始维数，区分二维投影与三维专家；公共 Muon 实现和配置接口不变 |

`reference_muon` 使用 legacy 五轮 NS。二维归一化保留 FP32 中间值，三维专家归一化
保留 BF16 范数边界；二维多项式计算 `(c * A) @ A`，三维计算 `c * (A @ A)`。
系数在融合乘加前舍入到 BF16，矩阵乘法输出仍为 BF16。

这是一组显式的数值复现选项，不是跨硬件、跨编译器的逐位一致性保证。
40 个分块和组大小来自已观测的参考内核，其他设备或编译配置需要重新核验。
组大小 3 / top-k 6 时计算 `((p0+p1)+p2)+((p3+p4)+p5)`，其中 `pi` 是 FP32 专家输出与概率的乘积；
分块按 token-group 工作量计算，最终才转回激活 dtype。组大小 1 保持第一份配置的求和顺序。
这是显式数值策略，不根据 hidden 或输入数据自动猜测参考编译器的调度。
本轮只验证上述 4K 单卡结构；不据此声称 256K 或所有并行组合通过。

## 改动范围与 PyNative 参考

对齐代码集中在 `models/jt_deepseek_v3/`。公共 Muon、公共 router、FSDP manager、
`ModelAdapterSpec` 均恢复到对齐前版本。JT 自己计算路由分数与辅助 loss；
局部 FSDP manager 只选择精度子单元和策略，仍调用公共实现管理参数、梯度与通信；
参考 Muon 只重写 NS 张量的调度方法，其更新、状态、分布式通信及 AdamW 继续复用公共实现。
未开启 `reference_muon` 时仍使用原来的公共 optimizer builder。

审查参考固定为 MF master [`44a47972`](https://github.com/mindspore-ai/mindformers/tree/44a47972f9ebd5ba746433588b99ed2068d419d7)，
当次读取的 GitHub / AtomGit master 一致：

- [PyNative Router](https://github.com/mindspore-ai/mindformers/blob/44a47972f9ebd5ba746433588b99ed2068d419d7/mindformers/pynative/transformers/moe/router.py)：路由分数及辅助目标属于模块自己的前向逻辑。
- [PyNative FSDP 组装](https://github.com/mindspore-ai/mindformers/blob/44a47972f9ebd5ba746433588b99ed2068d419d7/mindformers/pynative/base_models/gpt/parallelize.py)：模型组装层选择 wrap 单元和精度策略。
- [PyNative Muon](https://github.com/mindspore-ai/mindformers/blob/44a47972f9ebd5ba746433588b99ed2068d419d7/mindformers/pynative/optimizer/muon.py)及
  [布局工具](https://github.com/mindspore-ai/mindformers/blob/44a47972f9ebd5ba746433588b99ed2068d419d7/mindformers/pynative/optimizer/muon_utils.py)：按模型布局拆分逻辑矩阵，求解后恢复。

这里参考职责划分，没有把最新 PyNative 当成旧 Graph O1 的同一数值算法。
最新 PyNative 使用 FP32 范数及 `clamp(norm, min=eps)`、融合 addmm/baddbmm、逐逻辑块缩放；
指定 Graph O1 基线的二维/三维归一化、融合舍入和 packed 参数共享缩放不同。
这些差异只在 JT 的显式参考模式中保留，不改公共优化器来强制复现旧图编译器。

## 已完成验证

在 910B 上，MF Graph O1 / MindSpore 2.7.2 / CANN 8.5.0 与
HP / torch、PTA 2.12 / CANN 9.2.0-beta.2 比较：

- 每步校验 tokens、按 loss_mask 折叠的监督 targets 及 packed 边界，100 步全部相同。
- 0 Dense / hidden 5120：首步总 loss 均为 `16.920595169067383`，100 步最大绝对差 `0.0043659210205078125`。
- 1 Dense / hidden 3072：首步总 loss 均为 `16.338472366333008`，100 步最大绝对差 `0.0038080215454101562`。
- 两组首步总 loss 完全一致，各分量不要求逐位一致；100 步所有总 loss 绝对差均小于 `0.005`。
- 公共改动收回 JT 后，两组均重新训练 100 步；各自的 loss 分量、总 loss、梯度范数原始记录与收敛前逐字节相同。
- 二维及真实专家尺寸 `[8, 1536, 5120]` 的五轮 NS 独立复算与原生参考逐元素一致。

这是已完成的两组训练对比，不承诺其他初始化、数据顺序或环境也达到相同误差。
参考数据下沉模式下须在模型入口记录实际训练输入，不能把 Dataset 初始化预读当成训练步。
