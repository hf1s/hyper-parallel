# Muon 共享专家归一化的跨后端诊断

本例对应子问题 [#240](https://github.com/hf1s/hyper-parallel/issues/240)，关联总问题 [#217](https://github.com/hf1s/hyper-parallel/issues/217) 中第 396 步共享专家 down 的首差。
它比较同一真实输入在普通 NPU norm 与已观测旧 MF DVM 归约顺序下的结果，只输出统计和完整哈希。
这是显式诊断复现，不导入或修改 HP 优化器，不保存模型，不用参考结果修正输入。

## 已定位到什么

MF200 完整模型、Muon 动量和 AdamW 状态恢复后，MF/HP 独立继续 201–400 步。
第 99 组对照在 398 步的未加权 MTP 误差为 0.009670257568359375，超过 0.005；总 loss 与 Aux 已通过。
定位顺序为：先按训练步缩小边界，再分前向/反向/优化器，最后检查单个更新中的归一化。

- 395 的 48 项模型 FP32/BF16 值、目标动量精确；396 的 46 项裁剪前梯度、平方和及范数精确。
- 396 目标裁剪后梯度、新动量精确；两端原更新单独重放均精确复现各自完整训练输出。
- 首差落在主干 `model.layers.1.mlp.shared_experts.down_proj.weight`，形状 `(5120,3072)`。
  MF 对应 `decoder.layers.1.mlp.shared_experts.linear_fc2.weight`，无需转置映射状态。
- 两端归一化前 FP32 输入相同；归约/开方差异造成 92,955 个 BF16 normalized 值不同，随后影响 NS 更新。
- 只改变该形状实时输入的归约顺序，完整 15,728,640 个更新后权重精确等于 MF；动量不变。
  因素对所有训练步、主干与 MTP 同形状矩阵应用，无步号或坐标特判。

### 代码和算子位置

| 位置 | 行为 |
|---|---|
| MF 归档 `mindformers/core/optim/muon.py::zeropower_via_newtonschulz5_2d` | 源码 `x / (x.norm() + 1e-7)`；真实二维编译图执行 FP32 归一化后转 BF16 |
| MF 原生 DVM `Cast_Mul_ReduceSum_split-op0` | 9776 元素/块、1609 块、尾 8832 元素、16 lanes、40 vector cores |
| HP 实验 `muon_jit_probe.py::graph_jit_ns`，配套 `hp_ns_epsilon229_factor.py` | FP32 norm，加固定 epsilon，除法后转 BF16；属于实验控制，不等于未经修改的 HP 默认 |
| HP 产品 `hyper_parallel/core/optimizer/muon.py::zeropower_via_newtonschulz5` | `mat_x / (mat_x.norm(...) + epsilon)`；输入 dtype 由上游决定 |
| 本例 `_archived_normalization` | 复现已测平方和顺序，使用设备 FP32 sqrt，epsilon 保持 `1.0011717677116394e-7` |

MF 归档源码 SHA256：`43fee7992bc12281a2c4586799fb99f927cc15d0a9c085fe016ab44c588bdf49`。
源码 dtype 不能代替编译图 dtype 证据；固定 tile/core 数也不能推广到其它图、芯片或 CANN 版本。

## 哪边更准确

| 同一真实输入 | 普通 HP 实验路径 | MF 原生/归约重放 | FP64 参考 |
|---|---:|---:|---:|
| norm | 0.22601467370986938 | 0.2260146588087082 | 0.22601466135891418 |
| 92,955 个不同 BF16 normalized 值中更近 FP64 的数量 | 92,955 | 0 | 相同输入与 epsilon |

MF 的 norm 标量更接近 FP64，但跨过 BF16 舍入边界后，普通 HP 的 normalized 值反而全部更近 FP64。
因此“更接近 MF 训练轨迹”不等于“数学上更正确”。此例解决归档行为的可复现性和诊断问题，
不把普通归约判为 HP 数学缺陷，不以此固定调度替换通用默认。

2026-10-16 重新查询并固定的行业实现：

- [MF master `1c9258ca` 的 PyNative Muon](https://gitcode.com/mindspore/mindformers/blob/1c9258ca845e21ddd66fde83e262d5635359bd30/mindformers/pynative/optimizer/muon.py)：
  `_normalize_newton_schulz_input` 使用 FP32 norm、除以 `clamp(norm,min=eps)`，之后转 BF16；后续使用融合 addmm。
- [Megatron-LM `3e2807ce`](https://github.com/NVIDIA/Megatron-LM/blob/3e2807ce5e787f81b7275be2383092fe0c3bfc71/megatron/core/optimizer/muon.py)：
  Muon 文件已是兼容入口，具体实现委托 Emerging Optimizers。
- [NVIDIA Emerging Optimizers `4bbed19e`](https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/4bbed19e681caeaeb630f7ae91d0518d363404b8/emerging_optimizers/orthogonalized_optimizers/muon_utils.py)：
  要求 FP32 输入，默认 functional.normalize，可选 FP64 norm；进入 BF16 迭代还取决于 matmul precision。
  此处是当前 EO 上游参考，不声称任何 Megatron 安装都锁定该 commit。

这些实现均未要求重现旧 DVM tile/core 顺序。`clamp(norm,eps)` 与 `norm+eps` 是不同定义，
本次未改变 epsilon 定义；现代归一化算法/精度配置应与既有 [#210](https://github.com/hf1s/hyper-parallel/issues/210) 分别讨论。

## 运行

已验证设备为 125 的 Ascend 910B3。MF 原生证据来自 MindSpore 2.7.2 / CANN 8.5.0；
比较程序验证环境为 torch/torch_npu 2.12 / CANN 9.2.0-beta.2。没有验证 910C、多卡或吞吐。
先预留一张空闲卡，在独立进程运行；程序包含 CPU 归约、设备同步，不能作为训练热路径的性能实现。

准备 `ns_input.npy`：保存**实际送入二维 NS 的输入**，形状 `(5120,3072)`，dtype 为 FP32。
如果原路径先做 BF16 舍入，保存的值必须经过该舍入后再提升为 FP32，不能从未舍入梯度直接比较。
本例真实 Nesterov 输入的构造为 `(postclip_gradient + updated_momentum * 0.95).bfloat16().float()`；
这两个操作数必须来自同一次真实训练步，并先验证各自原更新重放精确。

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python examples/torch/alignment/compare_shared_down_normalization.py \
  --input /your/diagnostic/ns_input.npy \
  --output /your/diagnostic/normalization.json
```

`--device` 是可见设备编号，默认 0。输出路径必须不存在；输入不被覆盖，NPY 使用 `allow_pickle=False`。
报告保存输入、两种完整 BF16 输出的 SHA256、两个 norm、平方和，以及相对于 FP64 的误差比较计数。
不同形状、dtype、非有限值会拒绝；它不接受训练步或目标 loss 参数，也不读取 MF 参考结果。

## 验证范围与复现证据

程序需要对真实输入验证以下独立证据，而不是仅断言自己调用自己的输出相同：

1. 输入完整哈希对应 MF 原生统计的 Cast 输出。
2. 归约结果对应 MF 原生统计的平方和与 normalized 完整哈希。
3. 候选输出与实际执行完整训练的独立因素逐元素相同。
4. 普通路径的 norm、不同 BF16 数量及 FP64 比较复现先前独立 replay。
5. CLI 成功退出，拒绝覆盖现有报告、拒绝不支持输入，输入保持不变。

总问题 #217 发布实际执行记录、原生 IR/CSV、完整 200 步配置/指标及独立观测保真。
本例本身不执行训练；运行它成功不等于完整 200 步验收成功。公开内容不含用户数据、权重或梯度数组。
目标值采集后保留哈希和复现入口，审计完成即可删除不再使用的中间数组。


实际验证：[16项真实输入/CLI检查](https://github.com/xingleixu123/hyper-parallel/blob/f4825d1d03888ef17042e487b7efbecb05d7437d/evidence/frontier396-pr-diagnostic-validation.json)；
[原生统计与完整因果链](https://github.com/xingleixu123/hyper-parallel/blob/f4825d1d03888ef17042e487b7efbecb05d7437d/evidence/frontier396-causal-chain.json)；
[完整200步三项验收](https://github.com/xingleixu123/hyper-parallel/blob/f4825d1d03888ef17042e487b7efbecb05d7437d/evidence/frontier396-shared-independent200-audit.json)。
新因素使总loss、原始MTP、原始Aux最大误差分别降到1.90735e-6、3.17891e-6、1.29733e-6，三项每一步均低于0.005。
完整训练执行的是独立实验因素；本CLI另在同一真实输入上验证与该因素完整输出相同，不能说CLI自身执行过200步训练。
