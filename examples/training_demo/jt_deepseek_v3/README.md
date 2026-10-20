# JT DeepSeek V3 4K 单卡配方

本目录提供 HP 配方 `hp_4k_1_card.yaml` 和 MF 参考配方 `mf_4k_1_card.yaml`。
运行前需按实际环境设置数据、参考权重和断点路径。

## Router 参数精度

HP 配方选择 FP32 router 参数，与本目录 MF 参考配方的 FP32 router 权重契约一致。
该选择通过已有的 `fsdp_config.mix_precision.custom_params` 配置实现：

- `model.layers.1.mlp.gate.weight`：主干 MoE 的 router。
- `mtp.layers.0.transformer_layer.mlp.gate.weight`：MTP MoE 的 router。

这两个参数的前向计算副本和梯度归约使用 FP32，其余参数沿用全局 BF16 计算副本策略。
参数名对应当前“两层主干、第一层为 Dense、一个 MTP 层”的模型配置；调整模型结构时，
需同步检查覆盖列表。FSDP 策略构建时会拒绝不存在的参数名。

`model_init_dtype: float32` 只指定初始化精度，不能保证 FSDP 展开后的计算副本仍为 FP32。
同样，gate 内 `F.linear(hidden_states.float(), self.weight.float())` 指定的是计算精度：
如果权重已被 FSDP 舍入成 BF16，再转成 FP32 无法恢复原来的有效位。
因此，保留原始 FP32 router 权重需要同时配置参数副本精度。

这是本参考配方的精度选择。BF16 router 参数配合 FP32 GEMM 也是有效配置；
其他模型可根据精度和资源需求决定是否保留这些覆盖项，不需要修改 FSDP 全局策略。
两处 router 共 81,920 个参数，FP32 与 BF16 计算副本的纯张量字节差为 163,840 字节；
这不是训练峰值显存或吞吐的测量结果。

Router 精度配置不能单独保证跨框架的训练轨迹一致。比较续训结果还需一致的模型、
优化器状态、步数、数据位置，以及逐步核验 total、未加权 MTP 和未加权 aux loss。
