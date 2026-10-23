# Optimizer 使用指南

HyperParallel 提供 Muon+AdamW 链式优化器和学习率调度器，支持分片优化器与 FSDP/HSDP 集成。

## 核心概念

| 优化器 | 说明 | 适用场景 |
|--------|------|----------|
| AdamW | 标准 AdamW 优化器 | 通用训练 |
| Muon | Momentum-based optimizer | 大模型训练 |
| ChainedOptimizer | Muon+AdamW 链式组合 | 混合参数组训练 |

Muon 优化器针对特定参数组（如 embedding、projection 层）使用 momentum-based 更新策略，其余参数使用 AdamW，实现混合优化。

## 接口概览

| 接口 | 说明 |
|------|------|
| `AdamW` | 标准 AdamW 优化器 |
| `Muon` | Momentum-based 优化器 |
| `ChainedOptimizer` | 链式优化器（Muon+AdamW 组合） |
| `get_hyper_optimizer` | 优化器工厂函数 |

---

## 基础使用

### 1. 使用 get_hyper_optimizer 创建链式优化器

```python
from hyper_parallel.core.optimizer import get_hyper_optimizer

optimizer = get_hyper_optimizer(
    model=model,
    muon_params=muon_param_groups,   # Muon 参数组
    adamw_params=adamw_param_groups,  # AdamW 参数组
    muon_kwargs={"lr": 0.02, "momentum": 0.95},
    adamw_kwargs={"lr": 3e-4, "weight_decay": 0.1},
)
```

`get_hyper_optimizer` 内部创建 Muon 和 AdamW 实例，并用 `ChainedOptimizer` 将它们组合。参数分组：

- `muon_params`：Muon 优化的参数组（通常为大型 projection 参数）
- `adamw_params`：AdamW 优化的参数组（通常为 embedding、bias 等参数）
- 任一参数组为空列表时，对应优化器不创建

### Muon 的跨参数 Newton–Schulz 批处理策略

`ns_batch_across_params` 是显式布尔选项，默认 `True`，保留按逻辑矩阵形状合并不同参数的
Newton–Schulz（NS）调用。需要让不同原参数分别执行 NS 时，可设置为 `False`：

```python
optimizer = get_hyper_optimizer(
    model=model,
    muon_params=muon_param_groups,
    adamw_params=adamw_param_groups,
    muon_kwargs={"lr": 0.02, "ns_batch_across_params": False},
    adamw_kwargs={"lr": 3e-4},
)
```

训练配置通过现有 `muon_config` 传递同名字段；`muon_ns_batch_across_params` 前缀形式也受支持。
也可以在 Muon 参数组中覆盖该值。整数 `0/1`、字符串等非布尔值会被拒绝。

`False` 只隔离不同原参数；单个专家参数自带的矩阵批、卷积 reshape，以及同一参数经
`reshape_fn` 或 `ns_transform_fn` 产生的同形状片段仍保留原有批处理。它不会拆开融合参数的
逻辑 Q/KV 或 gate/up 边界，也不改变通信、owner、布局、动量、NS 系数、缩放、学习率和权重衰减。

该策略同时适用于原生路径和 `NSInputTransform` 路径。所有 callback 的函数签名保持不变；
设置 `False` 后，`zeropower_fn(input, steps=...)` 会收到按原参数隔离的输入，调用次数和输入批大小
可能变化。单个参数内部仍可能传入三维或更多维矩阵批，不能把 `False` 理解为逐二维矩阵执行。

浮点规约和矩阵乘法可能随批形状采用不同执行顺序；此选项提供数值策略选择，不保证任意设备或
并行布局之间逐位相同，也不表示批处理算法本身错误。

新 optimizer checkpoint 保存每组策略，加载时以保存值为准。旧 checkpoint 缺该字段时恢复为
兼容值 `True`，即使当前构造配置为 `False`；这也适用于 FQN 展平状态及训练器 DCP 加载。
只加载模型权重时，当前 optimizer 配置保持不变。需要对恢复后的策略进行显式调整时，应在加载
optimizer 状态之后设置参数组字段，并记录该调整。

### 2. 参数分组示例

```python
# 将模型参数分为 Muon 和 AdamW 两组
muon_params = []
adamw_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    # 大型 projection 层用 Muon
    if any(key in name for key in ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"]):
        muon_params.append(param)
    else:
        # embedding、bias、norm 等用 AdamW
        adamw_params.append(param)

optimizer = get_hyper_optimizer(
    model=model,
    muon_params=muon_params,
    adamw_params=adamw_params,
    muon_kwargs={"lr": 0.02, "momentum": 0.95},
    adamw_kwargs={"lr": 3e-4, "weight_decay": 0.1},
)
```

### 3. 训练循环

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        output = model(batch)
        loss = criterion(output)
        loss.backward()

        # ChainedOptimizer 自动处理两个优化器的 step
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
```

---

## 分片优化器（FSDP/HSDP 集成）

与 FSDP/HSDP 配合使用时，优化器状态自动分片：

```python
from hyper_parallel import fully_shard, init_device_mesh
from hyper_parallel.core.optimizer import get_hyper_optimizer

mesh = init_device_mesh("npu", (dp_size,), mesh_dim_names=("dp",))
model = fully_shard(model, mesh=mesh)

# FSDP 分片后，优化器状态也自动分片
optimizer = get_hyper_optimizer(
    model=model,
    muon_params=muon_param_groups,
    adamw_params=adamw_param_groups,
    ...
)
```

---

## gradient scaling factor + clip_grad

```python
from hyper_parallel import fully_shard, init_device_mesh

mesh = init_device_mesh("npu", (dp_size,), mesh_dim_names=("dp",))
model = fully_shard(model, mesh=mesh)

# 设置梯度缩放因子
model.set_gradient_scaling_factor(scale_factor=0.5)

# clip_grad 增强：对齐 clip_grad_norm_ reduction 与各 grad 的 process group
# 确保在混合并行场景下 clip_grad 正确工作
```

---

## 性能建议

1. **Muon 参数选择**：通常将大型 projection 层参数分配给 Muon，embedding/bias/norm 分配给 AdamW
2. **学习率配置**：Muon lr 通常比 AdamW lr 大（如 0.02 vs 3e-4）
3. **warmup**：建议使用 warmup_steps，从 0 线性增加到目标 lr
4. **weight_decay**：AdamW 的 weight_decay 建议设为 0.1
5. **与 FSDP 配合**：FSDP 下优化器状态自动分片，无需额外配置
