# DeepSeek-V3.2 模型族适配

DeepSeek-V3.2 在 DeepSeek-V3 的 MLA + MoE 结构上加入 DeepSeek Sparse Attention（DSA）：每层由
lightning indexer 为每个 query 选出 `index_topk` 个 key（官方 config 为 2048），MLA 只在这些 key
上计算 attention。模型主体直接使用 Transformers 原生的 `DeepseekV32ForCausalLM`，本族不带建模代码；
本目录是 HyperParallel 的适配层，包括模块替换工厂、CP wrapper、TP 分片规则和 RoPE 物化。

## 注册身份

| 项目 | 值 |
|---|---|
| HyperParallel 族名（`models/<family>/` 目录名） | `deepseek_v32` |
| HF `config.model_type` | `deepseek_v32` |
| HF `architectures` | `DeepseekV32ForCausalLM` |
| 实现来源 | Transformers 原生实现（`transformers.models.deepseek_v32`），不需要 `trust_remote_code` |

## 目录结构

```text
models/deepseek_v32/
├── __init__.py
├── adapter/
│   ├── registration.py              # ModelAdapterSpec：替换工厂、CP、TP 分片规则、RoPE 物化
│   ├── materialization.py           # 加载权重时重建 RoPE 的 inv_freq / original_inv_freq
│   ├── conversion/
│   │   └── module_replacement.py    # RMSNorm、DSA attention、GroupedExperts 的替换工厂
│   ├── distributed/
│   │   └── context_parallel.py      # DSA 的 CP wrapper
│   └── policies/
│       └── sharding.py              # MLA 与 indexer 的 TP 分片规则
├── generated/                       # codegen 生成的模型代码，示例不使用
└── recipes/
    └── train.yaml                   # 完整规格的单进程配方（master 已有，未在本目录的验证范围内）

examples/training_demo/deepseek_v32/
├── train_single_card.yaml           # 裁剪模型，单卡
├── train_8card.yaml                 # 同一裁剪模型，8 卡（TP2 × CP2 × DP2，EP4）
├── run_single_card.sh
└── run_8card.sh
```

## 训练语义

本族不修改 Transformers 的模型代码，模块替换和并行化都保持计算等价，因此训练语义就是
Transformers 实现的语义。它和 MindSpeed-LLM 的 DeepSeek-V3.2 预训练脚本
（`examples/mcore/deepseek32/pretrain_deepseek32_671b_4k_A3_ptd.sh`）有以下差别：

| 项目 | 本族（与 Transformers 一致） | MindSpeed-LLM 预训练脚本 |
|---|---|---|
| 专家选择 | sigmoid 分数加 `e_score_correction_bias` 做分组 top-k（`n_group` 8、`topk_group` 4、top-8），门控值归一化后乘 `routed_scaling_factor` | 相同 |
| 专家 bias | FP32 buffer，训练中保持初始值 | `--moe-router-enable-expert-bias`，每步按专家负载 ±1e-3 更新 |
| 负载均衡 loss | 无 | `--moe-router-load-balancing-type seq_aux_loss`，系数 1e-4 |
| MTP | Transformers 加载时忽略第 61 层（`_keys_to_ignore_on_load_unexpected`），不训练 | `--mtp-num-layers 1`，loss 系数 0.3 |
| indexer KL loss | 无，indexer 参数不更新 | `--indexer-loss-coeff 1.0` |
| indexer Hadamard 旋转 | 不做 | 默认在打分前对 q/k 做 Hadamard 旋转，`--no-use-sparse-c8-indexer` 关闭 |

几点说明：

- **专家 bias 与负载均衡、MTP**：这些是预训练目标，Transformers 实现只包含推理所需的部分。
  DeepSeek 官方推理代码（`DeepSeek-V3.2-Exp/inference/convert.py`）同样跳过第 61 层。需要这些
  目标时要在模型侧新增代码，本族不包含。
- **indexer**：indexer 的输出只用于 top-k 选择，这一步不可导，所以语言模型 loss 不会给 indexer
  梯度，indexer 保持 checkpoint 中的值。
- **Hadamard 旋转**：官方推理代码在 indexer 打分前对 q/k 乘归一化的 Hadamard 矩阵，配合 FP8
  量化打分。它是正交变换，不改变 q·k，Transformers 按 BF16 计算时跳过了这一步（见其
  `DeepseekV32Indexer` 的注释）。两种做法只差 BF16 舍入，在 top-k 边界上可能选出不同的 key。

## 适配内容

- **模块替换**（`adapter/conversion/module_replacement.py`，由 YAML 的 `plan_overrides` 选用）：

  | 匹配 | 替换为 | 说明 |
  |---|---|---|
  | `*.input_layernorm`、`*.post_attention_layernorm`、`model.norm` | 框架 `RMSNorm` | NPU 融合 RMSNorm |
  | `*.self_attn` | 框架 `DeepseekV32DSAAttention` | `q_a_proj` 与 `kv_a_proj_with_mqa` 合并为 `linear_qkv`，加载时由替换模块声明的转换拼接权重；indexer 与稀疏 attention 使用 CANN 融合算子 |
  | `*.mlp.experts` | 框架 `GroupedExperts` | grouped GEMM |

  DSA 替换只按因果关系计算 attention，不读取 `attention_mask` 的内容，不支持 padding mask、dropout
  和 KV cache。每行数据是一条完整的因果序列，这与 Megatron/MindSpeed 默认（不开
  `--reset-attention-mask`）的预训练数据一致。DSA 算子的依赖见
  [functional README](../../components/functional/README.md#deepseek-v32-dsa)。
- **CP**（`adapter/distributed/context_parallel.py`，YAML 中 `when: cp` 的条目，`cp_size > 1` 时生效）：
  query 留在本 rank，异步 all-gather indexer key 与 MLA 压缩 KV，每个 rank 只使用到本地 query 结尾的
  因果前缀。
- **EP**（YAML 中 `when: ep` 的条目，`ep_size > 1` 时生效）：直接使用框架内置的
  `deepseekv3_ep_compute_fn`（`deepseekv3_sigmoid_group_shared` 原型）。它的路由与 Transformers
  router 相同；token 分发、本地 grouped GEMM 和合并由框架 EP 骨架完成，shared experts 与分发通信
  重叠计算。条目只能匹配 MoE 层：裁剪模型只有第 1 层，写成 `model.layers.1.mlp.forward`；改变层数
  或 `first_k_dense_replace` 时要同步修改。
- **TP 分片规则**（`adapter/policies/sharding.py`）：`linear_qkv`（低秩下投影）复制，`q_b_proj` /
  `kv_b_proj` 按 head 列切，indexer 的 `wq_b` / `wk` / `weights_proj` 复制（indexer 头不切分）。
  MoE 层在 TP 下按 TP-extend-EP 处理，专家只按 EP 切分，所以 `tp_size > 1` 时 `ep_size` 也要大于 1。
- **RoPE 物化**（`adapter/materialization.py`）：Transformers 的 `inv_freq` / `original_inv_freq` 是
  非持久 buffer，checkpoint 里没有。`from_pretrained` 在 meta 设备上建模型、分配存储后直接加载
  checkpoint，不再执行 Transformers 的权重初始化，这两个 buffer 就停留在未初始化的内存（实测为
  NaN，loss 随之变成 NaN）。materialization provider 按同一个 config 在 CPU 上算出 YaRN 频率，登记为
  可重建 buffer，构建流程在加载后写回（与 Qwen3.5 的做法相同）。只要加载权重（官方权重、示例用的
  裁剪 checkpoint）就需要它；`from_config` 随机初始化时 Transformers 的初始化会算出这两个 buffer。

## 依赖与权重

- Transformers 需要包含 `deepseek_v32`（`requirements.txt` 要求 `transformers>=5.11.0`，本地测试使用
  5.20.2）。RMSNorm、GroupedExperts 和 DSA 替换只能在 Ascend NPU 上运行；DSA 依赖 CANN 与
  `cann_ops_transformer`，版本见上面的 functional README。
- `from_pretrained` 可以直接指向官方 checkpoint。官方权重是 FP8（`quant_method: fp8`，128×128 block，
  `scale_fmt: ue8m0`），默认的 DCP loader 在读取时按 block 反量化后再转换成训练参数
  （`hyper_parallel/components/checkpoint/quantized_hf_storage.py`）。第 61 层 MTP 的权重不属于模型，
  加载时会被忽略并打印 warning。

## 使用

### 单卡与 8 卡裁剪示例

两个 YAML 加载同一份裁剪后的 BF16 checkpoint。它保留官方的 hidden、MLA、indexer 和词表维度，只有
2 层（第 0 层 dense、第 1 层 MoE）和 16 个路由专家；16 是保持官方分组路由的最小专家数。共 3.40B
参数，FP32 参数、FP32 梯度和 AdamW 两个 FP32 动量约 51 GiB（估算，不含激活）。

| | `train_single_card.yaml` | `train_8card.yaml` |
|---|---|---|
| 拓扑 | 单卡 | `tp 2 × cp 2 × dp 2`，FSDP 在 dp 上切分；MoE 层 `ep 4 × edp 2`，每个 EP rank 4 个专家 |
| loss parallel | 关 | 开（按 TP 切词表） |
| 优化器状态 | `optimizer.swap` 换出到 host，降低单卡显存 | 不换出 |

除上表外两个 YAML 完全相同：`seed: 1234`、`global_batch_size: 2`、序列长度 4096、AdamW lr 1e-5
cosine 到 1e-7、无 warmup。

**1. 生成裁剪 checkpoint（一次）。** 用 Transformers 按官方 config 裁剪、随机初始化后保存，只需要官方
目录中的 `config.json` 和 tokenizer 文件：

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

source, target = "/path/to/DeepSeek-V3.2", "/path/to/DeepSeek-V3.2-crop"
config = AutoConfig.from_pretrained(
    source, num_hidden_layers=2, first_k_dense_replace=1, n_routed_experts=16,
    # Transformers 按读入时的 61 层推导逐层类型，覆盖层数后不会重新推导，因此一并给出。
    mlp_layer_types=["dense", "sparse"], layer_types=["indexed_attention"] * 2,
)
del config.quantization_config  # 随机初始化的权重是 BF16，不是官方的 FP8
torch.manual_seed(1234)
AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).save_pretrained(target)
AutoTokenizer.from_pretrained(source).save_pretrained(target)
```

**2. 准备数据。** 数据是 Megatron `.bin/.idx`。可以直接使用 MindSpeed 的
`data_convert_deepseek32_pretrain.sh` 生成的数据，也可以用 HyperParallel 的工具从 JSONL 的 `text` 字段
生成（参数见 [Offline 数据准备指南](../../../docs/guide/data/offline_preparation_guide.md)），输出为
`/path/to/dataset/deepseek32_text_document.bin/.idx`：

```bash
python -m hyper_parallel.data.tools.offline_preparation \
    --dataset-name-or-path /path/to/train.jsonl \
    --tokenizer-name-or-path /path/to/DeepSeek-V3.2 \
    --output-prefix /path/to/dataset/deepseek32
```

**3. 启动。** 第一个参数是裁剪 checkpoint 目录，第二个参数是 `.bin/.idx` 的公共前缀（不带后缀）：

```bash
bash examples/training_demo/deepseek_v32/run_single_card.sh /path/to/DeepSeek-V3.2-crop /path/to/dataset/deepseek32_text_document
bash examples/training_demo/deepseek_v32/run_8card.sh /path/to/DeepSeek-V3.2-crop /path/to/dataset/deepseek32_text_document
```

日志分别写到 `output/training_demo/deepseek_v32/run_single_card.log` 和 `run_8card.log`，末尾追加的
参数会覆盖 YAML，例如 `--training.train_iters=200`。

### 检查单卡与 8 卡自洽

两次运行从相同的状态出发，因此可以逐步比较：

- **初始权重相同**：两边加载同一份 checkpoint。不要改用 `from_config` 随机初始化：多进程时模型先在
  meta 设备上切分，再在各自的分片上随机初始化，单卡与 8 卡得到的初始权重不同。
- **输入相同**：每个 4096 窗口由 seed、split、序列长度和样本数唯一决定，每步的 2 个样本在两种拓扑下
  一样；8 卡时 dp 的两个 rank 各取 1 个样本，CP 再把序列切成两半。

比较两份日志中每步的 `training/foundation_loss` 与 `training/grad_norm`。并行只改变 BF16 计算的累加
顺序，第 1 步的差别在舍入量级，之后会随步数缓慢放大。
[模型接入验证指南](../../../docs/guide/trainer/model_integration_validation.md)的示例把跨拓扑门限设为
loss 绝对或相对误差不超过 5e-3、grad norm 相对误差不超过 5e-3（或绝对误差不超过 0.125），可以作为
参考。

### 与 MindSpeed-LLM 做 loss 对比

两边从同一份权重开始：HyperParallel 侧直接用单卡示例和上面生成的裁剪 checkpoint，MindSpeed 侧用
`ckpt_convert_deepseek32_hf2mcore.sh` 转换同一个目录，转换时的并行布局要与训练脚本一致。示例的数据
读取（同一份 `.bin/.idx`，seed 1234，split 100,0,0，每个窗口一条因果序列）和优化器设置（AdamW lr 1e-5、
cosine 到 1e-7、无 warmup、weight decay 0.01、betas (0.9, 0.999)、梯度裁剪 1.0）已经与 MindSpeed 脚本
一致，`global_batch_size` 和 `train_iters` 两边取相同的值。MindSpeed 脚本的 `index_topk` 是 1024，
HyperParallel 侧在启动命令末尾加 `--model.config_overrides='{index_topk: 1024}'`。其余设置这样统一：

| 项目 | HyperParallel | MindSpeed-LLM |
|---|---|---|
| MTP | 不训练 | 训练与权重转换脚本都去掉 `--mtp-num-layers`，训练脚本再去掉 `--mtp-loss-scaling-factor` |
| 负载均衡 loss | 无 | `--moe-router-load-balancing-type none`，去掉 `--moe-aux-loss-coeff` |
| 专家 bias | 用于选择，不更新 | 保留 `--moe-router-enable-expert-bias`，加 `--moe-router-bias-update-rate 0` |
| indexer KL | 无 | `--indexer-loss-coeff 0` |
| `index_topk` | `--model.config_overrides='{index_topk: 1024}'` | `--index-topk 1024` |
| Hadamard 旋转 | 不做 | 加 `--no-use-sparse-c8-indexer`，否则两边只差 BF16 舍入 |
| norm 权重精度 | 前向使用 BF16 参数 | 去掉 `--init-norm-weight-in-fp32`；它让 MindSpeed 的 `RMSNorm` 与 indexer `k_norm` 的权重保持 FP32 |
| 确定性 | `enable_full_determinism: true` | `--npu-deterministic` 与 `HCCL_DETERMINISTIC=true` |

MindSpeed 的 `lm loss` 对应 HyperParallel 的 `foundation_loss`。先比较第 1 步的 loss 再跑长程。

## 已知边界

- 不支持 PP：Trainer 目前不支持流水线并行。
- 不支持 sequence parallel：DSA 替换不支持 SP。
- 专家 bias 不更新、没有负载均衡 loss、不训练 MTP（与 Transformers 一致）。
- 验证范围是 2 层裁剪模型：CPU/Gloo 上的并行一致性见下表；NPU 上的融合算子与单卡、8 卡运行没有
  在本地验证。完整规格（61 层）和官方 FP8 权重的完整加载不在验证范围内。

## 测试

| 测试 | 覆盖内容 |
|---|---|
| `tests/ut/auto_models/models/deepseek_v32/test_deepseek_v32_adapter_contracts.py` | TP 角色（`linear_qkv` 与 indexer 复制、`q_b_proj` / `kv_b_proj` 列切）；两个示例的 EP 条目只命中 MoE 层、CP 条目作用于 attention；DSA 替换在稀疏选择和稠密选择下的输出与梯度对齐 Transformers attention；加载 checkpoint 时 RoPE buffer 与 CPU 构建一致 |
| `tests/torch/auto_models/deepseek_v32/test_deepseek_v32_parallel.py` | CPU/Gloo，经 Trainer 链路（`plan_overrides` → 模块替换 → `ShardingPlanner` → `apply_sharding_plan` → FSDP2）：CP2、TP2 + EP2 + loss parallel、TP2 × CP2 × EP4，loss 与全部梯度对齐单进程模型 |
| `tests/common/dsa_reference_kernels.py` | CPU 参考 DSA kernel（替代 CANN 入口），供上面两个测试使用 |
