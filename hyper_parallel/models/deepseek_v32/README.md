# DeepSeek-V3.2 模型族适配

DeepSeek-V3.2 在 DeepSeek-V3 的 MLA + MoE 结构上加入 DeepSeek Sparse Attention（DSA）：每层由
lightning indexer 为每个 query 选出 `index_topk` 个 key（官方 config 为 2048），MLA 只在这些 key
上计算 attention。模型主体直接使用 Transformers 原生的 `DeepseekV32ForCausalLM`，本族不带建模代码；
本目录是 HyperParallel 的适配层，包括模块替换工厂、CP wrapper、TP 分片规则、RoPE 物化、拼行文档
边界和训练 recipe。

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
│   ├── materialization.py           # meta 初始化后重建 RoPE 的 inv_freq / original_inv_freq
│   ├── conversion/
│   │   └── module_replacement.py    # RMSNorm、DSA attention、GroupedExperts 的替换工厂
│   ├── distributed/
│   │   └── context_parallel.py      # DSA 的 CP wrapper
│   ├── data/
│   │   └── runtime.py               # DeepseekV32PackedSequence：把拼行文档边界传给 DSA
│   └── policies/
│       └── sharding.py              # MLA 与 indexer 的 TP 分片规则
├── generated/                       # codegen 生成的模型代码，recipe 不使用
└── recipes/
    └── train.yaml                   # 512 卡主线配方
```

单卡裁剪示例在 `examples/training_demo/deepseek_v32/`（`train_single_card.yaml` 与
`run_single_card.sh`）。

## 训练语义

本族不修改 Transformers 的模型代码，模块替换和并行化都保持计算等价，因此训练语义就是
Transformers 实现的语义。它和 MindSpeed-LLM 的 DeepSeek-V3.2 预训练脚本
（`examples/mcore/deepseek32/pretrain_deepseek32_671b_4k_A3_ptd.sh`）有以下差别：

| 项目 | 本族（与 Transformers 一致） | MindSpeed-LLM 预训练脚本 |
|---|---|---|
| 专家选择 | sigmoid 分数加 `e_score_correction_bias` 做分组 top-k（`n_group` 8、`topk_group` 4、top-8），门控值归一化后乘 `routed_scaling_factor` | 相同 |
| 专家 bias | FP32 buffer，训练中保持 checkpoint 中的值 | `--moe-router-enable-expert-bias`，每步按专家负载 ±1e-3 更新 |
| 负载均衡 loss | 无 | `--moe-router-load-balancing-type seq_aux_loss`，系数 1e-4 |
| MTP | Transformers 加载时忽略第 61 层（`_keys_to_ignore_on_load_unexpected`），不训练 | `--mtp-num-layers 1`，loss 系数 0.3 |
| indexer KL loss | 默认关闭，indexer 参数不更新；可以开启，见[开启 indexer KL loss](#开启-indexer-kl-loss) | `--indexer-loss-coeff 1.0` |
| indexer Hadamard 旋转 | 不做 | 默认在打分前对 q/k 做 Hadamard 旋转，`--no-use-sparse-c8-indexer` 关闭 |

几点说明：

- **专家 bias 与负载均衡、MTP**：这些是预训练目标，Transformers 实现只包含推理所需的部分。
  DeepSeek 官方推理代码（`DeepSeek-V3.2-Exp/inference/convert.py`）同样跳过第 61 层。需要这些
  目标时要在模型侧新增代码，本族不包含。
- **indexer**：indexer 的输出只用于 top-k 选择，这一步不可导，所以语言模型 loss 不会给 indexer
  梯度。DSA 替换中 indexer 的输入是 detach 的，开启 KL loss 后它只更新 indexer，不影响主干。
- **Hadamard 旋转**：官方推理代码在 indexer 打分前对 q/k 乘归一化的 Hadamard 矩阵，配合 FP8
  量化打分。它是正交变换，不改变 q·k，Transformers 按 BF16 计算时跳过了这一步（见其
  `DeepseekV32Indexer` 的注释）。两种做法只差 BF16 舍入，在 top-k 边界上可能选出不同的 key。

## 适配内容

- **模块替换**（`adapter/conversion/module_replacement.py`，由 recipe 的 `plan_overrides` 选用）：

  | 匹配 | 替换为 | 说明 |
  |---|---|---|
  | `*.input_layernorm`、`*.post_attention_layernorm`、`model.norm` | 框架 `RMSNorm` | NPU 融合 RMSNorm |
  | `*.self_attn` | 框架 `DeepseekV32DSAAttention` | `q_a_proj` 与 `kv_a_proj_with_mqa` 合并为 `linear_qkv`，加载时由替换模块声明的转换拼接权重；indexer 与稀疏 attention 使用 CANN 融合算子；可选的 indexer KL loss |
  | `*.mlp.experts` | 框架 `GroupedExperts` | grouped GEMM |

  DSA 替换支持因果训练和拼行文档，不支持 padding mask、dropout 和 KV cache。DSA 算子的依赖见
  [functional README](../../components/functional/README.md#deepseek-v32-dsa)。
- **CP**（`adapter/distributed/context_parallel.py`，recipe 中 `when: cp` 的条目）：query 留在本 rank，
  异步 all-gather indexer key 与 MLA 压缩 KV，每个 rank 只使用到本地 query 结尾的因果前缀；拼行
  文档边界从全局 `cu_seq_lens` 投影到本地。开启 indexer KL loss 时，CP 下同样计算。
- **EP**（recipe 中 `when: ep` 的条目）：直接使用框架内置的 `deepseekv3_ep_compute_fn`
  （`deepseekv3_sigmoid_group_shared` 原型）。它的路由与 Transformers router 相同；token 分发、
  本地 grouped GEMM 和合并由框架 EP 骨架完成，shared experts 与分发通信重叠计算。条目只匹配
  MoE 层：官方 config 的 `first_k_dense_replace` 为 3，层 0-2 是 dense MLP，fnmatch 不支持数值区间，
  所以用 `model.layers.[3-9].mlp.forward` 和 `model.layers.[1-9][0-9].mlp.forward` 两条 glob 覆盖
  层 3-60；改变 dense 层数时要同步修改。
- **TP 分片规则**（`adapter/policies/sharding.py`）：`linear_qkv`（低秩下投影）复制，`q_b_proj` /
  `kv_b_proj` 按 head 列切，indexer 的 `wq_b` / `wk` / `weights_proj` 复制（indexer 头不切分）。
  MoE 层在 TP 下按 TP-extend-EP 处理，专家只按 EP 切分，所以 `tp_size > 1` 时 `ep_size` 也要大于 1。
- **RoPE 物化**（`adapter/materialization.py`）：Transformers 的 `inv_freq` / `original_inv_freq` 是
  非持久 buffer，checkpoint 里没有。模型在 meta 设备上构建、再用 `to_empty` 分配存储后，这两个
  buffer 是未初始化的内存。materialization provider 按同一个 config 在 CPU 上算出 YaRN 频率，登记为
  可重建 buffer，构建流程在物化后写回（与 Qwen3.5 的做法相同）。
- **拼行文档边界**（`adapter/data/runtime.py`）：DSA 只检查 `attention_mask` 的形状，不读内容。
  `DeepseekV32PackedSequence` 把 batch 中的全局 `cu_seq_lens` 作为 `cu_seq_lens_q` / `cu_seq_lens_k`
  传给 DSA，文档之间互不可见。不配置它时，每行是一条完整的因果序列。

## 依赖与权重

- Transformers 需要包含 `deepseek_v32`（`requirements.txt` 要求 `transformers>=5.11.0`，本地测试使用
  5.20.2）。RMSNorm、GroupedExperts 和 DSA 替换只能在 Ascend NPU 上运行；DSA 依赖 CANN 与
  `cann_ops_transformer`，版本见上面的 functional README。
- `from_pretrained` 可以直接指向官方 checkpoint。官方权重是 FP8（`quant_method: fp8`，128×128 block，
  `scale_fmt: ue8m0`），默认的 DCP loader 在读取时按 block 反量化后再转换成训练参数
  （`hyper_parallel/components/checkpoint/quantized_hf_storage.py`）。第 61 层 MTP 的权重不属于模型，
  加载时会被忽略并打印 warning。

## 使用

### 单卡裁剪示例

`examples/training_demo/deepseek_v32/train_single_card.yaml` 用 `HyperAutoModelForCausalLM.from_config`
读取官方 `config.json`，随机初始化，不加载权重。它保留官方的 hidden、MLA、indexer 和词表维度，只把
模型裁成 2 层（第 0 层 dense、第 1 层 MoE）和 16 个路由专家；16 是保持官方分组路由的最小专家数。
共 3.40B 参数，FP32 参数、FP32 梯度和 AdamW 两个 FP32 动量约 51 GiB（估算，不含激活），面向一张
64 GB 的 NPU（如 910B3）。

```bash
bash examples/training_demo/deepseek_v32/run_single_card.sh \
    /path/to/DeepSeek-V3.2 /path/to/train.jsonl
```

第一个参数是官方模型目录，只读取其中的 `config.json` 和 tokenizer 文件；第二个参数是文本 JSONL，
每行一个 JSON，正文在 `text` 字段。示例把多个文档拼成 4096 token 的行，文档边界由
`DeepseekV32PackedSequence` 传给 DSA。更多说明见
[examples/training_demo/README.md](../../../examples/training_demo/README.md#deepseek-v32-single-card-crop)。

### 512 卡主线配方

`recipes/train.yaml` 面向 64 节点 / 512 卡：dense 区域 `tp 1 × cp 1 × dp 512`，FSDP 在 512 卡上切分；
专家区域 `ep 64 × edp 8`，每个 EP rank 持有 4 个专家，edp 内按 `edp_shard 4 × 副本 2` 做 HSDP，每张卡
每个 MoE 层恰好 1 个专家。按 FP32 参数、梯度和两个 AdamW 动量约 16 B/param 估算，专家部分每卡约
2.55B 参数（约 41 GB），dense 部分约 18B 参数切到 512 卡后每卡约 0.6 GB，激活另计。默认数据是
wikitext-2（每行一个样本），只用于跑通流程。每个节点执行：

```bash
torchrun --nnodes=64 --nproc_per_node=8 --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} --master_port=6000 \
    scripts/train_lm.py hyper_parallel/models/deepseek_v32/recipes/train.yaml \
    --model.pretrained_model_name_or_path=/path/to/DeepSeek-V3.2 \
    --dataset.model_assets.tokenizer.pretrained_model_name_or_path=/path/to/DeepSeek-V3.2
```

其它并行维度用命令行覆盖打开：

- TP：`--accelerator.tp_size=2 --accelerator.loss_parallel=true --fsdp_config.dp_shard_size=256`
  （dp 随 TP 减半）。
- CP：`--accelerator.cp_size=2`，`when: cp` 的条目自动生效。
- EP：`--accelerator.ep_size` 与 `--fsdp_config.edp_shard_size` 一起调整。

修改拓扑时保持：`world_size = tp × cp × dp`；`dp × cp` 能被 `dp_shard_size` 整除；
`dp × cp × tp / ep` 能被 `edp_shard_size` 整除；256 个专家能被 `ep_size` 整除；`256 / ep_size` 能被
`edp_shard_size` 整除（否则 FSDP 按第 0 维切专家时会补空数据）；`global_batch_size` 能被
`micro_batch_size × dp` 整除。

### 开启 indexer KL loss

在 DSA 替换条目上设置 `dsa_loss_coeff`：

```yaml
plan_overrides:
  - match: "*.self_attn"
    replace:
      kind: module
      implementation:
        _target_: >-
          hyper_parallel.models.deepseek_v32.adapter.conversion.module_replacement.replace_deepseek_v32_dsa_attention
        dsa_loss_coeff: 1.0
```

- KL loss 让 indexer 的打分分布逼近主 attention 在所有 head 上求和后的分布。TP 会把 head 切到不同
  rank，而 KL 融合算子不做跨 rank 归约，所以 `tp_size > 1` 时开启会在构建阶段报错。
- KL 只有 CANN 实现。
- Transformers 的 config 不认识 `dsa_loss_coeff`，`config_overrides` 和 `AutoConfig` 的同名参数都会被
  忽略，所以要写在替换条目上。

### 与 MindSpeed-LLM 做 loss 对比

两边需要使用同一份权重和数据，并统一下面的设置：

| 项目 | HyperParallel | MindSpeed-LLM |
|---|---|---|
| MTP | 不训练 | 训练与权重转换脚本都去掉 `--mtp-num-layers`，训练脚本再去掉 `--mtp-loss-scaling-factor` |
| 负载均衡 loss | 无 | `--moe-router-load-balancing-type none`，去掉 `--moe-aux-loss-coeff` |
| 专家 bias | 用于选择，不更新 | 保留 `--moe-router-enable-expert-bias`，加 `--moe-router-bias-update-rate 0` |
| indexer KL | 替换条目设 `dsa_loss_coeff: 1.0`（`tp_size=1`） | `--indexer-loss-coeff 1.0` |
| `index_topk` | `model.config_overrides: {index_topk: 1024}` | `--index-topk 1024` |
| Hadamard 旋转 | 不做 | 加 `--no-use-sparse-c8-indexer`，否则两边只差 BF16 舍入 |
| norm 权重精度 | 前向使用 BF16 参数 | 去掉 `--init-norm-weight-in-fp32`；它让 MindSpeed 的 `RMSNorm` 与 indexer `k_norm` 的权重保持 FP32 |
| 优化器 | `lr_scheduler.lr_config.lr_warmup_steps: 0`、`min_lr: 1.0e-7`，其余同 recipe（AdamW lr 1e-5、betas (0.9, 0.999)、weight decay 0.01、梯度裁剪 1.0、cosine） | 脚本原值：`--lr 1.0e-5`、`--min-lr 1.0e-7`、`--lr-decay-style cosine`、`--lr-warmup-iters 0`、`--weight-decay 1e-2`、`--adam-beta2 0.999`、`--clip-grad 1.0` |
| 随机种子与确定性 | `training.seed: 1234`，`enable_full_determinism: true` | 默认 seed 1234，`--npu-deterministic` 与 `HCCL_DETERMINISTIC=true` |

- **权重**：官方 FP8 权重先反量化成 BF16 一次，两边都使用这一份。MindSpeed 的
  `ckpt_convert_deepseek32_hf2mcore.sh` 不做反量化；转换时的并行布局要与训练脚本一致。
- **数据**：用 MindSpeed 的 `data_convert_deepseek32_pretrain.sh` 生成 Megatron `.bin/.idx`，两边读同一个
  前缀。HyperParallel 侧把 recipe 的 `dataset` 与 `dataloader` 两节换成：

  ```yaml
  dataset:
    model_assets:
      chat_template: null
      tokenizer:
        _target_: hyper_parallel.data.text.build_tokenizer.AutoTokenizer.from_pretrained
        pretrained_model_name_or_path: /path/to/DeepSeek-V3.2
        tokenizer_type: pretokenized
        vocab_size: 129280
        eod_token_id: 1
    _target_: hyper_parallel.data.text.build_dataset.build_indexed_text_dataset
    data_path: /path/to/dataset/deepseek32_text_document   # 不带 .bin/.idx 后缀
    data_config:
      seq_length: 4096
      split: "100,0,0"
      low_level_dataset:
        _target_: hyper_parallel.data.indexed.indexed_pretrain_dataset.GPTDataset
      simple_blend: "no"
      skip_data_check: true
      labels_are_shifted: true
      create_attention_mask_in_dataloader: false
      create_ltor_fields_in_dataloader: false

  dataloader:
    _target_: hyper_parallel.data.batching.FixedBatchDataLoader
    collate_fn:
      _target_: hyper_parallel.data.batching.build_indexed_collate_fn
    get_batch:
      _target_: hyper_parallel.data.batching.TextParallelBatch
      source_type: indexed
      attention_mode: compressed
    sampler_type: single
    data_sharding: false
    drop_last: true
    num_workers: 0
  ```

  不配置 `runtime_input_adapter`，每个 4096 窗口是一条完整的因果序列，对应 MindSpeed 不开
  `--reset-attention-mask`。两边由 seed、split、序列长度和总样本数（`train_iters × global_batch_size`）
  决定样本顺序，这几项要一致；先比较第 1 步的 loss 再跑长程。
- **日志**：MindSpeed 的 `lm loss` 对应 HyperParallel 的 `foundation_loss`。

## 已知边界

- 不支持 PP：Trainer 目前不支持流水线并行，recipe 保持 `pp_size: 1`。
- 不支持 sequence parallel：DSA 替换不支持 SP，recipe 保持 `sequence_parallel: false`。
- indexer KL loss 要求 `tp_size=1`，且只有 CANN 实现。
- 专家 bias 不更新、没有负载均衡 loss、不训练 MTP（与 Transformers 一致）。
- 本目录的代码在 CPU/Gloo 上验证（见下表）；NPU 上的融合算子、FSDP 分片、官方 FP8 权重的完整加载、
  512 卡显存与端到端训练没有在本地验证。

## 测试

| 测试 | 覆盖内容 |
|---|---|
| `tests/ut/auto_models/models/deepseek_v32/test_deepseek_v32_adapter_contracts.py` | TP 角色（`linear_qkv` 与 indexer 复制、`q_b_proj` / `kv_b_proj` 列切）；recipe 的 EP glob 只命中 MoE 层；DSA 替换在稀疏选择、稠密选择和拼行文档下的输出与梯度对齐 Transformers attention；KL loss 与 TP 互斥；meta 构建后 RoPE buffer 与 CPU 构建一致 |
| `tests/torch/auto_models/deepseek_v32/test_deepseek_v32_parallel.py` | CPU/Gloo，经 Trainer 链路（`plan_overrides` → 模块替换 → `ShardingPlanner` → `apply_sharding_plan` → FSDP2）：CP2、CP2 + 跨 CP 切分点的拼行文档、TP2 + EP2 + loss parallel、TP2 × CP2 × EP4，loss 与全部梯度对齐单进程模型 |
| `tests/common/dsa_reference_kernels.py` | CPU 参考 DSA kernel（替代 CANN 入口），供上面两个测试使用 |
