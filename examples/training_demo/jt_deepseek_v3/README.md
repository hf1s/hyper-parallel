# JT packed SFT

输入为预分词 `input_ids`、等长且已移位的 `labels`（忽略值 `-100`），以及可选的累计文档边界
`cu_seqlens`（包含 0 和记录总长）。不会重新分词或再次移位标签；词表与初始权重须匹配数据。
先激活已安装的 CANN / PyTorch / PTA 环境，并从仓库根目录执行以下命令。

## 1. 准备数据

离线转换需要 `pyarrow`，训练只读取生成的 bin/idx；断网环境须提前安装依赖。

```bash
# 原始 256K；生成 4K 时增加 --sequence-length 4096 并使用另一个输出前缀
python -m hyper_parallel.data.tools.prepare_packed_sft \
  --input /path/to/sft_data_256k_demo/data-00000-of-00001.arrow \
  --output-prefix /path/to/sft-256k/train
```

每个前缀生成 `tokens`、`labels`、`loss_mask`、`cu_seqlens` 四组 bin/idx。
切窗不重叠，保留窗口内边界并跳过全忽略标签的窗口；定长导出要求整除。工具拒绝覆盖已有输出。

## 2. 修改已有 YAML

复制本目录的 JT YAML，保留模型、优化器和并行策略，合并以下字段：

```yaml
dataset:
  _target_: hyper_parallel.data.indexed.indexed_supervised_dataset.IndexedSupervisedDataset
  data_path: /path/to/sft-4k/train
  sequence_length: 4096
  packed: true

dataloader:
  get_batch:
    _target_: hyper_parallel.data.batching.TextParallelBatch
    source_type: indexed
    attention_mode: compressed
    reset_position_ids: true
    runtime_input_adapter:
      _target_: hyper_parallel.models.jt_deepseek_v3.adapter.data.runtime.JTPackedRuntime
```

- 权重：`model.reference_weights`；模型词表：`model.config.vocab_size`，须覆盖所有 token ID。
- 数据：`dataset.data_path` 填公共前缀；256K 同时改为对应前缀和 `sequence_length: 262144`。
- packed 路径要求 `training.micro_batch_size: 1`、`accelerator.cp_size: 1`。
- 验证步数：`training.train_iters: 10`；卡数须与 YAML 中的并行策略匹配。

大词表训练可设置 `model.config.loss_chunk_size`：4K 验证使用 1024，256K 使用 16384。
默认 0 关闭分块；启用 SP 时须为 TP 度数的整数倍。LM/MTP 共用投影与 CE 的分块重计算，
以额外计算换取更低的输出层显存；公共 CE 的同步优化独立于此开关，不承诺所有路径无 host sync。

## 3. 启动与日志

通用入口适用于任意 YAML；以下 4 卡命令要求使用相应的并行配置：

```bash
mkdir -p /workspace/logs
set -o pipefail
torchrun --standalone --nproc_per_node=4 --module examples.training_demo.train_text \
  /path/to/jt_sft.yaml --training.train_iters=10 \
  2>&1 | tee /workspace/logs/jt-sft.log
```

也可用本目录的 `run_jt_deepseek_v3.sh` 启动 **8 卡 indexed 训练**：
`JT_RECIPE_NAME=<本目录下的 YAML 文件名> bash examples/training_demo/jt_deepseek_v3/run_jt_deepseek_v3.sh <权重路径> <数据前缀> --training.train_iters=10`。
该脚本日志为 `output/training_demo/jt_deepseek_v3/run_<配置名>.log`；`tee` 覆盖同名日志。

## 4. 可选：直接读取原始 Arrow

此路径需要 `datasets`，Online 指运行时转换，与联网无关。用以下内容替换整个 `dataset` 段，
合并 `dataloader` 字段，保留 `get_batch` 其他字段及模型/并行配置；启动和日志同上。

```yaml
dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_online_text_mapping_dataset
  data_path: /path/to/sft_data_256k_demo/data-00000-of-00001.arrow
  data_config: {}
  model_assets:
    tokenizer: null
  data_transform:
    _target_: hyper_parallel.data.text.pretokenized_sft.PreTokenizedSFTTransform
    max_seq_len: 4096

dataloader:
  _target_: hyper_parallel.data.batching.TokenBatchLoader
  min_buffered_samples: 1
  num_workers: 0
  collate_fn:
    _target_: hyper_parallel.data.batching.build_online_text_collate_fn
  get_batch:
    source_type: online
```

切换 256K 使用 `max_seq_len: 262144`。Online 保留短尾窗，由 collator 补齐；两种入口共用
转换逻辑，但不保证逐步采样顺序一致。恢复训练须保持 DP 度数和 global batch size 不变。
更多字段映射和数据接口见 [公共数据说明](../../../hyper_parallel/data/README.md#预分词-sft)。
