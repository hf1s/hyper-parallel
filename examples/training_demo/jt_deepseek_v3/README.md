# 使用已分词的 packed SFT 数据

本适配保留 `input_ids`、已移位的 `labels`（忽略值为 `-100`）和 `cu_seqlens`。
不重新分词，不替换 token ID，不改写模型的词表或初始权重。词表大小和权重必须覆盖数据的 token ID。

## 离线转换

断网交付和反复训练建议预先转换，需要 PyArrow；此路径训练只读取 `.bin/.idx`。
转换实现在公共 `data/tools`，示例目录的同名脚本只保留兼容入口。先解开数据包，然后执行：

```bash
# 保留原始 256K 记录
python -m hyper_parallel.data.tools.prepare_packed_sft \
  --input /path/to/sft_data_256k_demo/data-00000-of-00001.arrow \
  --output-prefix /path/to/sft-256k/train

# 生成不重叠的 4K 窗口
python -m hyper_parallel.data.tools.prepare_packed_sft \
  --input /path/to/sft_data_256k_demo/data-00000-of-00001.arrow \
  --output-prefix /path/to/sft-4k/train --sequence-length 4096
```

每个前缀生成 `tokens`、`labels`、`loss_mask`、`cu_seqlens` 四组文件。
4K 窗口保留窗口内文档边界；没有有效标签的窗口跳过。原始长度须能整除窗口长度。
这份最小转换脚本使用不重叠窗口，不重复扩充样本。

## 修改已有 JT YAML

保留原生 `build_jt_model` 和 `IndexedSupervisedDataset`。在已有配置中增加以下字段：

```yaml
dataset:
  _target_: hyper_parallel.data.indexed.indexed_supervised_dataset.IndexedSupervisedDataset
  packed: true
  data_path: /path/to/sft-4k/train
  sequence_length: 4096

dataloader:
  get_batch:
    _target_: hyper_parallel.data.batching.TextParallelBatch
    source_type: indexed
    attention_mode: compressed
    reset_position_ids: true
    runtime_input_adapter:
      _target_: hyper_parallel.models.jt_deepseek_v3.adapter.data.runtime.JTPackedRuntime
```

保持 `micro_batch_size: 1`、`cp_size: 1`；DP/TP/EP 沿用已有配置。
切换 256K 时同时修改数据前缀和 `sequence_length: 262144`。
原生 Trainer 启动方式不变，短时验证设置 `training.train_iters: 10`。

修改所选 YAML 后，传入权重目录和数据文件公共前缀（前缀本身无需是文件）：

```bash
JT_RECIPE_NAME=jt_deepseek_v3.tp4_ep8_dp2.yaml \
bash examples/training_demo/jt_deepseek_v3/run_jt_deepseek_v3.sh \
  /path/to/initial_weights /path/to/sft-4k/train --training.train_iters=10
```

日志位于 `output/training_demo/jt_deepseek_v3/run_<配置名>.log`。

大词表长序列训练可在模型配置中启用输出层分块：

```yaml
model:
  config:
    loss_chunk_size: 16384
```

默认值 `0` 保留完整 logits 路径。正整数表示每次投影的全局序列长度上限；
开启 sequence parallel 时须为 TP 度数的整数倍。LM 和每层 MTP 共用同一实现，
投影及 CE 一起重计算，避免保存完整 `[序列长度, 词表大小]` 激活。
分块大小仅影响显存和速度，不改变文档边界、预移位标签或有效 token 的归一化。
256K 配合大词表时建议启用；仅使用基础数据适配并不保证完整 logits 能放入单卡显存。

## 直接读取原始数据（可选）

无需预先转 `.bin/.idx`。此路径需要 `datasets`；保留模型配置，将 Dataset/DataLoader 替换为以下公共组件：

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
    _target_: hyper_parallel.data.batching.TextParallelBatch
    source_type: online
    attention_mode: compressed
    reset_position_ids: true
    runtime_input_adapter:
      _target_: hyper_parallel.models.jt_deepseek_v3.adapter.data.runtime.JTPackedRuntime
```

256K 改为 `max_seq_len: 262144`。原始记录必须提供已移位的 `labels`；不会再次移位。
原有 Online source 支持的本地 Arrow/Parquet/JSONL 均可复用，Online 指运行期读取，与联网无关。
长记录经 transform 产生多个窗口，`TokenBatchLoader` 按 token budget 取样，并保存未消费窗口用于恢复；
Online 保留较短尾窗，由 collator 补并行对齐 padding。Indexed 定长导出仍要求记录长度能整除窗口长度。
两种入口保持窗口的 token、标签和文档边界一致，不保证逐步采样顺序一致；恢复训练须保持 DP 度数和全局 batch size 不变。

直接读取时使用通用训练入口（上面的 indexed bash 仅用于 indexed 数据）：

```bash
torchrun --standalone --nproc_per_node=4 --module examples.training_demo.train_text \
  /path/to/jt_online.yaml --training.train_iters=10
```

并行度须与 YAML 匹配。两种入口复用同一个 `PreTokenizedSFTTransform` 校验和切窗逻辑，
均经 `cu_seq_lens` 接入 `JTPackedRuntime`；collator 会保留每个窗口内部的文档边界。
Trainer 沿用现有 Dataset/transform/DataLoader 配置组装、DP 采样及 TP 广播。原 JT dataset 导入路径保留兼容。
