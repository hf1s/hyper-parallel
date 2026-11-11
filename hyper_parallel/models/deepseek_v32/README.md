# DeepSeek-V3.2 Model-Family Integration

**English** | [简体中文](README.zh.md)

DeepSeek-V3.2 adds DeepSeek Sparse Attention (DSA) to the DeepSeek-V3 MLA + MoE architecture: in every layer a
lightning indexer selects `index_topk` keys for each query (2048 in the released config), and MLA attends only to those
keys. The model body is the native Transformers `DeepseekV32ForCausalLM`; this family ships no modeling code. This
directory holds HyperParallel's integration layer: module replacement factories, the CP wrapper, TP sharding rules and
the RoPE materialization.

## Registration Identity

| Item | Value |
|---|---|
| HyperParallel family name (`models/<family>/` directory) | `deepseek_v32` |
| HF `config.model_type` | `deepseek_v32` |
| HF `architectures` | `DeepseekV32ForCausalLM` |
| Implementation | Native Transformers (`transformers.models.deepseek_v32`); no `trust_remote_code` |

## Directory Layout

```text
models/deepseek_v32/
├── __init__.py
├── adapter/
│   ├── registration.py              # ModelAdapterSpec: replacements, CP, TP sharding rules, RoPE materialization
│   ├── materialization.py           # rebuilds RoPE inv_freq / original_inv_freq when weights are loaded
│   ├── conversion/
│   │   └── module_replacement.py    # replacement factories for RMSNorm, DSA attention, GroupedExperts
│   ├── distributed/
│   │   └── context_parallel.py      # DSA CP wrapper
│   └── policies/
│       └── sharding.py              # TP sharding rules for MLA and the indexer
├── generated/                       # codegen model sources, not used by the example
└── recipes/
    └── train.yaml                   # full-size single-process recipe (already on master, outside this validation)

examples/training_demo/deepseek_v32/
├── train_single_card.yaml           # cropped model, single-device topology; 8 devices override the topology
├── run_single_card.sh
└── run_8card.sh                     # tp 2 × cp 2 × dp 2, ep 4 × edp 2, loss parallel
```

## Training Semantics

This family does not modify the Transformers modeling code, and the module replacements and parallelization keep the
computation equivalent, so training follows the semantics of the Transformers implementation. It differs from the
MindSpeed-LLM DeepSeek-V3.2 pretraining script (`examples/mcore/deepseek32/pretrain_deepseek32_671b_4k_A3_ptd.sh`) as
follows:

| Item | This family (as in Transformers) | MindSpeed-LLM pretraining script |
|---|---|---|
| Expert selection | sigmoid scores plus `e_score_correction_bias` for group-limited top-k (`n_group` 8, `topk_group` 4, top-8); gates normalized and multiplied by `routed_scaling_factor` | Same |
| Expert bias | FP32 buffer that keeps its initial value during training | `--moe-router-enable-expert-bias`, updated by ±1e-3 per step from the expert load |
| Load-balancing loss | None | `--moe-router-load-balancing-type seq_aux_loss`, coefficient 1e-4 |
| MTP | Transformers ignores layer 61 on load (`_keys_to_ignore_on_load_unexpected`); not trained | `--mtp-num-layers 1`, loss coefficient 0.3 |
| Indexer KL loss | None; the indexer parameters are not updated | `--indexer-loss-coeff 1.0` |
| Indexer Hadamard rotation | Not applied | Applied to q/k before scoring by default; `--no-use-sparse-c8-indexer` disables it |

Notes:

- **Expert bias, load balancing and MTP**: these are pretraining objectives; the Transformers implementation contains
  only what inference needs. DeepSeek's official inference code (`DeepSeek-V3.2-Exp/inference/convert.py`) also skips
  layer 61. Training them requires additional model code, which this family does not include.
- **Indexer**: the indexer output is used only for the non-differentiable top-k selection, so the language-model loss
  gives the indexer no gradient and it keeps its checkpoint values.
- **Hadamard rotation**: the official inference code multiplies q/k by a normalized Hadamard matrix before scoring, for
  FP8-quantized scoring. The transform is orthogonal and leaves q·k unchanged, so Transformers skips it when computing in
  BF16 (see the comment in its `DeepseekV32Indexer`). The two differ only by BF16 rounding, which can select different
  keys at the top-k boundary.

## Integration Contents

- **Module replacements** (`adapter/conversion/module_replacement.py`, selected by the YAML `plan_overrides`):

  | Match | Replaced by | Notes |
  |---|---|---|
  | `*.input_layernorm`, `*.post_attention_layernorm`, `model.norm` | framework `RMSNorm` | NPU fused RMSNorm |
  | `*.self_attn` | framework `DeepseekV32DSAAttention` | `q_a_proj` and `kv_a_proj_with_mqa` merge into `linear_qkv`, concatenated on load by the transform the replacement declares; the indexer and sparse attention use CANN fused operators |
  | `*.mlp.experts` | framework `GroupedExperts` | grouped GEMM |

  The DSA replacement computes causal attention only: it does not read the contents of `attention_mask` and supports no
  padding mask, dropout or KV cache. Each data row is one complete causal sequence, as in Megatron/MindSpeed
  pretraining data by default (without `--reset-attention-mask`). See the
  [functional README](../../components/functional/README.md#deepseek-v32-dsa) for the DSA operator dependencies.
- **CP** (`adapter/distributed/context_parallel.py`, the `when: cp` entry in the YAML, active when `cp_size > 1`): the
  query stays on its rank while the indexer keys and the compressed MLA KV are all-gathered asynchronously; each rank
  uses only the causal prefix that ends with its local queries.
- **EP** (the `when: ep` entry in the YAML, active when `ep_size > 1`): uses the framework's built-in
  `deepseekv3_ep_compute_fn` (the `deepseekv3_sigmoid_group_shared` archetype). It routes like the Transformers router;
  the framework EP skeleton dispatches tokens, runs the local grouped GEMM and combines the results, and the shared
  experts overlap the dispatch communication. The entry must match MoE layers only: the cropped model has just layer 1,
  written as `model.layers.1.mlp.forward`; update it when the depth or `first_k_dense_replace` changes.
- **TP sharding rules** (`adapter/policies/sharding.py`): `linear_qkv` (the low-rank down projection) is replicated,
  `q_b_proj` / `kv_b_proj` are column-sharded by head, and the indexer `wq_b` / `wk` / `weights_proj` are replicated (the
  indexer heads are not sharded). MoE layers use TP-extend-EP under TP and shard experts only by EP, so `ep_size` must be
  greater than 1 when `tp_size > 1`.
- **RoPE materialization** (`adapter/materialization.py`): Transformers' `inv_freq` / `original_inv_freq` are
  non-persistent buffers that checkpoints do not contain. `from_pretrained` builds the model on the meta device,
  allocates storage and loads the checkpoint without running the Transformers weight initialization, so these two
  buffers stay as uninitialized memory (NaN in our measurement, which turns the loss into NaN). The materialization
  provider computes the YaRN frequencies on the CPU from the same config and registers them as rebuildable buffers,
  which the build writes back after loading (as Qwen3.5 does). It is needed whenever weights are loaded (released
  weights or the cropped checkpoint of the example); with `from_config` random initialization, the Transformers
  initialization computes these buffers itself.

## Dependencies and Weights

- Transformers must include `deepseek_v32` (`requirements.txt` requires `transformers>=5.11.0`; validated on NPU with
  5.13.1). The RMSNorm, GroupedExperts and DSA replacements run only on Ascend NPU; DSA depends on CANN and
  `cann_ops_transformer`, with versions listed in the functional README above.
- `from_pretrained` can point directly at the released checkpoint. The released weights are FP8 (`quant_method: fp8`,
  128×128 blocks, `scale_fmt: ue8m0`); the default DCP loader dequantizes them block by block while reading, before
  converting them into training parameters (`hyper_parallel/components/checkpoint/quantized_hf_storage.py`). The
  layer-61 MTP weights do not belong to the model and are ignored with a warning on load.

## Usage

### Single-device and 8-device cropped example

The example loads a cropped BF16 checkpoint. It keeps the released hidden, MLA, indexer and vocabulary dimensions with
only 2 layers (layer 0 dense, layer 1 MoE) and 16 routed experts, the fewest that keep the released group routing.
The model has 3.40B parameters; FP32 parameters, FP32 gradients and the two FP32 AdamW moments take about 51 GiB
(estimated, without activations), and a single device swaps the optimizer states to host memory with
`optimizer.swap`.

Both runs use `train_single_card.yaml`: `run_single_card.sh` runs its single-device topology, and `run_8card.sh`
overrides it on the command line to `tp 2 × cp 2 × dp 2` (FSDP over dp) and `ep 4 × edp 2` for the MoE layer (4
experts per EP rank) with loss parallelism. Every other setting is identical: `seed: 1234`, `global_batch_size: 2`,
sequence length 4096, AdamW lr 1e-5 with cosine decay to 1e-7 and no warmup.

**1. Build the cropped checkpoint (once).** Crop the released config with Transformers, initialize randomly and save.
Only `config.json` and the tokenizer files of the released directory are needed:

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

source, target = "/path/to/DeepSeek-V3.2", "/path/to/DeepSeek-V3.2-crop"
released = AutoConfig.from_pretrained(source)
config = AutoConfig.from_pretrained(
    source, num_hidden_layers=2, first_k_dense_replace=1, n_routed_experts=16,
    # The per-layer types are derived from the 61 layers read from the config and are not re-derived after the depth
    # override, so take one dense and one MoE layer from the released lists. The type names differ between
    # Transformers versions (e.g. deepseek_sparse_attention / indexed_attention); do not hard-code them.
    mlp_layer_types=[released.mlp_layer_types[0], released.mlp_layer_types[-1]],
    layer_types=released.layer_types[:2],
)
del config.quantization_config  # the random weights are BF16, not the released FP8
torch.manual_seed(1234)
AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).save_pretrained(target)
AutoTokenizer.from_pretrained(source).save_pretrained(target)
```

**2. Prepare the data.** The data is a Megatron `.bin/.idx` dataset. Use the output of MindSpeed's
`data_convert_deepseek32_pretrain.sh` directly, or build it from the `text` field of a JSONL file with the HyperParallel
tool (options in the [Offline data preparation guide](../../../docs/guide/data/offline_preparation_guide.md)), which
writes `/path/to/dataset/deepseek32_text_document.bin/.idx`:

```bash
python -m hyper_parallel.data.tools.offline_preparation \
    --dataset-name-or-path /path/to/train.jsonl \
    --tokenizer-name-or-path /path/to/DeepSeek-V3.2 \
    --output-prefix /path/to/dataset/deepseek32
```

**3. Launch.** The first argument is the cropped checkpoint directory and the second the common `.bin/.idx` prefix
(without suffix):

```bash
bash examples/training_demo/deepseek_v32/run_single_card.sh /path/to/DeepSeek-V3.2-crop /path/to/dataset/deepseek32_text_document
bash examples/training_demo/deepseek_v32/run_8card.sh /path/to/DeepSeek-V3.2-crop /path/to/dataset/deepseek32_text_document
```

The logs are written to `output/training_demo/deepseek_v32/run_single_card.log` and `run_8card.log`. Trailing
arguments override the YAML, for example `--training.train_iters=200`.

### Check that single-device and 8-device runs agree

The two runs start from the same state and can be compared step by step:

- **Same initial weights**: both load the same checkpoint. Do not switch to `from_config` random initialization: in a
  multi-process build the model is sharded on the meta device first and each shard is initialized separately, so a
  single device and 8 devices start from different weights.
- **Same inputs**: each 4096-token window is determined by the seed, split, sequence length and sample count, so both
  topologies read the same 2 samples per step; on 8 devices each of the two dp ranks takes one sample and CP splits its
  sequence in half.

Compare `training/foundation_loss` and `training/grad_norm` of every step in the two logs. Parallelism changes only the
accumulation order of the BF16 computation: the first-step difference is at the rounding level and grows slowly with
the steps. The example in the [model integration validation guide](../../../docs/guide/trainer/model_integration_validation.md)
sets the cross-topology threshold to an absolute or relative loss error of at most 5e-3 and a relative grad-norm error
of at most 5e-3 (or an absolute error of at most 0.125), which can serve as a reference.

### Loss comparison with MindSpeed-LLM

Both sides start from the same weights: HyperParallel runs the single-device example with the cropped checkpoint built
above, and MindSpeed converts the same directory with `ckpt_convert_deepseek32_hf2mcore.sh`, using the parallel layout
of its training script. The example's data reading (the same `.bin/.idx`, seed 1234, split 100,0,0, one causal sequence
per window) and optimizer settings (AdamW lr 1e-5, cosine decay to 1e-7, no warmup, weight decay 0.01, betas
(0.9, 0.999), gradient clipping 1.0) already match the MindSpeed script; use the same `global_batch_size` and
`train_iters` on both sides. The MindSpeed script uses `index_topk` 1024, so append
`--model.config_overrides='{index_topk: 1024}'` to the HyperParallel launch command. Align the remaining settings as
follows:

| Item | HyperParallel | MindSpeed-LLM |
|---|---|---|
| MTP | Not trained | Remove `--mtp-num-layers` from both the training and the conversion scripts, and `--mtp-loss-scaling-factor` from the training script |
| Load-balancing loss | None | `--moe-router-load-balancing-type none`; remove `--moe-aux-loss-coeff` |
| Expert bias | Used for selection, not updated | Keep `--moe-router-enable-expert-bias`; add `--moe-router-bias-update-rate 0` |
| Indexer KL | None | `--indexer-loss-coeff 0` |
| `index_topk` | `--model.config_overrides='{index_topk: 1024}'` | `--index-topk 1024` |
| Hadamard rotation | Not applied | Add `--no-use-sparse-c8-indexer`; otherwise the two differ by BF16 rounding |
| Norm weight precision | BF16 parameters in the forward pass | Remove `--init-norm-weight-in-fp32`, which keeps the MindSpeed `RMSNorm` and indexer `k_norm` weights in FP32 |
| Determinism | `enable_full_determinism: true` | `--npu-deterministic` and `HCCL_DETERMINISTIC=true` |

MindSpeed's `lm loss` corresponds to HyperParallel's `foundation_loss`. Compare the first-step loss before a long run.

## Known Limits

- No PP: the Trainer does not support pipeline parallelism yet.
- No sequence parallelism: the DSA replacement does not support SP.
- The expert bias is not updated, there is no load-balancing loss and MTP is not trained (as in Transformers).
- The validation covers the single-device and 8-device runs of the 2-layer cropped model; the full-size (61-layer)
  model and loading the complete released FP8 weights are outside it.
