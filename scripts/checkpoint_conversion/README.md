# JT MF-to-HP checkpoint conversion

These two standalone utilities convert the checkpoint format used by the
current two-layer JT/MindFormers precision experiment. They are not a generic
MindFormers or HyperParallel checkpoint converter.

## Format flow

```text
MF safetensors checkpoint
  ├─ model tensors
  ├─ adam_m.* / adam_v.* / muon_m.* optimizer tensors
  └─ step_num / epoch_num / global_batch_size / loss_scale metadata
        │
        │ convert_mf_ckpt_to_hp.py
        ▼
HP offline state directory
  ├─ model.npz       # HP parameter names and layouts
  ├─ optimizer.npz   # HP parameter names with adam/muon prefixes
  ├─ state.json      # step, epoch, batch and data cursor
  └─ CONVERSION.json # conversion metadata and tensor counts
        │
        │ materialize_hp_native_checkpoint.py
        ▼
Native HyperParallel DCP checkpoint
  └─ <output>/step-<N>/
     # model, optimizer and extra training state written by HP DCP
```

The `model.npz` directory is passed to the JT model builder as
`reference_weights`. The materializer builds the configured HP trainer,
injects `optimizer.npz` and the dataloader cursor, then calls the existing HP
checkpoint callback once to write the native DCP checkpoint.

## Commands

```bash
python scripts/checkpoint_conversion/convert_mf_ckpt_to_hp.py \
  --ckpt /path/to/CKP_rank_0-300_1.safetensors \
  --out-dir /path/to/hp-offline-state

python scripts/checkpoint_conversion/materialize_hp_native_checkpoint.py \
  /path/to/train-4k.yaml \
  --state-dir /path/to/hp-offline-state \
  --output /path/to/hp-checkpoints \
  --step 300
```

The config must build the matching JT model and point `reference_weights` at
the offline state directory containing `model.npz`.

## Dtype contract

- `convert_mf_ckpt_to_hp.py` preserves the source array dtype through layout
  transforms and `np.savez`; it does not convert model FP32 weights to BF16.
- The current MF checkpoint contract stores model and optimizer tensors in
  FP32. The materializer loads optimizer tensors into HP as FP32, which is a
  no-op for that checkpoint and preserves FP32 Adam/Muon optimizer state.
- BF16 safetensors input is not supported by the NumPy loader. It fails
  explicitly instead of silently converting it.
- Checkpoint storage dtype is separate from forward compute dtype. Linear,
  RMSNorm, Router and RoPE compute precision is controlled by the HP model and
  mixed-precision configuration, not by this conversion directory.
- JT model loading requires the `model.npz` dtype to exactly match the
  configured model parameter dtype.

## Current scope

The source mapping is for the current two-layer JT experiment:

- layer 0 is treated as dense;
- layer 1 is treated as MoE;
- the MTP layer is mapped;
- the materializer currently requires `global_batch_size=1`;
- `--step N` creates a trainer configured for the experiment's `2*N`
  continuation and writes `step-N`.

For a full or differently shaped model, extend the mapping and make the
batch/cursor/continuation settings explicit before using these utilities.

Neither script calculates hashes or checksums.
