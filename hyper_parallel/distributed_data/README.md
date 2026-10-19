# Distributed sample balancing

This PyTorch data path separates Dataset Readers, planning, and Data
Constructors, inspired by MegaScale-Data. Step membership belongs to the
upstream producer, not to a metadata read-ahead window.

```text
native BatchSampler selects this step's Dataset-index occurrences
  -> metadata: look up only selected indices
               -> Planner -> target Constructor reads dataset[index], no payload A2A
  -> metadata_fn: read selected samples, extract metadata
                  -> Planner -> payload A2A when redistributed
  -> preserve complete Dataset outputs -> collate_fn

external_step_source emits a complete local step of raw-sample bins
  -> HP derives metadata and freezes the source step -> Planner
  -> node-local payload A2A when redistributed -> pack_fn -> collate_fn
  -> buffered H2D -> Trainer (pure DP, no loader checkpoint)

native BatchSampler
  -> broadcast the local batch to model-parallel peers (schema object + direct tensor leaves)
  -> commit consumed progress -> Trainer
```

Model-parallel batch broadcast serializes the Python structure and non-tensor
values once. Gloo uses `broadcast_object_list`; HCCL/NCCL transports the same
schema as a byte tensor on the accelerator. Tensor leaves in dictionaries,
lists, and tuples use `dist.broadcast` directly, avoiding pickle copies. CPU
loads use the Gloo model group; accelerator tensor leaves stay on device.
Unsupported custom containers remain inside the serialized schema.

Configuration, callback results, sample conservation, packing capacities, and
checkpoint compatibility remain validated at their boundaries. Fixed-group
snapshots and internally generated tensor descriptors are not revalidated at
every stage. Payload A2A uses one pickle format without an application-level
checksum; duplicate received keys and missing/unexpected Constructor payloads
still fail. All ranks in a job must run the same code version. Runtime errors
propagate locally without error-synchronization collectives.

Checkpoint compatibility is checked from the saved state fields, without a
separate format version. Legacy `version` fields are ignored during restore.

`DistributedDatasetConfig.metadata_mode` selects the metadata source explicitly.
The default `False` is online mode: pass `metadata_fn` and let HP derive
metadata from the samples selected by the native `batch_sampler` (or by an
`external_step_source`). Set it to `True` for offline metadata: explicitly pass
`metadata`, an indexed object implementing `__len__` and `__getitem__`. Online
mode calls `metadata_fn(sample)` after reading the sample; offline mode reads
`metadata[index]` before fetching its payload. Offline mode requires a native
`batch_sampler`, and the metadata and Dataset indices must have the same length
and ordering. Missing metadata arguments fail at build time.
Passing `metadata` in online mode or `metadata_fn` in offline mode fails at
build time. There is no metadata-only streaming mode.
The loader never expands a candidate window to infer how many samples a step
should contain; there is no refill/`attempt` loop.

`DistributedDatasetConfig.broadcast_plan` defaults to `True`: one Planner
computes the plan and broadcasts it to the data-plane ranks. Set it to `False`
to let every data-plane rank compute the complete plan from the same gathered
metadata, skipping the plan broadcast. This applies to native sampler and
external-step loading. Keep configuration, cost models, and assignment policies
identical and deterministic across ranks. Metadata gathering, payload exchange,
and model-parallel batch broadcast still follow the normal loading path.

## Batch semantics

- `seq_len` is the hard token capacity of one normal sequence bin.
- `local_batch_size` is the number of sequences returned per DP rank and yield.
  Native BatchSampler mode treats each whole Dataset output as one such unit.
- One synchronized iterator step covers `local_batch_size * dp_size` bins.
  `balance_window_size` defaults to one. Native loading can jointly plan several
  accumulation rounds, bounded by the sampler's optimizer and epoch boundaries.
- Native BatchSampler fixes index occurrences and uses singleton reference bins;
  `external_step_source` fixes samples and lets HP preserve the original packing
  bins.
  Balancing may only rearrange that frozen sample set. It cannot skip expensive
  samples or change window membership. External sources retain one-step membership.
- In source mode, the packer must agree with metadata token accounting,
  including special tokens and reserved image placeholders. If sample-level
  placement fails, the Planner falls back to the known-feasible reference bins.
- Native BatchSampler always emits its complete batch. An external source emits one
  complete local step; HP owns metadata extraction, payload caching, and final
  packing/collation. Checkpointing is supported only by the sampler path.

The Planner consumes the step's metadata and original grouping directly:
`planner.plan(samples, reference_bins=reference_bins, step=step)`.
`samples` contains `SampleMetadata` entries with a bound `key`; `reference_bins`
contains their `SampleKey` values grouped in original data-rank/bin order. This
does not select or read additional samples. Original bins remain available for cost
comparison and fallback.
Metadata callbacks and precomputed metadata may leave `key=None`. Readers bind
a key to a copy for each sampling occurrence, so repeated indices and reused
metadata objects remain independent. Stream positions are stored only in
`metadata.key.global_sample_position`.
The caller is responsible for a complete, valid step and reference grouping;
the Planner does not repeat request validation. Placement capacity enforcement
and the final sample-conservation audit (when `validate=True`) remain unchanged.

## Public API

External applications pass their selected steps as `external_step_source` to
`build_distributed_dataloader`, alongside `DistributedDatasetConfig` and the
`metadata_fn`, `pack_fn` and `collate_fn` callbacks. This path
automatically evaluates node-local balancing, double buffers and stages H2D,
and yields ready device microbatches. Sample exchange occurs only when the
algorithm's objective improves by more than `min_balance_gain`.
Set `balance_group_size` to partition a large pure-DP mesh into independent
balancing groups, such as ten 1024-rank groups on a 10240-rank job. Native
BatchSampler loading does not use this option and retains its global data plane.
See [external-step integration](NODE_LOCAL_BALANCING.md) for the complete
contract. `cost_model` and `balancing_algorithm` are independently replaceable.
Native BatchSampler loading retains its selection/checkpoint mechanisms and
uses these same policy arguments; default costs require `model_config`.

```python
from hyper_parallel.distributed_data import (
    DistributedDatasetConfig,
    SampleMetadata,
    build_distributed_dataloader,
)
```

### Native HP BatchSampler: preserve each forward/backward round

Pass `batch_sampler=` to keep the native HP sampling policy. This is required
for ahead-of-fetch metadata. For online iterable pipelines, use the
`external_step_source` API instead.

The loader validates the sampler interface, DP parameters, and cursor alignment
locally. It does not hash sampler settings or index mappings for cross-rank or
checkpoint comparisons. Restore delegates to the native sampler's
`load_state_dict`; keep its sampling policy unchanged to replay the same samples.

```python
from hyper_parallel.data.parallel import build_dataset_batch_sampler

# Build the rank-local sampler on every rank, using training DP coordinates.
# Only the Data Constructor in each DP replica advances its sampler.
batch_sampler = build_dataset_batch_sampler(
    total_samples=len(dataset),
    micro_batch_size=2,
    global_batch_size=32,
    dp_rank=dp_rank,
    dp_world_size=dp_size,
    sampler_type="cyclic",
    seed=1234,
)
loader = build_distributed_dataloader(
    dataset,
    mesh,
    DistributedDatasetConfig(seq_len=32768, local_batch_size=2, metadata_mode=False),
    batch_sampler=batch_sampler,
    metadata_fn=lambda sample: SampleMetadata(pack_tokens=len(sample["tokens"])),
    collate_fn=native_collate_fn,
    cost_model=my_cost_model,
)
```

```text
each DP Constructor advances its native BatchSampler once
  -> freeze that round's native Dataset-index occurrences
  -> online: read those Dataset outputs, then derive metadata
     metadata: look up metadata for exactly those Dataset indices
  -> balance only this frozen round (one complete Dataset output per bin)
  -> online: payload A2A / shared metadata: target reads, no payload A2A
  -> original collate_fn receives complete Dataset outputs
  -> broadcast the local batch to model-parallel peers
  -> validate the batch, commit the native sampler cursor, and return the data
```

For DP=2 and `local_batch_size=2`, if native samplers yield `[0, 1]` and
`[2, 3]`, the default window balances only these four occurrences. Larger windows
collect consecutive rounds within the same optimizer step. Index mappings that
repeat a Dataset index remain repeated occurrences; routing keys distinguish
them by `global_sample_position` rather than deduplicating the index.
In this mode that position is a stable DP-major occurrence slot within the
native round, not a physical document index or the sampler's internal shuffle
permutation position.

In the HP text Trainer, retain the original Indexed Dataset and collator and
enable this path with:

```yaml
dataset:
  data_config:
    packing_stage: dataset
    load_balance: true
    seq_length: 32768
```

`load_balance` is a boolean switch and defaults to `false`. Set it to `true`
to enable balancing; other types, including the former `native_batch_sampler`
string, are rejected.

### Balance several gradient-accumulation rounds

Set `dataset.data_config.balance_window_size` to the
maximum number of consecutive forward/backward rounds to plan together. For
DP8, micro-batch size one, accumulation ten, and a three-round window:

```yaml
training:
  global_batch_size: 80
  micro_batch_size: 1
dataset:
  data_config:
    load_balance: true
    repack_sequences: true
    balance_window_size: 3
```

Each optimizer step uses windows of `3 + 3 + 3 + 1` rounds. A three-round
window jointly balances 24 original records across eight ranks and three
micro-batches. With `repack_sequences: true`, the independent sequences inside
those records can be repacked across that entire window. Without repacking,
whole records are redistributed. Each `next(loader)` still returns one normal
micro-batch; optimizer frequency and global sample membership stay unchanged.

For direct construction, pass `balance_window_size=3` in
`DistributedDatasetConfig`. Accumulation is derived from
`batch_sampler.global_batch_size / batch_sampler.global_micro_batch_size`.
The positive integer defaults to one. Values larger than accumulation are
clipped at the next optimizer boundary. Epoch tails contain only the available
complete distributed micro-batches; windows never borrow from the next epoch.
External-step loaders require `balance_window_size=1`.
In Trainer configuration, this option belongs directly under `data_config`;
placing it inside `distributed_dataloader` is rejected.

Planning treats each `(micro-batch, DP rank)` as a destination. Custom assignment
policies therefore receive `DP size * window rounds` virtual ranks, ordered by
micro-batch then physical DP rank. The acceptance gate sums the per-round
objectives; the default makespan objective estimates the sum of each round's
slowest-rank cost. Samples are transferred, collated, and copied to the device
one micro-batch at a time.

For LPT policies with one pack per rank and multiple rounds, the planner also
schedules intact packs by decreasing backbone cost. It compares the unscheduled
candidate, scheduled candidate, and scheduled original layout using the sum of
per-round objectives, preserving better candidates for variance or custom LPT
objectives. Every candidate is validated before scheduling, and the final gain
must still strictly exceed `min_balance_gain` relative to the original order.
This can improve a window even when document-level repacking fails. It does not
change attention boundaries inside a pack. Single-round windows, multiple local
packs per rank, and non-LPT custom policies retain their existing behavior.

Native loading uses `DistributedDataLoader` to double-buffer balancing windows,
including accumulation one as a single-micro-batch window. A persistent
producer prepares every micro-batch in the current window before its first batch
is returned. It then immediately prepares the next window: Dataset reads,
splitting and metadata, window-level balancing, per-micro-batch A2A,
packing/collation, H2D, and model-peer input broadcasts. Data collectives run
in order on dedicated groups and a producer-owned device stream. A window is
published once its completion event is recorded, without a producer-side CPU
wait. Consumption makes the training stream wait on that event and records
the training stream on the returned tensor storage.

Balancing reports are disabled by default for native and external-step loading.
Enable them explicitly in the Trainer configuration:

```yaml
dataset:
  data_config:
    load_balance: true
    log_balance: true
```

Direct API users can set `DistributedDatasetConfig(..., log_balance=True)`.
Reports use INFO level on `hyper_parallel.distributed_data.balance_logging`;
the Trainer enables INFO by default. Setting the logger to INFO without enabling
`log_balance` does not emit balancing reports.

Each native `[dp-balance]` report describes one balancing window, including its epoch,
one-based optimizer iteration within that epoch and micro-batch range, original/candidate/accepted
objectives, gain threshold and acceptance decision. Per-round layouts show each
DP rank's packs with token counts, sample counts and estimated LLM costs. These
costs use the configured model's units, not measured milliseconds.

Only the configured planner rank writes these reports (normally global rank 0),
including when every reader independently plans with `broadcast_plan=False`.
Reports are frozen during planning and printed when the main thread starts
consuming that window, so speculative future windows do not appear early.
An optimizer step with several windows emits each report as its window becomes
current; a short epoch tail reports only its available rounds. Logs add no collectives, device reads or
synchronization, and are not included in checkpoints. Setting `log_balance=False`
or a WARNING-or-higher level on this logger disables native statistics
formatting as well as output.

Prefetch size follows `balance_window_size`, clipped at each optimizer boundary.
Accumulation ten and a three-round window prepare `3 + 3 + 3 + 1` batches in
separate windows. At most the remaining current-window inputs and one future
window are buffered, rather than two complete optimizer steps. The producer
does not start a third window until the second becomes current. Thus the loader
buffers at most twice the configured window's micro-batch count; this is not a
byte limit on variable-size payloads, temporary exchange/staging buffers,
in-flight device work, or inputs retained by training. Source-window lookahead
is disabled because the loader producer owns all speculative reads.
Accumulation one follows the same background pipeline. A blocked or failed
future window does not prevent delivery of current inputs; its error is raised
when that window is requested. `close()` drains production.

Epoch exhaustion stops iteration while keeping the producer thread and its
device stream alive, waiting for another request. `set_epoch()` drains any
pending preparation before resetting source and delivery state, then reuses
the same producer and stream. Explicit `close()` or context-manager exit ends
the loader lifetime; a closed loader cannot be restarted with `set_epoch()`.
Text and VLM Trainers close their collective loader before distributed teardown.

The Trainer retains its existing per-micro-batch loss normalization. If valid
token counts differ between rounds, cross-round rearrangement can change token
weights; optimizer sample conservation does not imply identical losses or gradients.
Window-level token normalization is not enabled by this data-loading option.

Only a window ending at a complete optimizer boundary carries a checkpoint
snapshot. The public cursor advances after the final micro-batch of that window
has been returned, completing the step. Intermediate windows do not advance it. During an
incomplete step it retains the preceding optimizer boundary, so restoring that
cursor replays the whole incomplete step. A short epoch tail does not advance it.
The prefetched future window is excluded, even if the live sampler has advanced.
Checkpoints store no remaining-round count, pending window metadata, plans or raw
sample tensors. Restore reads and plans again from the saved sampler boundary;
the Dataset, split callback and planning policies must remain deterministic.
Save model, optimizer, RNG and loader state together after the optimizer update:
the loader cannot observe that update, and partially accumulated gradients are
not saved. Legacy complete-step checkpoints remain readable; partial-step
checkpoints and initial sampler cursors inside an optimizer step are rejected.

### Window-aware greedy placement and bounded refinement

Ordinary LPT can fragment remaining capacity and fall back to the original
layout even when that layout is feasible. The default assignment policy,
`FeasibleLPTBalancingAlgorithm`, automatically attempts bounded feasibility repair
after greedy failure and refines successful `makespan` assignments. In multi-round
windows with one pack per rank, placements minimize the sum of scheduled round
maxima, with remaining capacity as the first tie-breaker. For offline-packed text, enable balancing and document
repacking in the Trainer with:

```yaml
dataset:
  data_config:
    load_balance: true
    repack_sequences: true
```

No separate repair option is needed. The builder and planner use the same
default policy. An explicitly supplied or configured `balancing_algorithm`
is preserved; callers can select `LPTBalancingAlgorithm()` for ordinary greedy
assignment without repair.

Refinement requires one pack per rank, at least two non-oversized packs, and no
additional `packing_budgets`. It uses suffix subset sums to keep two-pack
completion feasible and non-empty, including underfilled packs. There is no pack-count cutoff: a window
shares a budget of at most `max_repair_pairs` checks (default 64), including failed
or memory-limited attempts. Candidates are generated on demand from each round's
highest-cost pack and up to twelve heavy, light or nearby-cost partners, ranked
by their potential window gain. After accepting a move, repair recomputes the
bottlenecks; a pair can be checked again only if one of its packs has changed.
The complete pair set is never enumerated.

For `makespan`, each repair must lower the sum of round maxima after intact-pack
scheduling. Other objectives retain the original full-pack failure repair and round order. The
planner still validates sample conservation and capacity, compares the scheduled
reference, and applies `min_balance_gain` to the complete window. Each pair's
estimated bitset table remains capped at 32 MiB; this is not a process-memory
limit. Unsupported domains retain ordinary LPT. The deterministic check budget
bounds repair attempts, not elapsed CPU time; larger inputs still cost more to
read, sort and score. The heuristic does not guarantee an optimal layout.

Feasibility repair also applies to a single round without gradient accumulation.
Intact-pack round scheduling requires multiple rounds and also applies to
ordinary LPT. Both preserve the selected sample occurrences and use the
existing capacity checks and benefit gate. Checkpoint replay rebuilds plans from
the saved optimizer boundary, so exact replay requires unchanged planning policies.

The Trainer builds its normal `build_dataset_batch_sampler` first, including
`single`/`cyclic`, `data_sharding`, and the supported index rearrangement map.
It passes that sampler to the collective loader instead of applying another
Reader stride. Worker settings and the original `dataloader.collate_fn` are
retained. The text adapter supplies physical token lengths as `P`, with `D=0`;
it does not extract internal TND boundaries. Fixed-length GPT outputs therefore
do not automatically gain compute balance from this route. The Trainer supplies
its effective model configuration to the default FLOPs model. Architectures
outside that model's latent-attention scope require an explicit `cost_model`;
metadata's old proxy `cost` is not a fallback.
The Python builder accepts `cost_model` and `balancing_algorithm` directly.
AutoModels YAML can instead declare optional factories under `dataloader`:

```yaml
dataloader:
  cost_model:
    _target_: my_training.cost.SampleCost  # user-provided factory
  balancing_algorithm:
    _target_: hyper_parallel.distributed_data.FeasibleLPTBalancingAlgorithm
    objective: makespan
    max_repair_pairs: 64
```

These factories are constructed only for `load_balance: true`;
they may accept the runtime `model_config`. Explicit Python policy arguments
take precedence over configured factories. Omitting both selects default costs
and LPT, not a separate disabled planner.
The Trainer passes the foreground rank-local device for final H2D. NPU meshes
default to HCCL data transport; CPU/CUDA meshes default to Gloo data transport.
Explicit `communication_backend` settings take precedence. Accelerator model
peers use a separate HCCL/NCCL tensor-broadcast group when needed.

Current boundaries of the native-sampler path:

- `dataset[index]` stays **whole** by default. Online native loading can opt in
  to independent-sequence repacking with `split_fn` and `pack_fn`. The selected
  record set stays fixed; the original sequence grouping is the fallback when
  a feasible, sufficiently better placement cannot be found.
- Supply an HP-compatible native sampler on every rank. Only Constructor ranks
  need the mapping Dataset, and only those ranks advance their samplers. Readers
  must currently coincide with Constructors. The native sampler owns ordering
  and DP slicing, so `dataset_already_sharded=False` is required in
  `DistributedDatasetConfig`, and `drop_last=True` is required. Without
  `split_fn`, `pack_fn` must be omitted. Pure-DP iterable sources use
  `external_step_source`.
- Supplied metadata must describe the **logical native Dataset outputs** after
  blend/shuffle/sample-index mapping, not raw document IDs. Raw `.idx` lengths
  alone do not describe GPT-internal EOD/TND boundaries. This version does not
  automatically generate native-GPT metadata; the Trainer opt-in uses online
  metadata. Shared-metadata reads also require deterministic, rank-independent
  Dataset outputs.
- Save and restore through `loader.state_dict()` / `load_state_dict()`, not
  through the original sampler: its live cursor may include a prefetched window.
  Native cursor snapshots count Dataset-output occurrences, never
  tokens or fragments, and can restore at a synchronized forward/backward boundary with
  unchanged DP topology, global batch size, and sampler policy.
- Whole-record loading preserves pre-collation Dataset fields. Repacking must
  preserve each independent sequence's tokens, shifted labels, attention and
  position semantics. Neither mode promises bitwise training results. Worker
  RNG replay, batch-dependent transforms, dropout, reduction order and full NPU
  training parity are outside this sampler integration. The Trainer path rejects PP and
  a second `DynamicBatchDataLoader` selection stage.

### Repack offline-packed text within a native window

Set `dataset.data_config.repack_sequences: true` alongside
`packing_stage: dataset` and `load_balance: true`.
Each Dataset output must provide one-dimensional `input_ids`, pre-shifted
`labels`, and explicit leading-zero `cu_seq_lens` covering the complete record.
The boundaries must identify independent attention sequences. No boundary is
inferred from supervision masks or guessed from token values.

The Trainer installs `split_packed_text_sample`, `pack_indexed_text_samples`
and `collate_indexed_text_sequences`. Metadata and workload costs are evaluated
per independent sequence. The original records remain the sampler's unit of
selection and checkpoint progress. A runtime `SampleKey.sequence_index`
distinguishes sequences within each record occurrence, including repeated
Dataset indices. All sequences in a window are delivered before its sampler
cursor is committed; public checkpoint progress advances only at complete
optimizer-step boundaries.

Use `TextParallelBatch(source_type="indexed_source", reset_position_ids=True)`
with `create_attention_mask_in_dataloader: true`. Compressed attention also
requires the model's runtime adapter to pass the new boundaries to attention.
Repacking currently requires online metadata (`metadata_mode: false`); supplying
`split_fn` with precomputed metadata is rejected. The generic builder accepts
custom `split_fn`/`pack_fn` callbacks; the Trainer switch selects the text callbacks.

The [JT DP8 repacking recipe](../../examples/training_demo/jt_deepseek_v3/jt_deepseek_v3.tp1_ep1_dp8_repack.yaml)
reads a Hugging Face Dataset saved with `save_to_disk`, containing `input_ids`,
`labels` and `cu_seqlens`. It preserves all record tokens and targets, and uses
`JTSequenceRuntime` to deliver cumulative ends as `actual_seq_len`.
Use initial weights and model dimensions whose vocabulary matches the data.

```bash
JT_RECIPE_NAME=jt_deepseek_v3.tp1_ep1_dp8_repack.yaml \
  bash examples/training_demo/jt_deepseek_v3/run_jt_deepseek_v3.sh \
  /path/to/jt_initial_weights /path/to/packed_arrow_dataset
```

With DP8 and micro batch 1, each round still selects eight original records and
produces eight packed rows. Each row is padded to `seq_len`, with ignored labels
and a separate attention segment for padding. Fully occupied original rows can
leave little room for a better feasible partition; the default greedy algorithm
then retains the original grouping. Repacking does not imply a guaranteed speedup.

Native-sampler regression coverage:

```bash
  python -m pytest -q \
  tests/ut/distributed_data/test_batch_sampler.py \
  tests/ut/models/jt_deepseek_v3/test_jt_data.py \
  tests/torch/distributed_data/test_batch_sampler_gloo.py \
  tests/torch/distributed_data/test_dynamic_packing_gloo.py
```

### Native HP image-text Dataset (VLMTrainer)

VLMTrainer can use the same native-sampler path. Add `load_balance` to the
existing VLM Dataset configuration; keep the original transform and collator:

```yaml
dataset:
  _target_: hyper_parallel.data.omni.build_dataset.build_online_omni_mapping_dataset
  data_path: /data/conversations.json
  data_config:
    source_type: online
    load_balance: true
  data_transform:
    _target_: hyper_parallel.data.omni.omni_transform.AutoProcessorTransform
    max_seq_len: 32768
dataloader:
  _target_: hyper_parallel.data.batching.FixedBatchDataLoader
  sampler_type: single
  drop_last: true
  num_workers: 2
  prefetch_factor: 2
  collate_fn:
    _target_: hyper_parallel.data.batching.build_collate_fn.build_omni_collate_fn
training:
  micro_batch_size: 2
  global_batch_size: 32
```

This is a data-section example to merge into an existing VLMTrainer model/run
configuration. The distributed sequence capacity is inferred from the built
transform's `max_seq_len`; `data_config.seq_length` is unnecessary. If both are
supplied, they must agree. With DP=8 this example selects 16 global samples per
forward/backward round and accumulates two rounds per optimizer update.

```text
native HP BatchSampler selects each round's Dataset-index occurrences
  -> original VLM Dataset + processor read/decode/encode complete conversations
  -> vlm_sample_metadata extracts padded width and image patch counts
  -> gather metadata and balance this frozen round across DP ranks
  -> A2A complete image-text sample dictionaries
  -> original VLMCollator: stack text [local_batch_size, max_seq_len]
                          concatenate image patches and image grids
  -> VLMGetBatch -> model forward/backward -> gradient accumulation
```

- A conversation may contain multiple images and turns, but remains **one
  indivisible sample**. Input IDs, prompt-masked labels, attention masks,
  modality markers, pixels and grids move together. No label shifting, new
  padding policy, or cross-conversation packing is introduced.
- `hyper_parallel.data.omni.metadata.vlm_sample_metadata`
  describes padded width as `pack_tokens`/`P`, sets `D=0`, and exposes
  `image_patches=sum(T * H * W)` from the processor's actual `image_grid_thw`.
  Raw patches are not merged LLM image placeholders. Legacy proxy costs remain
  in the metadata for compatibility; the planner uses only the resolved cost
  model's estimates. Supply custom estimates through `cost_model`, separately
  from the metadata callback. The YAML above must be paired with a compatible
  default-model configuration or a user cost-model factory as shown earlier.
- The Trainer integration uses **online metadata**, extracted after the native
  Dataset transform. It cannot balance CPU image decoding/processing already
  performed by the Readers. Native `_TransformDataset` still performs its
  initial trainable-label filtering; no precomputed metadata is automatically generated.
  Manually supplied metadata through the lower-level API must match the
  filtered/transformed Dataset index space, not the original JSON row numbers.
- The HP VLM runtime currently requires **TP=CP=PP=1**; this integration keeps
  that boundary and supports DP. It does not add VLM packing, video/audio
  adapters, or a new model-parallel batch-broadcast implementation.
- Set `load_balance: false` or omit it to retain the native DataLoader. Enabled loaders own
  checkpoint cursors and epochs; VLMTrainer preserves a restored first-epoch
  cursor and waits for outstanding prefetch before distributed teardown.
  CUDA/NPU payload communication uses the existing device transport; the
  tests below cover CPU/Gloo, not full accelerator-model accuracy or speed.

VLM regression coverage uses real local images and the original HP Dataset,
transform, collator and get-batch path with a deterministic test processor
(no remote assets). It checks sample/field conservation, DP2 payload A2A,
worker loading, double buffering, resume, and native-vs-balanced gradients for
an image-conditioned toy loss using HP token normalization:

```bash
python -m pytest -q \
  tests/ut/data/omni/test_metadata.py \
  tests/ut/data/vlm/test_collator.py \
  tests/torch/distributed_data/test_batch_sampler_gloo.py
```

### Migration from metadata streaming

The Trainer shortcut `packing_stage: distributed_dataloader` depended on
streaming metadata selection and now raises a migration error. For native HP
training, retain `packing_stage: dataset` and set
`load_balance: true`. This preserves whole GPT Dataset outputs;
it does **not** rebalance document fragments inside a packed GPT sequence.

The lower-level Indexed source Dataset and text packing helpers remain available
for custom producers. To retain dynamic packing across raw source samples, that
producer must define complete steps and provide an `external_step_source`.
The source only iterates raw local steps; HP derives metadata, caches payloads,
and applies packing/collation. HP does not infer step boundaries by scanning
source metadata. Source recovery belongs to the caller's training components.

All paths use the same cost-model and assignment pipeline (default: LPT with bounded repair),
one-step double buffering, and automatic H2D on the selected accelerator.
There is no balance-enable or synchronous-loader switch. CPU execution keeps
batches on the host. The original step membership and gain gate are unchanged.
Gloo defaults to CPU batches; pass an explicit NPU/CUDA `device` to enable H2D
while retaining Gloo for data-plane communication.
Native sampler loading retains checkpoint/resume and metadata direct reads;
metadata mode still skips payload A2A.

`next(loader)` returns device-ready inputs. One shared producer-owned copy
stage pins host tensors, enqueues non-blocking H2D on a separate stream, and
records a ready event. The consumer stream waits on that event and records
storage ownership before model broadcast or training uses the inputs.
Tensor leaves in standard containers move recursively without changing dtype;
non-tensor metadata remains unchanged. Batch objects with a `to` method use that
method. For source-only local steps, copies run per microbatch; for samplers they
run on the collated batch. Local-step loaders retain CPU inputs in
`last_host_batch` for host metering.
The old standalone trainer-owned device-prefetch slot has been removed.

HCCL/NCCL data collectives run in a consistent order on dedicated groups.
The native loader's persistent producer owns the entire next-window pipeline,
including metadata length exchange, result decoding, planning, payload A2A,
H2D and model-peer broadcasts. This applies with or without gradient accumulation.
It starts the next window before returning the first current-window batch.

```python
batch = next(loader)  # the next balancing window is already being prepared
loss = forward(batch)
backward(loss)
```

Native loading exposes `prefetch()` as an optional, idempotent request for the
next balancing window. It neither waits for metadata/plans nor launches foreground
data communication. External loops can rely on `next(loader)` to start prefetch
automatically. The foreground waits for unfinished preparation only when `next(loader)`
needs the next window. CPU scheduling and shared device resources can still limit
actual overlap; hiding all loading time depends on compute time and data volume.

All model peers call `next(loader)` in the same order. Constructor ranks stage
H2D before direct tensor broadcast to model peers; Gloo data transport can use
a separate HCCL/NCCL model tensor group. CPU-only fields retain their placement.

Checkpoint through the native sampler loader after the optimizer update for all
micro-batches in the step. Its snapshot excludes speculative progress; requesting
it does not wait for or consume pending batches. Call
`wait_for_prefetch()` to wait for device completion before process-group teardown,
or use the loader as a context manager to also release its persistent worker.
Local-step loaders retain CPU metering inputs in `last_host_batch` and do not
support checkpoint/resume.

PyTorch DataLoader execution options can be passed through
`dataloader_kwargs`:

```python
loader = build_distributed_dataloader(
    dataset,
    mesh,
    DistributedDatasetConfig(seq_len=32768, local_batch_size=2, metadata_mode=False),
    batch_sampler=batch_sampler,
    metadata_fn=metadata_fn,
    cost_model=my_cost_model,
    dataloader_kwargs={
        "num_workers": 8,
        "pin_memory": True,
        "prefetch_factor": 4,
        "persistent_workers": True,
        "timeout": 60,
        "worker_init_fn": worker_init_fn,
        "multiprocessing_context": "spawn",
        "pin_memory_device": "",
        "in_order": True,
    },
)
```

Values in `dataloader_kwargs` override the legacy `num_workers`,
`pin_memory`, `prefetch_factor`, and `persistent_workers` fields in
`DistributedDatasetConfig`. The supported keys are those four fields plus
`timeout`, `worker_init_fn`, `multiprocessing_context`, `pin_memory_device`,
and `in_order`; deterministic routing requires `in_order=True`. Accelerator
jobs using NPU or CUDA should generally select the `spawn` multiprocessing
context. `timeout` controls how long DataLoader waits for workers and is not a
distributed collective timeout. A custom `worker_init_fn` must be consistent
across ranks and picklable when using `spawn`; with persistent workers, PyTorch
invokes it only when each worker process is created.

DataLoader topology, configuration and cost/assignment policies are not
fingerprinted across ranks or in checkpoints. Callers must keep topology, data
semantics and policies consistent across ranks and compatible with the saved
state. Worker count, prefetch settings, and other execution options may change
when resuming; Dataset outputs must remain deterministic. Global rank ownership
is still checked, readers validate their saved state, and the native sampler
validates its restored cursor and batch sizes. Legacy loader fingerprint and
plan-ID fields are ignored; component checkpoint formats must still be
compatible. Plans expose their step, sample assignments and costs
directly; no plan digest or `last_plan_id` is generated.

Dataset and distributed batching semantics remain internal. Do not pass
`dataset`, `batch_size`, `shuffle`, `sampler`, `batch_sampler`, `collate_fn`,
`drop_last`, or `generator` through `dataloader_kwargs`.

### Ahead-of-fetch metadata

Supply metadata aligned one-to-one with the mapping Dataset's logical index
space. Each Constructor has the same Dataset and metadata, plus its DP-local
native sampler:

```python
loader = build_distributed_dataloader(
    dataset,
    mesh,
    DistributedDatasetConfig(seq_len=32768, local_batch_size=2, metadata_mode=True),
    batch_sampler=batch_sampler,
    metadata=precomputed_metadata,
    collate_fn=native_collate_fn,
    cost_model=my_cost_model,
)
```

The sampler determines exactly which indices belong to this step. Readers look
up metadata only for those occurrences, the Planner balances them, and each
Constructor reads only its assigned payloads. No payload A2A is needed; metadata
gather and plan broadcast still occur. Do not pass `pack_fn`: each Dataset
output remains one indivisible sample. The default collator returns a tuple of
these samples.

All Constructors must be able to read the same global Dataset indices.
Reader-private shards with metadata routing are no longer supported by this
builder. Compose multiple datasets and their metadata in the same global index
order, then let the native BatchSampler own DP slicing and shuffling.

### External online steps

The source API separates the external producer from HP's distributed data
semantics. The source only iterates complete local steps; an optional epoch
hook resets source selection:

```python
class MyStepSource:
    def __iter__(self): ...  # yields local_batch_size raw-sample bins
    def set_epoch(self, epoch): ...
```

Pass it as `external_step_source`. HP calls `metadata_fn` for each raw sample,
keeps payloads while preparing the step, performs distributed placement, and
then invokes `pack_fn` and `collate_fn` on the final local batch:

```python
loader = build_distributed_dataloader(
    None,
    mesh,
    config,
    external_step_source=source,  # every rank supplies its own pure-DP raw steps
    metadata_fn=metadata_for_sample,
    pack_fn=pack_one_sequence,
    collate_fn=collate_packed_sequences,
    device=torch.device("npu", local_rank),
    cost_model=my_cost_model,
)
```

The source must yield exactly `local_batch_size` non-empty raw-sample bins per
step. This node-local route automatically buffers preparation/H2D and applies
the shared gain gate. It does not provide checkpoint/resume. HP owns metadata,
payload and placement state; source epoch handling stays in the source.

The caller's training components own checkpointing and restore the source to
its next unconsumed step before building a new loader. This layer maintains
only transient metadata, payloads, and placement for preparation and delivery;
it requires no source `state_dict`, `load_state_dict`, or Reader lifecycle
methods. `set_epoch` is optional. Configure DataLoader workers on the external
producer itself.

The default packer and collator preserve the structure as a tuple of bins,
each containing the raw samples. Custom callbacks must be consistent across
ranks.

Online payload A2A defaults to CPU/Gloo. A rank-local `device`
uses the WORLD accelerator backend by default (HCCL for NPU or NCCL for CUDA);
Python payloads still incur serialization and Host/device copies. The enabled
node-local path always exchanges raw samples with Gloo and uses the training
device only for final H2D. Shared metadata mode bypasses payload transport entirely.

## Current boundaries

- Native BatchSampler requires a mapping Dataset; metadata also requires shared indices.
- Iterable/streaming online data uses an external complete-step source on every pure-DP rank.
- Node-local balancing retains at most one prefetched step, including
  final H2D. The first step waits for preparation; the original path is synchronous.
- Gloo control plane and pickle payloads; online A2A
  may use Gloo, NCCL, or HCCL.
- `drop_last=True`; every DP rank receives the same number of non-empty bins.
- Native sampler checkpoints are per rank and save the last fully delivered
  optimizer-step cursor, excluding prefetched payloads and incomplete steps.
  Sample keys and plans replay when the Dataset stream and planning policies are
  deterministic for a given epoch; arbitrary Dataset/worker RNG state is not captured.
  Every training rank must save and restore its own loader state.
- Elastic resharding and structured zero-copy tensor transport are separate
  extensions.

## Buffered node-local balancing

For automatic Host/H2D buffering around an existing rank-local loader, v1 cost
estimation, node-local LPT balancing and rank-zero DP logs, see
[the reference configuration](NODE_LOCAL_BALANCING.md). Supply `model_config`
to use the default cost, or provide `cost_model`. A custom
`balancing_algorithm` owns assignment and its scalar objective; Hyper applies
the shared `min_balance_gain` acceptance threshold before sample exchange.
Gloo uses a speculative producer thread. HCCL keeps collective launches on
the training thread, with independent data groups and a data stream. Calling
`prefetch_plan()` after forward submission and `prefetch()` after backward
submission overlaps CPU planning, payload exchange, collation and H2D with
pending computation. Call these hooks consistently on all ranks before host
synchronization; ordinary iteration remains supported without the hooks.
