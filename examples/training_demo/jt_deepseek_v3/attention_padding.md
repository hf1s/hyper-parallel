# Skip trailing padding in JT Attention

This opt-in path uses the existing prepared SFT dataset and shared text loader.
It does not require data balancing or sequence repacking. Keep
`PreTokenizedSFTTransform`, and replace only the collator in the training YAML:

```yaml
dataloader:
  collate_fn:
    _target_: hyper_parallel.models.jt_deepseek_v3.adapter.data.collate.build_attention_padding_collate_fn
    padding_token_id: 151645
  get_batch:
    _target_: hyper_parallel.data.batching.TextParallelBatch
    source_type: online
    attention_mode: compressed
    reset_position_ids: true
    runtime_input_adapter:
      _target_: hyper_parallel.models.jt_deepseek_v3.adapter.data.runtime.JTSequenceRuntime
```

`151645` is the explicit padding contract of the example dataset, not a universal
padding ID. Only a continuous trailing run of this ID is marked as padding, and
its labels must all be `-100`. Ignored prompt tokens and interior occurrences
remain real tokens. Without `padding_token_id`, only padding added by the shared
collator for CP/TP sequence alignment is excluded from Attention. The ordinary
collator retains the existing behavior.

One prepacked record is accepted per micro-batch; it may contain multiple
independent documents. Token IDs, pre-shifted labels and physical sequence
length remain unchanged. The collator separates the padding tail in
`cu_seq_lens` and emits `attention_token_count`. CP preserves the global count,
TP broadcasts it, and `JTSequenceRuntime` produces physical `actual_seq_len`
and real `attention_seq_len`. MTP retains the real document-end boundaries.

Attention crops Q/K/V to the real domain before FA (or the CPU reference path),
then restores the physical output shape with zeros before CP redistribution.
Only real tokens contribute to the observed QK maximum. This applies to trunk
and MTP Attention, including both JT MLA CP strategies.

QKV projections, MLP/MoE, routing statistics and auxiliary losses still operate
on physical tokens. Their padding cost remains; auxiliary losses and QK
clipping statistics can change compared with a run that computes Attention on
padding. This feature does not establish full-training numerical equivalence
to the old padded path. It requires separate training validation for that claim.
