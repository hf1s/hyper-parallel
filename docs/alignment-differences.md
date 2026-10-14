# Alignment differences

This branch is an archive of the adapter-only Muon alignment patch applied to
`feat/jt-muon-optimizer-switch` at baseline `57c35de0f58ae30cba8aa80cf0b62db35e1f51b7`.
It is not a claim that every open alignment item is included.

## Included patch: Muon adapter

The patch adds `hyper_parallel/models/jt_deepseek_v3/adapter/runtime/mf_muon_ns.py`
and changes `mf_muon_parity.py`; `hyper_parallel/core/optimizer/muon.py` is not
changed.

- `mf_muon_ns.py` implements the MindFormers-aligned Newton--Schulz arithmetic,
  including the 2-D/3-D numerator and denominator handling, bf16-stored legacy
  coefficients, per-operation rounding, and the fused update arithmetic.
- It provides a float32 momentum lane and a shape-keyed float32 stash for the
  whole, non-transposed 2-D blocks that require the pre-cast float32 numerator.
- `mf_muon_parity.build_optimizer` injects the existing `zeropower_fn` and
  `momentum_update_fn` hooks.
- The adapter calls `setdefault("fp32_update", True)` and
  `setdefault("ns_per_param", True)` on the `mf_muon` configuration path. The
  framework patch in this branch adds these two generic switches with defaults
  `False`; the adapter therefore requests them as `True` on this path. No
  MF-specific arithmetic is added to the framework.

The adapter's switch defaults are an implementation/configuration statement;
they are not by themselves an end-to-end 300-step result.

### Available verification evidence

The adapter Newton--Schulz intermediates were verified on 910B + torch_npu:

- all seven `ns1` block classes: 100% matching;
- router `ns2`, `ns3`, `ns4`, `ns5`, and `x_ret`: 40960/40960 (100%) matching.

These are module/intermediate checks. They are not a claim of a valid 300-step
end-to-end comparison.
### Trace-only wiring acceptance

The clean trace-only Z+ run produced 42 `[MFNS] call ...` records. The
observed classes matched the adapter rules: whole, non-transposed 2-D blocks
such as `(3072, 5120)` and `(5120, 16384)` reported
`fp32_numerator=True, stash_hit=True`; sliced or internally transposed blocks
reported `fp32_numerator=False`; and 3-D blocks reported no float32 numerator
classification. This verifies hook invocation and the intended stash/class
selection behavior.

Parameter-level bitwise comparison was not completed because all three output
disks were full. Consequently, this branch makes no parameter-dump
逐字节一致 claim. The `lm_head` gradient, attention backward GEMM, and shared
`dy` accumulation boundary remain unfinished alignment scope.


## #190 items already merged or represented by open PRs

- #148: merged.
- #162: merged.
- #163: merged.
- #180: draft.
- #192: draft.

This branch does not copy the contents of those PRs unless they are part of the
adapter-only patch above.

## #180 open items

The following items remain open and are not implemented by this branch:

- `lm_head` weight gradient discrepancy: 0.229 ULP;
- attention backward GEMM;
- Fix2/Fix4 switches.

No claim is made here that these items are resolved.

## Measurement scope lesson

All numbers from the 17:41--18:05 pollution period are invalid and must not be
used as evidence. The shared experiment directory was contaminated by testing
changes during that period; those runs do not establish code correctness.
