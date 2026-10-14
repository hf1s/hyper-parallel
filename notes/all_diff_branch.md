# All-differences branch report

## Scope

- Branch: `align/all-differences-300step`
- Baseline remote ref fetched: `ghfork/feat/jt-muon-optimizer-switch`
- Baseline SHA: `57c35de0f58ae30cba8aa80cf0b62db35e1f51b7`
- Worktree: `hyper-worktrees/align-all-differences-300step`
- Applied patch: `mf-muon-align-20261015/mf-muon-adapter-only.gitpatch`

## Included

The branch contains the Z+ patch: the new `mf_muon_ns.py`, the
`mf_muon_parity.py` hook wiring, and the generic framework switch changes in
`core/optimizer/muon.py`. The framework change adds only `fp32_update` and
`ns_per_param`, both defaulting to `False`; it contains no MF-specific
arithmetic. The adapter requests both switches as `True` on its `mf_muon`
path. The adapter's effective module/intermediate evidence is seven `ns1`
classes at 100% and router `ns2`--`ns5` plus `x_ret` at 40960/40960 (100%).

The switch defaults and hook wiring are configuration/code facts, not a claim
of a valid 300-step result.

The clean trace-only Z+ run produced 42 `[MFNS] call ...` records. Whole,
non-transposed 2-D shapes `(3072, 5120)` and `(5120, 16384)` showed
`fp32_numerator=True, stash_hit=True`; sliced/transposed blocks showed
`fp32_numerator=False`; 3-D blocks had no float32 numerator classification.
This verifies hook invocation and class/stash selection. Parameter-level
逐字节 comparison was not completed because all three output disks were full.
The `lm_head`/attention backward items and shared `dy` accumulation boundary
remain unverified or unfinished.

## Not included and why

- The #148/#162/#163 merged changes, #180 draft, and #192 draft are not copied
  wholesale: this branch is deliberately limited to the requested Z+ patch
  and does not alter unrelated framework code.
- The #180 open items (`lm_head` 0.229 ULP, attention backward GEMM, and Fix2 /
  Fix4 switches) are not implemented by this branch.
- Any Fix5/framework scaling change is not included; no such change is part of
  the requested patch.
- No 300-step run result is included. The 17:41--18:05 pollution-period
  numbers are invalid.

## Conclusion

This branch contains the adapter-side Muon arithmetic and hook wiring plus the
two generic framework switches and the full declared difference list; it does
not contain the open backward-pass, Fix2/Fix4/Fix5, or other unrelated PR
implementations.
