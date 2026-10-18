# Exact archive step300 continuation result

Run contract:

- HP CANN9.2.0-beta.2, one NPU;
- source head `fe16c866`;
- exact archive config `hp-archive200-r2.yaml`;
- model/optimizer/data restored at step300;
- MF reference dataset used;
- steps301–600, exit code 0;
- audit-only checksum callbacks disabled.

Observed comparison:

```text
total mean absolute error: 0.0041
total max absolute error: 0.1028
total errors > 0.005: 68/300
first total error > 0.005: step438
```

`summary.md` is produced by `/workspace/tools/compare_hp_mf.py`. PNG names are stable and semantic: `loss-delta.png`, `loss-smoothed.png`, `early.png`.
