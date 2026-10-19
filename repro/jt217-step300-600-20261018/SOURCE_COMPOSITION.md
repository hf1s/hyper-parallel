# Source composition

The reproduction does **not** consist of a YAML file alone.

## Base and exact source head

```text
base: a1e6c222eab097c9609d6e744a3c5f98aeed6bdc
head: fe16c866cba6f40e5fea432e8147e7446fbf1e7f
```

The exact source tree is the branch root in the Git worktree. The Huawei runtime bundle copies that same tree into its sibling `source/` directory.

## Tree delta from base

`git diff --name-status a1e6c222 fe16c866` contains 19 files:

### Muon storage/update compatibility

```text
hyper_parallel/core/optimizer/muon.py
hyper_parallel/models/jt_deepseek_v3/adapter/runtime/jt_optimizer.py
hyper_parallel/models/jt_deepseek_v3/adapter/runtime/muon_layout.py
tests/ut/core/optimizer/test_muon.py
tests/ut/model_integration/test_jt_muon_fused_matrices.py
```

These are the product-side Muon write-back and fused/logical matrix contracts represented by the PR213/PR243 tree-equivalence review.

### Explicit MF archive recipe

The compatibility implementation is now placed in the standard JT module
families. `mf_archive/` retains only documentation, the original recipe, and
thin legacy import markers; the active YAML targets do not import those
markers.

```text
hyper_parallel/components/losses/mf_archive.py
hyper_parallel/models/jt_deepseek_v3/archived_gradient_layout.json
hyper_parallel/models/jt_deepseek_v3/archived_layers.py
hyper_parallel/models/jt_deepseek_v3/archived_model.py
hyper_parallel/models/jt_deepseek_v3/archived_reductions.py
hyper_parallel/models/jt_deepseek_v3/archived_run.py
hyper_parallel/models/jt_deepseek_v3/legacy_gemm.py
hyper_parallel/models/jt_deepseek_v3/adapter/runtime/archived_optimizer.py
hyper_parallel/trainer/archived.py
hyper_parallel/trainer/text_trainer.py
hyper_parallel/models/jt_deepseek_v3/mf_archive/README.md
hyper_parallel/models/jt_deepseek_v3/mf_archive/__init__.py
hyper_parallel/models/jt_deepseek_v3/mf_archive/gradient_layout.json
hyper_parallel/models/jt_deepseek_v3/mf_archive/layers.py
hyper_parallel/models/jt_deepseek_v3/mf_archive/legacy_gemm.py
hyper_parallel/models/jt_deepseek_v3/mf_archive/losses.py
hyper_parallel/models/jt_deepseek_v3/mf_archive/model.py
hyper_parallel/models/jt_deepseek_v3/mf_archive/optimizer.py
hyper_parallel/models/jt_deepseek_v3/mf_archive/recipe.yaml
hyper_parallel/models/jt_deepseek_v3/mf_archive/reductions.py
hyper_parallel/models/jt_deepseek_v3/mf_archive/run.py
hyper_parallel/models/jt_deepseek_v3/mf_archive/training.py
tests/ut/model_integration/test_jt_mf_archive.py
```

These files implement the fixed compatibility recipe selected by
`hp-archive200-r2.yaml`. They are not the default product path and do not imply
that the old MF behavior is generally correct.

## Not included

The historical `/alignment/phase12` runtime factor chain is not imported by this bundle. There is no `P12_*` environment setup, no `vocabulary_both_builder`, no `moe_backward_builder`, and no phase12 factor path in `run-hp.sh`.
