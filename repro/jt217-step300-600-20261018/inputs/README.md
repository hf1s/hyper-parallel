# Required Huawei-129 inputs

The runner reads `paths.env` inside the container. These paths are explicit and must exist before running:

```text
/workspace/mf-weights/full-state-step300
/workspace/mf-weights/native-step300/global_step_300
/workspace/datasets/sft_data_4k_from_256k
```

The dataset is the MF reference Arrow dataset. `sft_4k_from_256k` is a different copy and is not accepted for this reproduction.
