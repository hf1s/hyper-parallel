"""Read-only final-step taps for shared decoder-gradient boundaries."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
TARGETS = {"model.layers.1.mlp": "dy_mlp", "model.norm": "dx_model_norm", "mtp.layers.0.hnorm": "dx_mtp_hnorm"}
def _clean(name):
    name = name.replace("_checkpoint_wrapped_module.", "")
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part in ("model", "mtp"): return ".".join(parts[i:])
    return name
def install(model, directory, step=301):
    import torch
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    state = {key: {"calls": 0, "grad_calls": 0, "captured": False} for key in TARGETS.values()}
    handles = []; found = {}
    for name, module in model.named_modules():
        clean = _clean(name)
        if clean in TARGETS and clean not in found: found[clean] = module
    print(f"[DYTAP] targets={sorted(found)}", flush=True)
    def save(key, value, rows=None):
        if value is None or not torch.is_tensor(value): return
        tensor = value.detach(); source_dtype = str(tensor.dtype); shape = list(tensor.shape)
        if rows is not None and tensor.ndim >= 2: tensor = tensor[:rows]
        array = np.ascontiguousarray(tensor.float().cpu().numpy())
        np.save(str(directory / f"{key}.npy"), array)
        state[key].update({"captured": True, "source_dtype": source_dtype, "shape": shape,
                           "saved_shape": list(array.shape), "min": float(array.min()) if array.size else None,
                           "max": float(array.max()) if array.size else None})
    def grad_hook(key, grad, rows=None):
        state[key]["grad_calls"] += 1
        if state[key]["grad_calls"] == step: save(key, grad, rows)
        return None
    for name, module in found.items():
        key = TARGETS[name]
        def forward_hook(_module, inputs, output, *, key=key):
            state[key]["calls"] += 1
            # NPU graph replay may run this Python module hook only during capture; its
            # tensor hook is replayed each step, so count gradients rather than forwards.
            if state[key]["calls"] != 1: return
            if key == "dy_mlp" and inputs and torch.is_tensor(inputs[0]) and inputs[0].requires_grad:
                handles.append(inputs[0].register_hook(lambda grad, key=key: grad_hook(key, grad)))
            else:
                values = output if isinstance(output, tuple) else (output,)
                if values and torch.is_tensor(values[0]) and values[0].requires_grad:
                    handles.append(values[0].register_hook(lambda grad, key=key: grad_hook(key, grad, 64)))
        handles.append(module.register_forward_hook(forward_hook))
    missing = sorted(set(TARGETS) - set(found))
    if missing: print(f"[DYTAP] missing={missing}", flush=True)
    def finish():
        (directory / "index.json").write_text(json.dumps({"step": step, "targets": TARGETS, "state": state}, indent=2) + "\n")
        print(f"[DYTAP] wrote {json.dumps(state, sort_keys=True)}", flush=True)
    import atexit; atexit.register(finish)
    return handles
