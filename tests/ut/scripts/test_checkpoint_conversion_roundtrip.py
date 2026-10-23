# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Exercise real checkpoint I/O with CPU tensors and mocked rank communication."""

import hashlib
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from safetensors.numpy import save_file

from hyper_parallel.components.checkpoint.dcp_checkpointer import DistributedCheckpointer, initialize_optimizer_state
from hyper_parallel.components.optim.lr_scheduler import MultiLRScheduler
from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.data.batching import FixedBatchDataLoader
from hyper_parallel.data.parallel.batch_sampler import build_dataset_batch_sampler
from scripts.checkpoint_conversion.convert_mf_ckpt_to_hp import (
    convert_model, convert_optimizer, load_checkpoint_input, load_safetensors, write_artifacts,
)
from scripts.checkpoint_conversion.materialize_hp_native_checkpoint import load_optimizer_npz, set_nested_cursor
from scripts.checkpoint_conversion.manifest_input import load_conversion_manifest
from scripts.checkpoint_conversion.resume_state import validate_resume_state


class _SmallModel(torch.nn.Module):
    """Use three actual conversion names without requiring the training stack."""

    def __init__(self) -> None:
        """Create the three parameters addressed by the source fixture."""
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(3, 2, dtype=torch.bfloat16)
        self.model.norm = torch.nn.LayerNorm(2, bias=False, dtype=torch.bfloat16)
        self.lm_head = torch.nn.Linear(2, 3, bias=False, dtype=torch.bfloat16)


def _training_objects():
    model = _SmallModel()
    leaf = torch.optim.AdamW(model.parameters(), lr=.003, betas=(.8, .95), eps=1e-8, weight_decay=.01)
    optimizer = Float16OptimizerWithFloat16Params(ChainedOptimizer(model, {"adamw": leaf}), model)
    scheduler = MultiLRScheduler(optimizer, "constant", 400, {"lr": .003, "lr_warmup_steps": 0}).get_lr_scheduler()
    sampler = build_dataset_batch_sampler(total_samples=500, micro_batch_size=1, global_batch_size=1,
                                          dp_rank=0, dp_world_size=1)
    loader = FixedBatchDataLoader(list(range(500)), batch_sampler=sampler, num_workers=0)
    return model, leaf, optimizer, scheduler, loader


def _gather_one(output, value, *args, **kwargs):
    del args, kwargs
    output[0] = value


class TestCheckpointConversionRoundtrip(unittest.TestCase):
    """Persist converted master weights, moments, counters, and data position."""

    def test_safetensors_to_native_reload_and_update_201(self):
        self._roundtrip(manifest_input=False)

    def test_manifest_to_native_reload_and_update_201(self):
        self._roundtrip(manifest_input=True)

    def test_mf_native_input_to_native_reload_and_update_201(self) -> None:
        """Carry verified native aliases through DCP, fresh objects, data and the next update."""
        self._roundtrip(manifest_input=False, native_input=True)

    def _roundtrip(self, *, manifest_input, native_input=False):
        source_weights = {
            "embedding.word_embeddings.weight": np.arange(6, dtype=np.float32).reshape(3, 2) / 10 + 1.0017,
            "decoder.final_layernorm.weight": np.array([.9131, 1.0037], np.float32),
            "output_layer.weight": np.arange(6, dtype=np.float32).reshape(3, 2) / 7 + .8931,
        }
        source = dict(source_weights)
        for name, value in source_weights.items():
            source[f"adam_m.{name}"] = np.full_like(value, .2)
            source[f"adam_v.{name}"] = np.full_like(value, .3)
        source.update(step_num=np.array(200, np.int64), global_batch_size=np.array(1, np.int64),
                      loss_scale=np.array(1, np.float32))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "source.safetensors"
            input_report = None
            if native_input:
                source.update(global_step=np.array([200], np.int32), scale_sense=np.array(1, np.float32),
                              lm_loss=np.array([.5], np.float32), mtp_loss=np.array([.25], np.float32),
                              aux_loss=np.array([.001], np.float32),
                              **{"embedding.embedding_dropout.seed": np.array(0, np.int64),
                                 "embedding.embedding_dropout.offset": np.array(0, np.int64)})
                save_file(source, str(checkpoint), metadata={"format": "ms", "remove_redundancy": "False"})
                loaded, input_report = load_checkpoint_input(checkpoint)
            else:
                save_file(source, str(checkpoint))
                loaded = load_safetensors(checkpoint)
            weights, moments = convert_model(loaded), convert_optimizer(loaded)
            offline = root / "offline"
            write_artifacts(loaded, weights, moments, offline,
                            source_checkpoint=checkpoint, input_report=input_report)
            metadata = json.loads((offline / "state.json").read_text())
            validate_resume_state(metadata, 200)
            self.assertEqual(metadata["source_checkpoint"], str(checkpoint.resolve()))

            manifest = None
            if manifest_input:
                offline = root / "manifest-input"
                offline.mkdir()
                np.savez(offline / "model.npz", **weights)
                fields = {"exp_avg": "adam_m", "exp_avg_sq": "adam_v"}
                named = {f"{name}::{field}": moments[f"{prefix}.{name}"]
                         for name in weights for field, prefix in fields.items()}
                np.savez(offline / "optimizer.npz", **named)
                manifest = {
                    "format_version": 1, "optimizer_type": "AdamW", "completed_steps": 200,
                    "optimizer_global_step": 200, "mapped_model_tensors": len(weights),
                    "optimizer_state_tensors": len(named),
                    "optimizer_manifest": [{"name": name, "use_muon": False,
                                            "state": {field: f"{name}::{field}" for field in fields}}
                                           for name in weights],
                    "sha256": {name: hashlib.sha256((offline / name).read_bytes()).hexdigest()
                               for name in ("model.npz", "optimizer.npz")},
                }
                (offline / "CONVERSION.json").write_text(json.dumps(manifest), encoding="utf8")
                manifest = load_conversion_manifest(offline, 200)

            model, _, optimizer, scheduler, loader = _training_objects()
            trainer = SimpleNamespace(base=SimpleNamespace(model=model, optimizer=optimizer))
            self.assertEqual(load_optimizer_npz(trainer, offline, 200, optimizer_manifest=manifest), (6, 6))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Detected call of.*lr_scheduler.step")
                for _ in range(200):
                    scheduler.step()
            loader_state = loader.state_dict()
            self.assertTrue(set_nested_cursor(loader_state, 200, 1))
            loader.load_state_dict(loader_state)
            extra = {"global_step": 200, "lr_scheduler": scheduler.state_dict(),
                     "train_dataloader": loader.state_dict()}
            native = str(root / "native" / "global_step_200")
            checkpointer = DistributedCheckpointer()

            # Only rank communication is mocked; planners, tensor serialization,
            # filesystem writes, and fresh-object reload run their real code.
            with (patch("torch.distributed.get_rank", return_value=0),
                  patch("torch.distributed.get_world_size", return_value=1),
                  patch("torch.distributed.barrier"),
                  patch("torch.distributed.all_gather_object", side_effect=_gather_one)):
                checkpointer.save(native, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                           "extra_state": extra}, global_step=200)
                fresh_model, fresh_leaf, fresh_optimizer, fresh_scheduler, fresh_loader = _training_objects()
                initialize_optimizer_state(fresh_optimizer)
                state = {"model": fresh_model.state_dict(), "optimizer": fresh_optimizer.state_dict()}
                checkpointer.load(native, state, extra_state_skeleton={"global_step": -1})
                fresh_model.load_state_dict(state["model"])
                fresh_optimizer.load_state_dict(state["optimizer"])

            self.assertTrue((Path(native) / ".metadata").is_file())
            self.assertEqual(state["extra_state"]["global_step"], 200)
            fresh_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
            fresh_loader.load_state_dict(state["extra_state"]["train_dataloader"])
            self.assertEqual(fresh_scheduler.state_dict(), scheduler.state_dict())
            self.assertEqual(fresh_scheduler.get_last_lr(), [.003])
            self.assertEqual(next(iter(fresh_loader)).tolist(), [200])
            for name, parameter in fresh_model.named_parameters():
                main = fresh_optimizer.optimizer_param_by_model_param[parameter]
                np.testing.assert_array_equal(main.detach().numpy(), weights[name])
                for prefix, field in (("adam_m", "exp_avg"), ("adam_v", "exp_avg_sq")):
                    np.testing.assert_array_equal(fresh_leaf.state[main][field].numpy(), moments[f"{prefix}.{name}"])
                self.assertEqual(fresh_leaf.state[main]["step"].item(), 200)
                torch.testing.assert_close(parameter, main.bfloat16(), rtol=0, atol=0)
                parameter.grad = torch.full_like(parameter, .125)
            fresh_optimizer.step()
            for name, parameter in fresh_model.named_parameters():
                main = fresh_optimizer.optimizer_param_by_model_param[parameter]
                first = moments[f"adam_m.{name}"].astype(np.float64) * .8 + .125 * .2
                second = moments[f"adam_v.{name}"].astype(np.float64) * .95 + .125**2 * .05
                expected = weights[name].astype(np.float64) * (1 - .003 * .01)
                expected -= .003 * first / (1 - .8**201) / (np.sqrt(second / (1 - .95**201)) + 1e-8)
                np.testing.assert_allclose(main.detach().numpy(), expected, rtol=3e-7, atol=3e-7)
                self.assertEqual(fresh_leaf.state[main]["step"].item(), 201)
                torch.testing.assert_close(parameter, main.bfloat16(), rtol=0, atol=0)
