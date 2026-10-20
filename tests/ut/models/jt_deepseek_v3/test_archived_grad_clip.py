# Copyright 2026 Huawei Technologies Co., Ltd.
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

"""Check the archived gradient clipping and its optimizer pre-step hook independently of saved outputs."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.models.jt_deepseek_v3.adapter.runtime import archived_grad_clip
from hyper_parallel.models.jt_deepseek_v3.adapter.runtime.archived_grad_clip import (
    clip_archived_gradients, mf_gradient, register_archived_clip,
)
from tests.common.mark_utils import arg_mark
from tests.ut.models.jt_deepseek_v3.test_jt_optimizer import replaced_model

# Native JT storage. The archived Q/KV down projections are fixed row ranges of
# ``linear_qkv`` (0:1536, 1536:2112); every other dimension is small. Logical
# sizes cover each archived SumX tiling: <= 5120 elements, exactly 40960, and
# > 5120 with a partial final tile.
NATIVE_SHAPES = {
    'model.embed_tokens.weight': (1024, 17),
    'model.layers.0.self_attn.linear_qkv.weight': (2112, 4),
    'model.layers.0.self_attn.q_a_layernorm.weight': (64,),
    'model.layers.0.self_attn.kv_a_layernorm.weight': (64,),
    'model.layers.0.self_attn.q_b_proj.weight': (16, 4),
    'model.layers.0.self_attn.kv_b_proj.weight': (16, 8),
    'model.layers.0.self_attn.o_proj.weight': (8, 16),
    'model.layers.0.mlp.gate_proj.weight': (16, 8),
    'model.layers.0.mlp.up_proj.weight': (16, 8),
    'model.layers.0.mlp.down_proj.weight': (8, 16),
    'model.layers.0.input_layernorm.weight': (64,),
    'model.layers.0.post_attention_layernorm.weight': (64,),
    'model.layers.1.mlp.gate.weight': (2, 32),
    'model.layers.1.mlp.experts.gate_up_proj': (2, 8, 16),
    'model.layers.1.mlp.experts.down_proj': (2, 16, 4),
    'model.layers.1.mlp.shared_experts.gate_proj.weight': (4, 16),
    'model.layers.1.mlp.shared_experts.up_proj.weight': (4, 16),
    'model.layers.1.mlp.shared_experts.down_proj.weight': (16, 4),
    'model.norm.weight': (64,),
    'lm_head.weight': (80, 512),
    'mtp.layers.0.eh_proj.weight': (8, 16),
    'mtp.layers.0.transformer_layer.mlp.gate.weight': (2, 32),
    'mtp.layers.0.final_layernorm.weight': (64,),
}

# Archived logical gradients in registration order; together they cover every
# native element above exactly once.
ARCHIVED_LAYOUT = [
    'embedding.word_embeddings.weight',
    'decoder.layers.0.self_attention.linear_proj.weight',
    'decoder.layers.0.self_attention.linear_q_down_proj.weight',
    'decoder.layers.0.self_attention.linear_q_up_proj.weight',
    'decoder.layers.0.self_attention.linear_kv_down_proj.weight',
    'decoder.layers.0.self_attention.linear_kv_up_proj.weight',
    'decoder.layers.0.mlp.linear_fc1.weight',
    'decoder.layers.0.mlp.linear_fc2.weight',
    'decoder.layers.1.mlp.router.weight',
    'decoder.layers.1.mlp.experts.weight1',
    'decoder.layers.1.mlp.experts.weight2',
    'decoder.layers.1.mlp.shared_experts.linear_fc1.weight',
    'decoder.layers.1.mlp.shared_experts.linear_fc2.weight',
    'mtp.layers.0.eh_proj.weight',
    'mtp.layers.0.transformer_layer.mlp.router.weight',
    'output_layer.weight',
    'decoder.layers.0.input_layernorm.weight',
    'decoder.layers.0.self_attention.q_layernorm.weight',
    'decoder.layers.0.self_attention.kv_layernorm.weight',
    'decoder.layers.0.pre_mlp_layernorm.weight',
    'decoder.final_layernorm.weight',
    'mtp.layers.0.final_layernorm.weight',
]

# Archived names whose gradient is one native gradient, unchanged.
RENAMED = {
    'embedding.word_embeddings.weight': 'model.embed_tokens.weight',
    'output_layer.weight': 'lm_head.weight',
    'decoder.final_layernorm.weight': 'model.norm.weight',
    'decoder.layers.0.self_attention.linear_proj.weight': 'model.layers.0.self_attn.o_proj.weight',
    'decoder.layers.0.self_attention.linear_q_up_proj.weight': 'model.layers.0.self_attn.q_b_proj.weight',
    'decoder.layers.0.self_attention.linear_kv_up_proj.weight': 'model.layers.0.self_attn.kv_b_proj.weight',
    'decoder.layers.0.self_attention.q_layernorm.weight': 'model.layers.0.self_attn.q_a_layernorm.weight',
    'decoder.layers.0.self_attention.kv_layernorm.weight': 'model.layers.0.self_attn.kv_a_layernorm.weight',
    'decoder.layers.0.input_layernorm.weight': 'model.layers.0.input_layernorm.weight',
    'decoder.layers.0.pre_mlp_layernorm.weight': 'model.layers.0.post_attention_layernorm.weight',
    'decoder.layers.0.mlp.linear_fc2.weight': 'model.layers.0.mlp.down_proj.weight',
    'decoder.layers.1.mlp.router.weight': 'model.layers.1.mlp.gate.weight',
    'decoder.layers.1.mlp.shared_experts.linear_fc2.weight': 'model.layers.1.mlp.shared_experts.down_proj.weight',
    'mtp.layers.0.eh_proj.weight': 'mtp.layers.0.eh_proj.weight',
    'mtp.layers.0.transformer_layer.mlp.router.weight': 'mtp.layers.0.transformer_layer.mlp.gate.weight',
    'mtp.layers.0.final_layernorm.weight': 'mtp.layers.0.final_layernorm.weight',
}


def register(root: torch.nn.Module, name: str, parameter: torch.nn.Parameter) -> None:
    """Register ``parameter`` under its dotted name, creating parent modules."""
    *path, leaf = name.split('.')
    owner = root
    for part in path:
        if not hasattr(owner, part):
            owner.add_module(part, torch.nn.Module())
        owner = getattr(owner, part)
    owner.register_parameter(leaf, parameter)


def jt_storage(seed: int) -> tuple[torch.nn.Module, dict[str, torch.nn.Parameter]]:
    """Build a module with native JT parameter names holding random FP32 gradients."""
    generator = torch.Generator().manual_seed(seed)
    root = torch.nn.Module()
    parameters = {}
    for name, shape in NATIVE_SHAPES.items():
        parameter = torch.nn.Parameter(torch.zeros(shape))
        parameter.grad = torch.randn(shape, generator=generator)
        register(root, name, parameter)
        parameters[name] = parameter
    return root, parameters


def write_layout(directory: str, layout: list[str]) -> Path:
    """Write ``layout`` in the versioned archived JSON format."""
    path = Path(directory) / 'gradient_layout.json'
    path.write_text(json.dumps([{'name': name} for name in layout]), encoding='utf-8')
    return path


def double_norm(parameters: dict[str, torch.nn.Parameter]) -> float:
    """FP64 L2 norm over every native gradient."""
    return math.sqrt(sum(parameter.grad.double().square().sum().item() for parameter in parameters.values()))


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
class TestArchivedGradientClipping(unittest.TestCase):
    """Archived logical gradient layout, FP32 norm and clipping stage contracts."""

    def test_renamed_gradients_are_native_values(self):
        """Renamed archived tensors expose the live native gradient values unchanged."""
        _, parameters = jt_storage(seed=1)
        for archived, native in RENAMED.items():
            with self.subTest(archived=archived):
                torch.testing.assert_close(mf_gradient(parameters, archived), parameters[native].grad, rtol=0, atol=0)

    def test_fused_storage_is_split_into_archived_matrices(self):
        """Q/KV rows, gate/up row interleave and expert transposes match per-row references."""
        _, parameters = jt_storage(seed=2)
        grad = {name: parameter.grad for name, parameter in parameters.items()}
        qkv = grad['model.layers.0.self_attn.linear_qkv.weight']
        attention = 'decoder.layers.0.self_attention.'
        torch.testing.assert_close(mf_gradient(parameters, attention + 'linear_q_down_proj.weight'),
                                   qkv[:1536], rtol=0, atol=0)
        torch.testing.assert_close(mf_gradient(parameters, attention + 'linear_kv_down_proj.weight'),
                                   qkv[1536:], rtol=0, atol=0)
        for archived, prefix in [('decoder.layers.0.mlp.linear_fc1.weight', 'model.layers.0.mlp.'),
                                 ('decoder.layers.1.mlp.shared_experts.linear_fc1.weight',
                                  'model.layers.1.mlp.shared_experts.')]:
            gate, up = grad[prefix + 'gate_proj.weight'], grad[prefix + 'up_proj.weight']
            expected = torch.stack([row for pair in zip(gate, up) for row in pair])
            torch.testing.assert_close(mf_gradient(parameters, archived), expected, rtol=0, atol=0)
        # gate_up_proj is [expert, 2 * intermediate, hidden]; weight1 row e * hidden + h is its column h.
        gate_up = grad['model.layers.1.mlp.experts.gate_up_proj']
        experts, _, hidden = gate_up.shape
        expected = torch.stack([gate_up[e, :, h] for e in range(experts) for h in range(hidden)])
        torch.testing.assert_close(mf_gradient(parameters, 'decoder.layers.1.mlp.experts.weight1'),
                                   expected, rtol=0, atol=0)
        # down_proj is [expert, hidden, intermediate]; weight2 row e * intermediate + i is its column i.
        down = grad['model.layers.1.mlp.experts.down_proj']
        intermediate = down.shape[2]
        expected = torch.stack([down[e, :, i] for e in range(experts) for i in range(intermediate)])
        torch.testing.assert_close(mf_gradient(parameters, 'decoder.layers.1.mlp.experts.weight2'),
                                   expected, rtol=0, atol=0)

    def test_norm_and_clipped_gradients_match_double_precision(self):
        """The archived FP32 schedule computes the full gradient norm and clips every gradient."""
        with tempfile.TemporaryDirectory() as directory:
            layout_path = write_layout(directory, ARCHIVED_LAYOUT)
            for ratio in (0.25, 4.0):
                with self.subTest(max_norm_ratio=ratio):
                    model, parameters = jt_storage(seed=3)
                    originals = {name: parameter.grad.clone() for name, parameter in parameters.items()}
                    expected_norm = double_norm(parameters)
                    max_norm = expected_norm * ratio
                    norm = clip_archived_gradients(model, max_norm, layout_path=layout_path)
                    self.assertEqual(norm.dtype, torch.float32)
                    torch.testing.assert_close(norm.double(), torch.tensor(expected_norm, dtype=torch.float64),
                                               rtol=1e-6, atol=0)
                    coefficient = min(max_norm / (expected_norm + 1e-6), 1.0)
                    for name, parameter in parameters.items():
                        if coefficient == 1.0:
                            # A clamped coefficient multiplies by exactly one.
                            torch.testing.assert_close(parameter.grad, originals[name], rtol=0, atol=0)
                        else:
                            torch.testing.assert_close(parameter.grad.double(), originals[name].double() * coefficient,
                                                       rtol=1e-6, atol=0)

    def test_shipped_layout_reads_every_jt_parameter(self):
        """Every archived name resolves on the JT model, and together they read all its gradients."""
        model = replaced_model(torch.float32)
        parameters = dict(model.named_parameters())
        for index, parameter in enumerate(parameters.values(), start=1):
            parameter.grad = torch.full_like(parameter, float(index))
        layout = json.loads(archived_grad_clip.GRADIENT_LAYOUT_PATH.read_text(encoding='utf-8'))
        seen = set()
        for entry in layout:
            seen.update(mf_gradient(parameters, entry['name']).unique().tolist())
        self.assertEqual(seen, set(range(1, len(parameters) + 1)))

    def test_pre_hook_clips_once_per_step_before_any_leaf_update(self):
        """Each optimizer step clips exactly once, before the first leaf updates, matching clip-then-step."""

        def chained(module: torch.nn.Module, parameters: dict[str, torch.nn.Parameter]) -> ChainedOptimizer:
            """Chain two SGD leaves over disjoint halves of the native parameters."""
            names = list(parameters)
            leaves = {'first': torch.optim.SGD([parameters[name] for name in names[::2]], lr=1.0),
                      'second': torch.optim.SGD([parameters[name] for name in names[1::2]], lr=1.0)}
            return ChainedOptimizer(module, optimizers=leaves, flatten=True)

        with tempfile.TemporaryDirectory() as directory:
            layout_path = write_layout(directory, ARCHIVED_LAYOUT)
            model, parameters = jt_storage(seed=6)
            reference, reference_parameters = jt_storage(seed=6)
            max_norm = double_norm(parameters) / 4
            optimizer, reference_optimizer = chained(model, parameters), chained(reference, reference_parameters)
            register_archived_clip(optimizer, model, max_norm, layout_path=layout_path)
            calls = []
            snapshot = {}

            def recording_clip(*args: Any, **kwargs: Any) -> torch.Tensor:
                """Record whether every parameter still holds its pre-step value when the archived clip runs."""
                calls.append(all(torch.equal(parameter, snapshot[name]) for name, parameter in parameters.items()))
                return clip_archived_gradients(*args, **kwargs)

            for step in range(2):
                snapshot.update({name: parameter.detach().clone() for name, parameter in parameters.items()})
                with patch.object(archived_grad_clip, 'clip_archived_gradients', recording_clip):
                    optimizer.step()
                self.assertEqual(calls, [True] * (step + 1))
                clip_archived_gradients(reference, max_norm, layout_path=layout_path)
                reference_optimizer.step()
                for name, parameter in parameters.items():
                    torch.testing.assert_close(parameter.grad, reference_parameters[name].grad, rtol=0, atol=0)
                    torch.testing.assert_close(parameter.data, reference_parameters[name].data, rtol=0, atol=0)
