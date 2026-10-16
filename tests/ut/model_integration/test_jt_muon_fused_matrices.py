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

"""Check that storage fusion preserves the JT optimizer's logical matrices."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.models.jt_deepseek_v3.adapter.runtime import jt_optimizer
from tests.common.mark_utils import arg_mark


def _builder_options(**options):
    """Capture the actual model adapter contract without distributed allocation."""
    config = SimpleNamespace(q_lora_rank=3, kv_lora_rank=2, qk_rope_head_dim=2)
    model = SimpleNamespace(config=config)
    with patch.object(jt_optimizer, 'Muon') as builder:
        builder.return_value.get_optimizer.return_value.chained_optimizers = [MagicMock()]
        jt_optimizer.build_optimizer(model=model, qk_clip_threshold=100.0,
                                     muon_config=options.pop('muon_config', {}), adamw_config={}, **options)
    return builder.call_args.kwargs['muon_config']


def _compute_updates(parameters, inputs, transform=None):
    """Run native shape routing, NS and logical-shape scaling on CPU tensors."""
    optimizer = object.__new__(Muon)
    optimizer.reshape_fn = None
    optimizer.ns_transform_fn = transform
    optimizer.zeropower_fn = None
    group = {'lr': .0001, 'matched_adamw_rms': .2, 'ns_steps': 5, 'ns_variant': 'legacy',
             'ns_coefficients': None, 'ns_epsilon': 1e-7,
             'zero_rms_scale_mode': 'zero', 'apply_lr_in_update': False}
    return optimizer._compute_batched_ns_updates(  # pylint: disable=protected-access
        parameters, inputs, group, no_shard=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
class TestJTFusedMuonMatrices(unittest.TestCase):
    """Compare fused storage with separately owned projection parameters."""

    def test_attention_storage_fusion_preserves_updates(self):
        """Fusing Q and latent-KV parameters must not change their NS updates."""
        generator = torch.Generator().manual_seed(72)
        inputs = [torch.randn(3, 5, generator=generator), torch.randn(4, 5, generator=generator)]
        separate = [torch.nn.Parameter(torch.zeros_like(value)) for value in inputs]
        expected = _compute_updates(separate, dict(zip(separate, [value.clone() for value in inputs])))
        fused = torch.nn.Parameter(torch.zeros(7, 5))
        fused.model_name = 'model.layers.0.self_attn.linear_qkv.weight'
        actual = _compute_updates([fused], {fused: torch.cat(inputs)}, _builder_options().get('ns_transform_fn'))
        torch.testing.assert_close(actual[fused], torch.cat([expected[p] for p in separate]), rtol=0, atol=0)

    def test_expert_storage_fusion_preserves_updates(self):
        """Fused gate/up must match separate logical expert projections."""
        generator = torch.Generator().manual_seed(91)
        inputs = [torch.randn(2, 3, 5, generator=generator), torch.randn(2, 3, 5, generator=generator)]
        separate = [torch.nn.Parameter(torch.zeros_like(value)) for value in inputs]
        expected = _compute_updates(separate, dict(zip(separate, [value.clone() for value in inputs])))
        fused = torch.nn.Parameter(torch.zeros(2, 6, 5))
        fused.model_name = 'mtp.layers.0.transformer_layer.mlp.experts.gate_up_proj'
        actual = _compute_updates([fused], {fused: torch.cat(inputs, dim=1)},
                                 _builder_options().get('ns_transform_fn'))
        torch.testing.assert_close(actual[fused], torch.cat([expected[p] for p in separate], dim=1), rtol=0, atol=0)

    def test_historical_opt_out_preserves_custom_callback(self):
        """Historical and caller-defined recipes remain explicitly selectable."""
        custom = MagicMock()
        supplied = {'ns_transform_fn': custom}
        result = _builder_options(split_fused_matrices=False, muon_config=supplied)
        self.assertIs(result['ns_transform_fn'], custom)
        self.assertEqual(supplied, {'ns_transform_fn': custom})
        self.assertNotIn('ns_transform_fn', _builder_options(split_fused_matrices=False))

    def test_conflicting_callbacks_fail_before_optimizer_construction(self):
        """A model policy must not silently replace a caller's existing recipe."""
        for option in ('reshape_fn', 'ns_transform_fn'):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, 'split_fused_matrices=False'):
                _builder_options(muon_config={option: MagicMock()})

    def test_invalid_policy_and_fused_shapes_are_rejected(self):
        """The public boundary rejects accidental truthy strings and wrong layouts."""
        with self.assertRaisesRegex(ValueError, 'boolean'):
            _builder_options(split_fused_matrices='true')
        transform = _builder_options()['ns_transform_fn']
        cases = [('model.layers.0.self_attn.linear_qkv.weight', torch.zeros(8, 5)),
                 ('model.layers.0.mlp.experts.gate_up_proj', torch.zeros(2, 5, 3))]
        for name, value in cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Unexpected fused'):
                transform(name, value)
        self.assertIsNone(transform('model.layers.0.self_attn.q_b_proj.weight', torch.zeros(7, 5)))

    def test_invalid_restore_does_not_write_output(self):
        """Reject incomplete NS results before mutating the destination."""
        transform = _builder_options()['ns_transform_fn'](
            'model.layers.0.self_attn.linear_qkv.weight', torch.zeros(7, 5))
        output = torch.ones(7, 5)
        with self.assertRaisesRegex(ValueError, 'pieces do not match'):
            transform.restore([torch.zeros(3, 5)], output)
        torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=0)
