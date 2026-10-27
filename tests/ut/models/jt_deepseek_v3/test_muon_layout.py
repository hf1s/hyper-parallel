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
"""Logical matrix boundaries and reversible row grouping for JT Muon."""

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.models.jt_deepseek_v3.adapter.runtime import jt_optimizer
from hyper_parallel.models.jt_deepseek_v3.adapter.runtime.muon_layout import JTMuonLayout
from tests.common.mark_utils import arg_mark


def layout_config(**overrides: Any) -> SimpleNamespace:
    """Use unequal Q/KV channel sizes so wrong grouping cannot pass accidentally."""
    values = {"q_lora_rank": 3, "kv_lora_rank": 2, "qk_nope_head_dim": 2,
              "qk_rope_head_dim": 1, "v_head_dim": 2, "num_attention_heads": 2}
    values.update(overrides)
    return SimpleNamespace(**values)


class TestJTMuonLayout(unittest.TestCase):
    """Test storage-sharing fused blocks and gathered whole-head-group matrices."""

    def setUp(self) -> None:
        """Build a complete two-head layout without any distributed setup."""
        self.layout = JTMuonLayout.from_config(layout_config())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_qkv_views_keep_q_down_and_two_kv_sections_distinct(self):
        """
        Feature: Fused QKV logical boundaries.
        Description: Modify one KV section through a storage-sharing view.
        Expectation: Q-down and the other KV section remain unchanged.
        """
        value = torch.arange(24).view(6, 4)
        pieces = self.layout.reshape_fn("model.layers.0.self_attn.linear_qkv.weight", value)
        self.assertEqual([tuple(piece.shape) for piece in pieces], [(3, 4), (2, 4), (1, 4)])
        for piece in pieces:
            self.assertEqual(piece.untyped_storage().data_ptr(), value.untyped_storage().data_ptr())
        pieces[1].fill_(-1)
        self.assertTrue(torch.equal(value[3:5], torch.full((2, 4), -1)))
        self.assertTrue(torch.equal(value[:3], torch.arange(12).view(3, 4)))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_expert_gate_up_splits_within_each_expert(self):
        """
        Feature: Expert gate/up partitioning.
        Description: Split two experts with distinct row values.
        Expectation: Gate/up split within each expert without mixing expert identities.
        """
        value = torch.arange(48).view(2, 6, 4)
        pieces = self.layout.reshape_fn("mtp.layers.0.transformer_layer.mlp.experts.gate_up_proj", value)
        self.assertEqual([tuple(piece.shape) for piece in pieces], [(2, 3, 4), (2, 3, 4)])
        self.assertEqual(pieces[0][:, 0, 0].tolist(), [0, 24])
        self.assertEqual(pieces[1][:, 0, 0].tolist(), [12, 36])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_dense_shared_and_mtp_gate_up_use_row_halves(self):
        """
        Feature: Dense and shared gate/up partitioning.
        Description: Apply the layout to trunk and MTP parameter names.
        Expectation: Both logical row halves match their original gate/up values.
        """
        value = torch.arange(24).view(6, 4)
        for name in ("model.layers.0.mlp.linear_fc1.weight",
                     "model.layers.1.mlp.shared_experts.linear_fc1.weight",
                     "mtp.layers.0.transformer_layer.mlp.shared_experts.linear_fc1.weight"):
            with self.subTest(name=name):
                first, second = self.layout.reshape_fn(name, value)
                self.assertTrue(torch.equal(first, torch.arange(12).view(3, 4)))
                self.assertTrue(torch.equal(second, torch.arange(12, 24).view(3, 4)))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_q_up_groups_matching_channels_across_heads_and_restores(self):
        """
        Feature: Periodic Q-up transform.
        Description: Gather unequal NoPE and RoPE sections across two heads.
        Expectation: Logical matrices contain all heads and restore to the original row positions.
        """
        value = torch.arange(12).view(6, 2)
        transform = self.layout.ns_transform_fn("model.layers.0.self_attn.q_b_proj.weight", value)
        self.assertTrue(torch.equal(transform.tensors[0], torch.tensor([[0, 1], [2, 3], [6, 7], [8, 9]])))
        self.assertTrue(torch.equal(transform.tensors[1], torch.tensor([[4, 5], [10, 11]])))
        output = torch.empty_like(value)
        transform.restore([torch.full((4, 2), 7), torch.full((2, 2), 9)], output)
        self.assertEqual(output[:, 0].tolist(), [7, 7, 9, 7, 7, 9])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_kv_up_uses_nope_and_value_groups(self):
        """
        Feature: Periodic KV-up transform.
        Description: Gather key/value channels and restore their input values.
        Expectation: All head channels are covered and the complete tensor round-trips exactly.
        """
        value = torch.arange(16).view(8, 2)
        transform = self.layout.ns_transform_fn("mtp.layers.0.transformer_layer.self_attn.kv_b_proj.weight", value)
        self.assertEqual(transform.tensors[0][:, 0].tolist(), [0, 2, 8, 10])
        self.assertEqual(transform.tensors[1][:, 0].tolist(), [4, 6, 12, 14])
        output = torch.empty_like(value)
        transform.restore(transform.tensors, output)
        self.assertTrue(torch.equal(output, value))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_q_up_scale_uses_the_last_logical_matrix(self):
        """
        Feature: MF Q-up shared RMS scale.
        Description: Transform unequal NoPE and RoPE groups across two heads.
        Expectation: Both NS matrices use the final RoPE matrix dimensions for scaling.
        """
        transform = self.layout.ns_transform_fn("model.layers.0.self_attn.q_b_proj.weight", torch.ones(6, 2))
        self.assertEqual([tuple(value.shape) for value in transform.tensors], [(4, 2), (2, 2)])
        self.assertEqual(transform.scale_shapes, [(2, 2), (2, 2)])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_fused_q_down_keeps_its_separate_mf_scale_group(self):
        """
        Feature: Fused QKV contains two MF parameter scale groups.
        Description: Use Q-down rows larger than the hidden dimension and unequal KV sections.
        Expectation: Q-down keeps its shape while both KV sections share the last KV shape.
        """
        layout = JTMuonLayout.from_config(layout_config(q_lora_rank=7, kv_lora_rank=3))
        transform = layout.ns_transform_fn("model.layers.0.self_attn.linear_qkv.weight", torch.ones(11, 2))
        self.assertEqual(transform.scale_shapes, [(7, 2), (1, 2), (1, 2)])
        output = torch.empty(11, 2)
        transform.restore([torch.full((7, 2), 1), torch.full((3, 2), 2), torch.full((1, 2), 3)], output)
        self.assertEqual(output[:, 0].tolist(), [1] * 7 + [2] * 3 + [3])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_expert_scale_shape_excludes_the_expert_batch_dimension(self):
        """
        Feature: Expert logical RMS dimensions.
        Description: Use more experts than either individual matrix dimension.
        Expectation: Scale shapes contain only matrix rows and columns, independent of expert count.
        """
        transform = self.layout.ns_transform_fn("model.layers.1.mlp.experts.gate_up_proj", torch.ones(8, 6, 2))
        self.assertEqual(transform.scale_shapes, [(3, 2), (3, 2)])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_unmatched_parameter_keeps_native_layout(self):
        """
        Feature: Unmatched parameter identity.
        Description: Pass an output projection through both layout callbacks.
        Expectation: Its original storage is retained and no periodic transform is produced.
        """
        value = torch.ones(4, 4)
        name = "model.layers.0.self_attn.o_proj.weight"
        self.assertIs(self.layout.reshape_fn(name, value)[0], value)
        self.assertIsNone(self.layout.ns_transform_fn(name, value))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_invalid_dimensions_or_incomplete_head_groups_fail(self):
        """
        Feature: Layout validation.
        Description: Supply invalid dimensions and incomplete logical matrices.
        Expectation: A descriptive ValueError is raised before Newton-Schulz runs.
        """
        for dimension in (0, -1, True, None):
            with self.subTest(dimension=dimension), self.assertRaisesRegex(ValueError, "positive integer"):
                JTMuonLayout.from_config(layout_config(num_attention_heads=dimension))
        with self.assertRaisesRegex(ValueError, "periodic attention shape"):
            self.layout.ns_transform_fn("model.layers.0.self_attn.q_b_proj.weight", torch.ones(5, 2))
        with self.assertRaisesRegex(ValueError, "fused QKV shape"):
            self.layout.reshape_fn("model.layers.0.self_attn.linear_qkv.weight", torch.ones(5, 2))


class TestJTMuonBuilderLayout(unittest.TestCase):
    """The recipe builder supplies callbacks without replacing the core optimizer."""

    def setUp(self) -> None:
        """Attach the minimal JT configuration used by the builder."""
        self.model = torch.nn.Module()
        self.model.config = layout_config()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_mindformers_layout_binds_both_callbacks_and_preserves_input_options(self):
        """
        Feature: JT builder callback binding.
        Description: Enable the MindFormers matrix layout on the existing builder.
        Expectation: Both callbacks share one layout, caller options stay intact and one JT post-hook remains.
        """
        options = {"lr": 1e-4}
        leaf = MagicMock()
        with patch.object(jt_optimizer, "Muon") as factory:
            factory.return_value.get_optimizer.return_value.chained_optimizers = [leaf]
            jt_optimizer.build_optimizer(model=self.model, qk_clip_threshold=100,
                                         muon_layout="mindformers", muon_config=options, adamw_config={})
        selected = factory.call_args.kwargs["muon_config"]
        self.assertEqual(options, {"lr": 1e-4})
        self.assertIsInstance(selected["reshape_fn"].__self__, JTMuonLayout)
        self.assertIs(selected["reshape_fn"].__self__, selected["ns_transform_fn"].__self__)
        leaf.register_step_post_hook.assert_called_once()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_native_layout_preserves_existing_builder_contract(self):
        """
        Feature: JT builder default compatibility.
        Description: Build without selecting an alternative logical layout.
        Expectation: Native Muon options reach the existing factory unchanged.
        """
        with patch.object(jt_optimizer, "Muon") as factory:
            jt_optimizer.build_optimizer(model=self.model, qk_clip_threshold=100,
                                         muon_config={"lr": 1e-4}, adamw_config={})
        self.assertEqual(factory.call_args.kwargs["muon_config"], {"lr": 1e-4})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_unknown_layout_and_callback_conflicts_fail(self):
        """
        Feature: JT builder configuration validation.
        Description: Select an invalid layout or combine the built-in layout with explicit callbacks.
        Expectation: Invalid or ambiguous callback ownership raises ValueError.
        """
        with self.assertRaisesRegex(ValueError, "muon_layout"):
            jt_optimizer.build_optimizer(model=self.model, qk_clip_threshold=100,
                                         muon_layout="other", muon_config={}, adamw_config={})
        for name in ("reshape_fn", "ns_transform_fn", "muon_reshape_fn", "muon_ns_transform_fn"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "owns"):
                jt_optimizer.build_optimizer(model=self.model, qk_clip_threshold=100, muon_layout="mindformers",
                                             muon_config={name: lambda *_: None}, adamw_config={})
