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
"""What each Xing4.0 MTP depth embeds, scores, fuses and positions itself at.

Depth ``k`` keeps the whole sequence: it embeds ``input_ids`` shifted left by ``k`` and padded
with token 0, and scores the targets shifted the same way and padded with ``IGNORE_INDEX``.
Consecutive, traceable token ids make every expectation an exact list: row ``r`` holds
``10 * (r + 1) + i`` at position ``i`` and the next-token target is that id plus one, so moving
the shift by one position changes these lists rather than only the loss value.
"""

import unittest
from collections.abc import Callable
from unittest.mock import patch

import torch

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.models._transformers.auto_model import _model_init_context
from hyper_parallel.models._transformers.model_builder import _initialize_model_weights
from hyper_parallel.models.xing40.adapter.model import Xing40ForCausalLM, Xing40MoE
from hyper_parallel.models.xing40.configuration_xing4_0 import Xing4_0Config
from hyper_parallel.models.xing40.modeling_xing4_0 import (
    Xing4_0HyperConnection,
    Xing4_0MoE,
    Xing4_0RMSNorm,
)
from tests.common.mark_utils import arg_mark

_DEPTHS = 2
_SEQUENCE = 6
_BATCH = 2
_LAYERS = 3
# One document per row, which SDPA can run; the values still expose any per-depth remap.
_DOCUMENT_ENDS = (_SEQUENCE, _BATCH * _SEQUENCE)

# The smallest configuration that still routes an MoE through every MTP depth: the depths sit
# at layer indices _LAYERS and _LAYERS + 1, past first_k_dense_replace.
_CONFIG = dict(
    vocab_size=64, hidden_size=16, intermediate_size=32, moe_intermediate_size=8,
    num_hidden_layers=_LAYERS, first_k_dense_replace=2, num_attention_heads=2,
    num_key_value_heads=2, q_lora_rank=8, kv_lora_rank=8, qk_rope_head_dim=4,
    qk_nope_head_dim=4, v_head_dim=4, rms_norm_eps=1e-6, hidden_act="silu",
    rope_theta=10000.0,
    rope_scaling={"type": "yarn", "factor": 8.0, "beta_fast": 32, "beta_slow": 1,
                  "mscale": 1, "mscale_all_dim": 1, "original_max_position_embeddings": 64},
    n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
    routed_scaling_factor=2.0, norm_topk_prob=True, n_group=1, topk_group=1,
    moe_aux_loss_coeff=6e-4, moe_router_enable_expert_bias=True,
    moe_router_bias_update_rate=1e-4, use_pad_tokens=True, hc_mult=2, hc_sinkhorn_iters=5,
    hc_eps=1e-6, mhc_h_res_clamp_min=-30, mhc_h_res_clamp_max=30,
    mhc_init_gating_factor=0.01, mtp_loss_factor=0.1, chunk_loss_size=32,
    use_cache=False, tie_word_embeddings=False, max_position_embeddings=64,
)


def _config(num_nextn_predict_layers: int = _DEPTHS) -> Xing4_0Config:
    """Resolve the test configuration with the given number of MTP depths."""
    return Xing4_0Config(**{**_CONFIG, "num_nextn_predict_layers": num_nextn_predict_layers})


def _build_model(num_nextn_predict_layers: int = _DEPTHS) -> Xing40ForCausalLM:
    """Build the model under a fixed seed so every test sees the same weights."""
    torch.manual_seed(0)
    return Xing40ForCausalLM(_config(num_nextn_predict_layers))


def _shifted(rows: list[list[int]], depth: int, pad: int) -> list[list[int]]:
    """Shift every row left by ``depth`` positions and pad its end with ``pad``."""
    return [row[depth:] + [pad] * depth for row in rows]


def _record_kwargs(store: dict[str, dict], name: str) -> Callable[..., None]:
    """Return a forward pre-hook that keeps the keyword arguments of one call under ``name``."""
    def hook(_module: torch.nn.Module, _args: tuple, kwargs: dict) -> None:
        """Keep this call's keyword arguments."""
        store[name] = kwargs
    return hook


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
class TestXing40MTPAlignment(unittest.TestCase):
    """Record what the trunk and every depth actually received, rather than only the loss."""

    def setUp(self) -> None:
        """Run one training forward while recording each depth's inputs, tables and targets."""
        self.model = _build_model()
        self.model.train()
        positions = torch.arange(_SEQUENCE)
        self.input_ids = torch.stack([positions + 10 * (row + 1) for row in range(_BATCH)])
        self.shift_labels = self.input_ids + 1
        # One scored position is dropped, so every depth has to carry the loss mask with it.
        self.loss_mask = torch.ones_like(self.input_ids)
        self.loss_mask[0, 3] = 0
        self.embedded: list[torch.Tensor] = []
        self.scored: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.trunk_states: list[torch.Tensor] = []
        self.hnorm_inputs: list[torch.Tensor] = []
        self.depth_states: list[torch.Tensor] = []
        self.layer_kwargs: dict[str, dict] = {}
        self.attention_kwargs: dict[str, dict] = {}
        original_cross_entropy = Xing40ForCausalLM._cross_entropy

        def record_cross_entropy(model: Xing40ForCausalLM, hidden: torch.Tensor,
                                 targets: torch.Tensor) -> torch.Tensor:
            """Record one scored (states, targets) pair, then score it as usual."""
            self.scored.append((hidden, targets.clone()))
            return original_cross_entropy(model, hidden, targets)

        decoders = {"trunk": self.model.model.layers[0]}
        decoders.update({f"depth {depth}": layer.transformer_layer
                         for depth, layer in enumerate(self.model.mtp.layers, start=1)})
        handles = [
            self.model.model.embed_tokens.register_forward_pre_hook(
                lambda _module, args: self.embedded.append(args[0].clone())),
            self.model.model.norm.register_forward_hook(
                lambda _module, _args, output: self.trunk_states.append(output)),
        ]
        for layer in self.model.mtp.layers:
            handles.append(layer.hnorm.register_forward_pre_hook(
                lambda _module, args: self.hnorm_inputs.append(args[0])))
            handles.append(layer.final_layernorm.register_forward_hook(
                lambda _module, _args, output: self.depth_states.append(output)))
        for name, decoder in decoders.items():
            handles.append(decoder.register_forward_pre_hook(
                _record_kwargs(self.layer_kwargs, name), with_kwargs=True))
            handles.append(decoder.self_attn.register_forward_pre_hook(
                _record_kwargs(self.attention_kwargs, name), with_kwargs=True))
        try:
            with patch.object(Xing40ForCausalLM, "_cross_entropy", record_cross_entropy):
                self.model(input_ids=self.input_ids, shift_labels=self.shift_labels,
                           actual_seq_len=_DOCUMENT_ENDS, loss_mask=self.loss_mask)
        finally:
            for handle in handles:
                handle.remove()

    def test_depth_embeds_the_sequence_shifted_by_its_depth(self) -> None:
        """The trunk embeds every token; depth k embeds input_ids[:, k:] followed by k zeros."""
        rows = self.input_ids.tolist()
        self.assertEqual(self.embedded[0].tolist(), rows,
                         f"the trunk must embed input_ids: expected={rows}, got={self.embedded[0].tolist()}")
        for depth in range(1, _DEPTHS + 1):
            with self.subTest(depth=depth):
                expected = _shifted(rows, depth, 0)
                self.assertEqual(self.embedded[depth].tolist(), expected,
                                 (f"depth {depth} must embed input_ids shifted by {depth} with 0 pad: "
                                  f"expected={expected}, got={self.embedded[depth].tolist()}"))

    def test_depth_scores_the_masked_targets_shifted_by_its_depth(self) -> None:
        """Depth k scores the masked trunk targets shifted by k, padded with ignored slots."""
        trunk_targets = self.shift_labels.masked_fill(self.loss_mask == 0, IGNORE_INDEX).tolist()
        self.assertEqual(self.scored[0][1].tolist(), trunk_targets,
                         (f"the trunk must score the masked shift_labels: expected={trunk_targets}, "
                          f"got={self.scored[0][1].tolist()}"))
        for depth in range(1, _DEPTHS + 1):
            with self.subTest(depth=depth):
                states, targets = self.scored[depth]
                expected = _shifted(trunk_targets, depth, IGNORE_INDEX)
                self.assertEqual(targets.tolist(), expected,
                                 (f"depth {depth} must score the targets shifted by {depth} with "
                                  f"IGNORE_INDEX pad: expected={expected}, got={targets.tolist()}"))
                self.assertEqual(tuple(states.shape[:2]), (_BATCH, _SEQUENCE),
                                 (f"depth {depth} must keep all {_SEQUENCE} positions, "
                                  f"got={tuple(states.shape[:2])}"))
                scored = targets != IGNORE_INDEX
                successors = self.embedded[depth][scored] + 1
                self.assertEqual(targets[scored].tolist(), successors.tolist(),
                                 (f"depth {depth} must predict the successor of every token it embedded "
                                  f"and scores: expected={successors.tolist()}, got={targets[scored].tolist()}"))

    def test_first_depth_fuses_the_trunks_normalized_state_unshifted(self) -> None:
        """Depth 1's hnorm reads the trunk's post-final-norm output, position for position."""
        trunk_state = self.trunk_states[0]
        torch.testing.assert_close(
            self.scored[0][0], trunk_state, rtol=0, atol=0,
            msg=lambda detail: f"the LM head must read the trunk's final norm output: {detail}")
        torch.testing.assert_close(
            self.hnorm_inputs[0], trunk_state, rtol=0, atol=0,
            msg=lambda detail: f"depth 1 hnorm must read the unshifted post-norm trunk state: {detail}")

    def test_later_depth_fuses_the_previous_depths_output(self) -> None:
        """Depth k >= 2 reads depth k-1's final_layernorm output, position for position."""
        for depth in range(2, _DEPTHS + 1):
            with self.subTest(depth=depth):
                torch.testing.assert_close(
                    self.hnorm_inputs[depth - 1], self.depth_states[depth - 2], rtol=0, atol=0,
                    msg=lambda detail, depth=depth: (f"depth {depth} hnorm must read depth {depth - 1}'s "
                                                     f"final_layernorm output: {detail}"))

    def test_every_depth_uses_the_trunks_rope_tables(self) -> None:
        """Every depth's decoder layer receives the trunk's full-length cos and sin tables unchanged."""
        trunk_tables = self.layer_kwargs["trunk"]["position_embeddings"]
        self.assertEqual(tuple(trunk_tables[0].shape[1:2]), (_SEQUENCE,),
                         f"the trunk tables must cover {_SEQUENCE} positions, got {tuple(trunk_tables[0].shape)}")
        for depth in range(1, _DEPTHS + 1):
            tables = self.layer_kwargs[f"depth {depth}"]["position_embeddings"]
            for name, got, expected in zip(("cos", "sin"), tables, trunk_tables):
                with self.subTest(depth=depth, table=name):
                    torch.testing.assert_close(
                        got, expected, rtol=0, atol=0,
                        msg=lambda detail, depth=depth, name=name: (
                            f"depth {depth} must use the trunk's {name} table unchanged: {detail}"))

    def test_every_depth_gets_fp32_rope_tables_in_a_bf16_model(self) -> None:
        """A BF16 model still hands every depth the FP32 tables the released rotary returns.

        Only a BF16 model tells an FP32 table from one cast to the activation dtype.
        """
        model = _build_model().to(torch.bfloat16)
        received: dict[str, dict] = {}
        handles = [layer.transformer_layer.register_forward_pre_hook(
            _record_kwargs(received, f"depth {depth}"), with_kwargs=True)
            for depth, layer in enumerate(model.mtp.layers, start=1)]
        try:
            with torch.no_grad():
                model(input_ids=self.input_ids, shift_labels=self.shift_labels)
        finally:
            for handle in handles:
                handle.remove()
        for depth in range(1, _DEPTHS + 1):
            tables = received[f"depth {depth}"]["position_embeddings"]
            for name, table in zip(("cos", "sin"), tables):
                with self.subTest(depth=depth, table=name):
                    self.assertIs(table.dtype, torch.float32,
                                  f"depth {depth} {name} must stay FP32 in a BF16 model, got {table.dtype}")

    def test_every_depth_uses_the_trunks_document_ends(self) -> None:
        """Every depth's attention receives the trunk's actual_seq_len, not a per-depth remap."""
        trunk_ends = self.attention_kwargs["trunk"].get("actual_seq_len")
        self.assertEqual(trunk_ends, _DOCUMENT_ENDS,
                         f"the trunk attention must receive actual_seq_len={_DOCUMENT_ENDS}, got {trunk_ends}")
        for depth in range(1, _DEPTHS + 1):
            with self.subTest(depth=depth):
                ends = self.attention_kwargs[f"depth {depth}"].get("actual_seq_len")
                self.assertEqual(ends, trunk_ends,
                                 f"depth {depth} attention must receive the trunk's ends: "
                                 f"expected={trunk_ends}, got={ends}")


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
class TestXing40MTPStructure(unittest.TestCase):
    """Pin the depth's parameter names, its norms and the training MoE everywhere."""

    def setUp(self) -> None:
        """Build one model with the configured MTP depths."""
        self.model = _build_model()

    def test_depth_registers_the_mindformers_submodules(self) -> None:
        """The recipe's plan_overrides address these exact names."""
        names = [name for name, _ in self.model.mtp.layers[0].named_children()]
        expected = ["enorm", "hnorm", "eh_proj", "transformer_layer", "final_layernorm"]
        self.assertEqual(names, expected,
                         f"MTP depth submodules changed: expected={expected}, got={names}")

    def test_every_depth_norm_is_the_released_rmsnorm(self) -> None:
        """The recipe's exact-type RMSNorm replacement matches only the released ``Xing4_0RMSNorm``."""
        for depth, layer in enumerate(self.model.mtp.layers, start=1):
            for name in ("enorm", "hnorm", "final_layernorm"):
                with self.subTest(depth=depth, norm=name):
                    norm = getattr(layer, name)
                    self.assertIsInstance(norm, Xing4_0RMSNorm,
                                          f"{name} must be Xing4_0RMSNorm, got {type(norm).__name__}")

    def test_zero_depth_builds_no_mtp_and_logs_no_mtp_loss(self) -> None:
        """``num_nextn_predict_layers: 0`` is the configuration that trains without MTP."""
        model = _build_model(0)
        self.assertIsNone(model.mtp, f"expected no MTP module, got {type(model.mtp).__name__}")
        keys = [key for key in model.state_dict() if key.startswith("mtp")]
        self.assertEqual(keys, [], f"a disabled MTP must register no parameters, got {keys}")
        output = model(input_ids=torch.zeros(_BATCH, _SEQUENCE, dtype=torch.long),
                       shift_labels=torch.ones(_BATCH, _SEQUENCE, dtype=torch.long))
        self.assertEqual(sorted(output.loss), ["foundation_loss"],
                         f"expected only the LM loss, got {sorted(output.loss)}")

    def test_no_released_inference_moe_remains(self) -> None:
        """Every routed layer of the trunk and of each depth is the training MoE."""
        released = [name for name, module in self.model.named_modules() if isinstance(module, Xing4_0MoE)]
        self.assertEqual(released, [], f"released inference MoE layers remain: {released}")
        routed = {f"model.layers.{index}": layer.mlp for index, layer in enumerate(self.model.model.layers)
                  if index >= _CONFIG["first_k_dense_replace"]}
        routed.update({f"mtp.layers.{index}.transformer_layer": layer.transformer_layer.mlp
                       for index, layer in enumerate(self.model.mtp.layers)})
        for name, mlp in routed.items():
            with self.subTest(layer=name):
                self.assertIsInstance(mlp, Xing40MoE, f"{name}.mlp must be Xing40MoE, got {type(mlp).__name__}")


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
class TestXing40HyperConnectionInitialization(unittest.TestCase):
    """Every hyper-connection, trunk and MTP alike, starts from the MindFormers gating scheme.

    The released backbone is a nested ``PreTrainedModel``, and ``initialize_weights`` would hand
    it its own ``_init_weights``: the released scheme, a gating scale of one and an
    ``initializer_range``-wide projection.
    """

    def _assert_mindformers_scheme(self, model: Xing40ForCausalLM) -> None:
        """Check the gating scale, bias and projection width of every hyper-connection."""
        connections = [(name, module) for name, module in model.named_modules()
                       if isinstance(module, Xing4_0HyperConnection)]
        expected_count = 2 * (_LAYERS + _DEPTHS)
        self.assertEqual(len(connections), expected_count,
                         f"expected {expected_count} hyper-connections, got {[name for name, _ in connections]}")
        factor = _CONFIG["mhc_init_gating_factor"]
        for name, connection in connections:
            with self.subTest(module=name):
                scale = connection.hc_scale.detach()
                torch.testing.assert_close(
                    scale, torch.full_like(scale, factor), rtol=0, atol=0,
                    msg=lambda detail, name=name: f"{name}.hc_scale must be {factor}: {detail}")
                base = connection.hc_base.detach()
                torch.testing.assert_close(
                    base, torch.zeros_like(base), rtol=0, atol=0,
                    msg=lambda detail, name=name: f"{name}.hc_base must be zero: {detail}")
                width = connection.hc_fn.detach().std()
                torch.testing.assert_close(
                    width, torch.tensor(1e-4), rtol=0.3, atol=0,
                    msg=lambda detail, name=name: f"{name}.hc_fn std must be about 1e-4: {detail}")

    def test_direct_construction(self) -> None:
        """``Xing40ForCausalLM(config)`` initializes every hyper-connection with the family scheme."""
        self._assert_mindformers_scheme(_build_model())

    def test_framework_initialization_after_meta_materialization(self) -> None:
        """The shared builder's path: meta construction, ``to_empty``, ``_initialize_model_weights``."""
        with _model_init_context(True):
            model = Xing40ForCausalLM.from_config(_config())
        model.to_empty(device="cpu")
        torch.manual_seed(0)
        _initialize_model_weights(model)
        self._assert_mindformers_scheme(model)


if __name__ == "__main__":
    unittest.main()
