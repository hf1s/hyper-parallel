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
"""CPU training-lifecycle tests of the Xing4.0 model."""

import math
from collections.abc import Callable
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from hyper_parallel.components.modules.mhc import MhcPostModule, MhcPreModule
from hyper_parallel.components.modules.moe_utils import MoEMonitorCallback
from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models import HyperAutoModelForCausalLM
from hyper_parallel.models.xing40.adapter.builder import build_model
from hyper_parallel.models.xing40.modeling_xing40 import (
    Xing40Config,
    Xing40ForCausalLM,
    Xing40MhcPre,
    Xing40MoE,
)
from hyper_parallel.models.replacement import ModuleReplacementSpec
from tests.common.mark_utils import arg_mark

TINY_CONFIG = {
    "vocab_size": 128, "hidden_size": 64, "intermediate_size": 96, "moe_intermediate_size": 32,
    "num_hidden_layers": 3, "first_k_dense_replace": 1, "num_attention_heads": 4, "num_key_value_heads": 4,
    "q_lora_rank": 32, "kv_lora_rank": 16, "qk_rope_head_dim": 8, "qk_nope_head_dim": 8, "v_head_dim": 8,
    "n_routed_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 1, "n_group": 1, "topk_group": 1,
    "moe_aux_loss_coeff": 1e-4, "moe_router_enable_expert_bias": True, "moe_router_bias_update_rate": 1e-3,
    "use_pad_tokens": True, "num_residual_streams": 4, "mhc_sinkhorn_iterations": 20,
    "mhc_init_gating_factor": 0.01, "mhc_layernorm_epsilon": 1e-6, "num_nextn_predict_layers": 1,
    "mtp_loss_factor": 0.3, "chunk_loss_size": 8,
}
# Reference mHC shift initialization: pre = 1/4, post = 1 and a strongly diagonal residual mixing.
_IDENTITY_SHIFT = torch.cat((torch.full((4,), -math.log(3.0)), torch.zeros(4), ((torch.eye(4) - 1.0) * 5.0).flatten()))
_TO_EMPTY = nn.Module.to_empty


def _config(**overrides) -> Xing40Config:
    """Return the tiny configuration with ``overrides`` applied."""
    return Xing40Config(**{**TINY_CONFIG, **overrides})  # pylint: disable=unexpected-keyword-arg


def _model(**overrides) -> Xing40ForCausalLM:
    """Build the tiny model with fixed random weights."""
    torch.manual_seed(0)
    return Xing40ForCausalLM(_config(**overrides))


def _train_step(model: Xing40ForCausalLM) -> dict[str, torch.Tensor]:
    """Run one forward/backward on a fixed sequence and return the named losses."""
    generator = torch.Generator().manual_seed(1)
    input_ids = torch.randint(0, TINY_CONFIG["vocab_size"], (1, 16), generator=generator)
    losses = model(input_ids, shift_labels=torch.roll(input_ids, -1, dims=1)).loss
    torch.stack(list(losses.values())).sum().backward()
    return losses


def _poisoned_to_empty(module: nn.Module, *, device, recurse: bool = True) -> nn.Module:
    """Materialize like ``to_empty`` but fill every float tensor with NaN, as arbitrary memory may hold."""
    _TO_EMPTY(module, device=device, recurse=recurse)
    with torch.no_grad():
        for tensor in (*module.parameters(), *module.buffers()):
            if tensor.is_floating_point():
                tensor.fill_(float("nan"))
    return module


class TestXing40Training(unittest.TestCase):
    """Initialization, objective coverage and router-statistics lifecycle."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_meta_materialization_initializes_every_tensor(self):
        """
        Feature: Meta-device construction through the recipe's builder.
        Description: to_empty() storage is NaN; the framework then runs the random initialization.
        Expectation: No tensor, RoPE tables included, stays NaN and the mHC gates and shifts take their
            reference values.
        """
        with mock.patch.object(nn.Module, "to_empty", _poisoned_to_empty):
            model = build_model(config=TINY_CONFIG)
        uninitialized = [name for name, tensor in (*model.named_parameters(), *model.named_buffers())
                         if tensor.is_floating_point() and torch.isnan(tensor).any()]
        self.assertEqual(uninitialized, [], f"uninitialized tensors: {uninitialized}")
        connection = model.mtp.layers[0].transformer_layer.ffn_hc_pre
        torch.testing.assert_close(connection.branch_alpha.detach(), torch.full((3,), 0.01),
                                   msg="case: gates take mhc_init_gating_factor")
        torch.testing.assert_close(connection.branch_beta.detach(), _IDENTITY_SHIFT,
                                   msg="case: shifts start as the identity residual")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_pretrained_load_rebuilds_rope_tables(self):
        """
        Feature: Pretrained loading through the framework builder.
        Description: Save a model in the Transformers layout, then load it with NaN-filled to_empty() storage.
        Expectation: Every saved tensor loads exactly, and the RoPE tables, which checkpoints do not hold, are
            rebuilt.
        """
        source = _model()
        with tempfile.TemporaryDirectory() as directory:
            source.save_pretrained(directory)
            with mock.patch.object(nn.Module, "to_empty", _poisoned_to_empty):
                loaded = HyperAutoModelForCausalLM.from_pretrained(directory)
        expected = source.state_dict()
        mismatched = [name for name, tensor in loaded.state_dict().items() if not torch.equal(tensor, expected[name])]
        self.assertEqual(mismatched, [], f"tensors that did not load: {mismatched}")
        torch.testing.assert_close(loaded.model.rotary_emb.inv_freq, source.model.rotary_emb.inv_freq, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_objective_reaches_every_parameter(self):
        """
        Feature: Model-computed training objective.
        Description: One step through the LM and MTP losses and the injected router balance loss.
        Expectation: Both losses are finite and every trainable parameter, MTP included, has a finite gradient.
        """
        model = _model()
        losses = _train_step(model)
        self.assertEqual(sorted(losses), ["foundation_loss", "foundation_loss/mtp"], f"losses: {sorted(losses)}")
        for name, loss in losses.items():
            self.assertTrue(torch.isfinite(loss), f"{name}={loss}")
        missing = [name for name, parameter in model.named_parameters()
                   if parameter.grad is None or not torch.isfinite(parameter.grad).all()]
        self.assertEqual(missing, [], f"parameters without a finite gradient: {missing}")
        mtp_gradient = model.mtp.layers[0].eh_proj.weight.grad.abs().sum()
        self.assertGreater(mtp_gradient.item(), 0.0, f"MTP projection gradient sum={mtp_gradient}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_mtp_depth_follows_mindformers_layout(self):
        """
        Feature: MindFormers multi-token-prediction layout.
        Description: Record the inputs and outputs of the backbone norm and the MTP depth's modules in one step.
        Expectation: The depth reads the backbone's final-normed state, projects [enorm(next-token embeddings),
            hnorm(state)] and collapses its streams with its own head.
        """
        model = _model()
        layer = model.mtp.layers[0]
        seen: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        def _recorder(key: str) -> Callable[[nn.Module, tuple, torch.Tensor], None]:
            """Return a forward hook that keeps the first input and output of one module."""
            def record(_: nn.Module, args: tuple, output: torch.Tensor) -> None:
                """Keep the first call's input and output under ``key``."""
                seen.setdefault(key, (args[0], output))
            return record

        modules = {"norm": model.model.norm, "enorm": layer.enorm, "hnorm": layer.hnorm,
                   "eh_proj": layer.eh_proj, "mtp_head": layer.hc_head}
        hooks = [module.register_forward_hook(_recorder(key)) for key, module in modules.items()]
        _train_step(model)
        for hook in hooks:
            hook.remove()
        torch.testing.assert_close(seen["hnorm"][0], seen["norm"][1], rtol=0, atol=0,
                                   msg="case: the MTP depth reads the final-normed backbone state")
        torch.testing.assert_close(seen["eh_proj"][0], torch.cat((seen["enorm"][1], seen["hnorm"][1]), dim=-1),
                                   rtol=0, atol=0, msg="case: the fusion projects [embedding, state]")
        self.assertIsNot(layer.hc_head, model.model.hc_head, "the MTP depth must own its stream-collapse head")
        self.assertIn("mtp_head", seen, "the MTP depth must collapse its streams with its own head")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_balance_loss_reaches_router(self):
        """
        Feature: Sequence-wise router balance loss.
        Description: The same step with the balance loss disabled and enabled.
        Expectation: Only the router gradients change; the reported losses do not.
        """
        disabled, enabled = _model(moe_aux_loss_coeff=0.0), _model(moe_aux_loss_coeff=1.0)
        disabled_losses, enabled_losses = _train_step(disabled), _train_step(enabled)
        torch.testing.assert_close(enabled_losses["foundation_loss"], disabled_losses["foundation_loss"])
        router_delta = (enabled.model.layers[1].mlp.gate.weight.grad
                        - disabled.model.layers[1].mlp.gate.weight.grad).abs().max()
        self.assertGreater(router_delta.item(), 0.0, f"router gradient change={router_delta}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_expert_bias_updates_once_per_step(self):
        """
        Feature: Expert-load bias update by the Trainer's MoE monitor.
        Description: One forward, then the monitor's step-end update twice.
        Expectation: The first update moves each bias by rate*sign(mean-count); the second finds no new
            tokens and changes nothing.
        """
        model = _model()
        _train_step(model)
        layers = [module for module in model.modules() if isinstance(module, Xing40MoE)]
        counts = [layer.tokens_per_expert.clone() for layer in layers]
        monitor = MoEMonitorCallback(model)
        monitor.on_step_end()
        for layer, count in zip(layers, counts):
            expected = (count.mean() - count).sign() * layer.expert_bias_update_rate
            torch.testing.assert_close(layer.gate.e_score_correction_bias, expected)
        monitor.on_step_end()
        for layer, count in zip(layers, counts):
            expected = (count.mean() - count).sign() * layer.expert_bias_update_rate
            torch.testing.assert_close(layer.gate.e_score_correction_bias, expected,
                                       msg="case: no new tokens leave the bias unchanged")


class TestMhcKernelContract(unittest.TestCase):
    """The mHC modules are valid sources of the framework's fused NPU mHC modules."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_fused_modules_replace_the_mhc_modules(self):
        """
        Feature: NPU fused mHC replacement.
        Description: Build MhcPreModule/MhcPostModule from a decoder layer's mHC modules; run the post
            module's portable path on random streams and coefficients.
        Expectation: The pre module reuses the layer's parameters and settings; the post module returns
            the layer's updated streams.
        """
        layer = _model().model.layers[1]
        fused_pre = MhcPreModule(module=layer.attn_hc_pre)
        self.assertIs(fused_pre.phi, layer.attn_hc_pre.phi, "MhcPreModule must reuse the mixing projection")
        self.assertIs(fused_pre.branch_alpha, layer.attn_hc_pre.branch_alpha, "MhcPreModule must reuse branch_alpha")
        self.assertIs(fused_pre.branch_beta, layer.attn_hc_pre.branch_beta, "MhcPreModule must reuse branch_beta")
        self.assertEqual((fused_pre.num_stream, fused_pre.mhc_recur_norm), (4, 20),
                         f"streams/iterations: {(fused_pre.num_stream, fused_pre.mhc_recur_norm)}")
        generator = torch.Generator().manual_seed(2)
        hidden = TINY_CONFIG["hidden_size"]
        output = torch.randn(2, 5, hidden, generator=generator)
        streams = torch.randn(2, 5, 4 * hidden, generator=generator)
        post, mixing = torch.rand(2, 5, 4, generator=generator), torch.rand(2, 5, 4, 4, generator=generator)
        torch.testing.assert_close(MhcPostModule(module=layer.attn_hc_post)(output, streams, post, mixing),
                                   layer.attn_hc_post(output, streams, post, mixing),
                                   msg="case: MhcPostModule's portable path equals the post module")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_fused_pre_module_keeps_the_mhc_initialization(self):
        """
        Feature: Random initialization with the fused mHC replacement.
        Description: Build through the recipe's builder with '*_hc_pre' replaced by MhcPreModule and NaN-filled
            to_empty() storage.
        Expectation: The fused modules take the mHC reference initialization.
        """
        setup = DistributedSetup(module_replacements=[ModuleReplacementSpec(
            match=("*_hc_pre",), factory=MhcPreModule, module_type=Xing40MhcPre)])
        with mock.patch.object(nn.Module, "to_empty", _poisoned_to_empty):
            model = build_model(config=TINY_CONFIG, distributed_setup=setup)
        fused_pre = model.mtp.layers[0].transformer_layer.ffn_hc_pre
        self.assertIsInstance(fused_pre, MhcPreModule, f"replaced module: {type(fused_pre).__name__}")
        torch.testing.assert_close(fused_pre.branch_alpha.detach(), torch.full((3,), 0.01),
                                   msg="case: gates take mhc_init_gating_factor")
        torch.testing.assert_close(fused_pre.branch_beta.detach(), _IDENTITY_SHIFT,
                                   msg="case: shifts start as the identity residual")
        self.assertFalse(torch.isnan(fused_pre.phi.weight).any().item(), "the mixing projection stays NaN")


if __name__ == "__main__":
    unittest.main()
