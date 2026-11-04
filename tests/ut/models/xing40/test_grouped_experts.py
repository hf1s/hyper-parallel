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
"""Where every released per-expert Xing4.0 weight lands in the packed NPU expert tensors.

``Xing40GroupedExperts.make_transforms()`` is the only thing that puts values into the packed
``gate_up_proj``/``down_proj``: the parameters are created on meta and the per-expert
``nn.ModuleList`` the replacement is built from never supplies them. What is pinned here is
therefore the conversion as the loader runs it -- the transforms carry the replaced module's FQN
as their ``scope_prefix``, keys are collected in the loader's natural-key order
(``hyper_parallel.components.checkpoint.load_groups.sorted_checkpoint_keys``), and each group is
converted under its target name -- and not the operations in isolation.

Every expert's three weights hold disjoint, index-carrying values, so swapping two experts,
pairing the wrong expert's gate with another's up, or fusing the experts into one row changes the
asserted tensors rather than only a shape.
"""

import importlib.util
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.components.checkpoint.load_groups import sorted_checkpoint_keys
from hyper_parallel.models.xing40.adapter.model import Xing40ModuleListExperts
from hyper_parallel.models.xing40.modeling_xing4_0 import Xing4_0MLP
from tests.common.mark_utils import arg_mark

_SCOPE = "model.layers.3.mlp.experts"
_HIDDEN = 4
# Deliberately not _HIDDEN / 2: that would make the packed [experts, hidden, 2 * intermediate]
# and [experts, 2 * intermediate, hidden] layouts the same shape and hide a missing transpose.
_INTERMEDIATE = 3


def _absent(name):
    """An inert ``torch_npu`` API, raising for dunder lookups as a real module would.

    ``inspect.getmodule`` walks ``sys.modules`` reading ``module.__file__`` when torch registers
    custom operators, and a callable ``__file__`` crashes it.

    Args:
        name: Attribute looked up on the stub.

    Returns:
        A callable accepting anything and returning ``None``.

    Raises:
        AttributeError: For dunder names.
    """
    if name.startswith("__"):
        raise AttributeError(name)
    return lambda *args, **kwargs: None


def _torch_npu_stub():
    """Patch an inert ``torch_npu`` into ``sys.modules`` so the NPU experts import on CPU.

    The stub carries a ``__spec__``: Transformers' own ``is_torch_npu_available()`` probe calls
    ``importlib.util.find_spec``, which raises ``ValueError`` for a ``sys.modules`` entry without
    one. Construction and ``make_transforms()`` never call into the backend; only ``forward()``
    does, and this module does not exercise it.

    The stub must not be in place while torch first initializes TorchDynamo: its device registry
    would then register an NPU interface against a stub and double-register torch's compile cache
    artifacts. The released classes imported at module level warm that chain first, which is why
    they are imported there and not here.

    Returns:
        The unstarted ``patch.dict`` context manager.
    """
    stub = types.ModuleType("torch_npu")
    stub.__getattr__ = _absent
    stub.__spec__ = importlib.util.spec_from_loader("torch_npu", loader=None)
    return patch.dict(sys.modules, {"torch_npu": stub})


def _released_experts(num_experts):
    """Build a released per-expert experts container and the checkpoint keys it is stored under.

    The container and the per-expert MLP are the released classes the recipe's ``*.mlp.experts``
    rule actually replaces, so the keys are spelled as the released
    ``model.safetensors.index.json`` spells them: one ``gate_proj``/``up_proj``/``down_proj``
    triple per expert, under the replaced module's FQN.

    Args:
        num_experts: Number of experts in the container.

    Returns:
        The container and ``{checkpoint key: tensor}`` for every per-expert weight.
    """
    config = SimpleNamespace(hidden_size=_HIDDEN, intermediate_size=_INTERMEDIATE,
                             hidden_act="silu", initializer_range=0.02)
    experts = Xing40ModuleListExperts([Xing4_0MLP(config) for _ in range(num_experts)])
    # Distinct decades per projection, distinct hundreds per expert, then a running offset: every
    # single element of the packed result identifies the expert and projection it came from.
    bases = {"gate_proj": 1000.0, "up_proj": 2000.0, "down_proj": 3000.0}
    checkpoint = {}
    with torch.no_grad():
        for index, expert in enumerate(experts):
            for name, base in bases.items():
                weight = getattr(expert, name).weight
                values = base + 100.0 * index + torch.arange(
                    weight.numel(), dtype=torch.float32).reshape(weight.shape)
                weight.copy_(values)
                checkpoint[f"{_SCOPE}.{index}.{name}.weight"] = values.clone()
    return experts, checkpoint


def _convert(transforms, checkpoint):
    """Route the checkpoint through ``transforms`` the way the replacement load path does.

    Each transform is scoped to the replaced module's FQN, keys are offered in the loader's
    natural-key order, every key goes to the first transform whose pattern matches it, and each
    group is then converted under the target name its keys were renamed to.

    Args:
        transforms: Transforms returned by ``make_transforms()``.
        checkpoint: ``{checkpoint key: tensor}`` to convert.

    Returns:
        ``{model tensor name: tensor}`` the conversion produced.

    Raises:
        AssertionError: If a checkpoint key matches no transform.
    """
    for transform in transforms:
        transform.scope_prefix = _SCOPE
    groups = {}
    for key in sorted_checkpoint_keys(checkpoint):
        matched = None
        for transform in transforms:
            target, pattern = transform.rename_source_key(key)
            if pattern is not None:
                matched = (transform, target, pattern)
                break
        assert matched is not None, \
            (f"no transform matched checkpoint key: key={key}, "
             f"patterns={[transform.source_patterns for transform in transforms]}")
        transform, target, pattern = matched
        groups.setdefault(target, transform)
        transform.add_tensor(target, key, pattern, checkpoint[key])
    converted = {}
    for target, transform in groups.items():
        converted.update(transform.convert(target, model=None, config=None))
    return converted


def _expected_packed(checkpoint, num_experts, transposed):
    """The packed tensors the conversion must produce, built straight from the checkpoint.

    Args:
        checkpoint: ``{checkpoint key: tensor}`` of per-expert weights.
        num_experts: Number of experts.
        transposed: Whether to apply the NPU ``[experts, in, out]`` transpose.

    Returns:
        ``{model tensor name: tensor}`` of the expected ``gate_up_proj`` and ``down_proj``.
    """
    # Gate rows before up rows within each expert: the consumer splits the fused projection
    # output with ``chunk(2, dim=-1)``, gate first.
    gate_up = torch.stack([
        torch.cat((checkpoint[f"{_SCOPE}.{index}.gate_proj.weight"],
                   checkpoint[f"{_SCOPE}.{index}.up_proj.weight"]), dim=0)
        for index in range(num_experts)
    ], dim=0)
    down = torch.stack([
        checkpoint[f"{_SCOPE}.{index}.down_proj.weight"] for index in range(num_experts)
    ], dim=0)
    if transposed:
        gate_up = gate_up.transpose(-2, -1).contiguous()
        down = down.transpose(-2, -1).contiguous()
    return {f"{_SCOPE}.gate_up_proj": gate_up, f"{_SCOPE}.down_proj": down}


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
class TestXing40GroupedExpertsTransforms(unittest.TestCase):
    """The released per-expert checkpoint, as the packed NPU expert parameters receive it."""

    def setUp(self) -> None:
        """Make the NPU-backed experts importable, then build the replacement under test."""
        stub = _torch_npu_stub()
        stub.start()
        self.addCleanup(stub.stop)
        from hyper_parallel.models.xing40.adapter.conversion.grouped_experts import (  # pylint: disable=C0415
            Xing40GroupedExperts,
        )

        self.replacement_class = Xing40GroupedExperts
        self.num_experts = 3
        experts, self.checkpoint = _released_experts(self.num_experts)
        self.replacement = Xing40GroupedExperts(module=experts, module_fqn=_SCOPE)

    def test_collects_the_per_expert_keys_with_one_glob_per_projection(self) -> None:
        """One pattern per projection, not per expert index, and both targets are declared.

        ``MergeModulelist`` stacks the tensors that a single source pattern collected, so a
        per-expert-index pattern list would leave it one tensor per pattern: the fused
        ``gate_up_proj`` would come out as a single ``[1, hidden, 2 * intermediate * experts]``
        row and ``down_proj``'s following ``Transpose`` would be handed several tensors and
        reject them. The patterns are relative to the replaced module's FQN, which is why the
        glob stands where the released key has the bare expert index.
        """
        transforms = self.replacement.make_transforms()

        patterns = [transform.source_patterns for transform in transforms]
        targets = [transform.target_patterns for transform in transforms]
        assert patterns == [["*.gate_proj.weight", "*.up_proj.weight"], ["*.down_proj.weight"]], \
            (f"expected one glob per projection, gate before up: "
             f"expected=[['*.gate_proj.weight', '*.up_proj.weight'], ['*.down_proj.weight']], "
             f"got={patterns}")
        assert targets == [["gate_up_proj"], ["down_proj"]], \
            (f"expected the packed fused and down projections as targets: "
             f"expected=[['gate_up_proj'], ['down_proj']], got={targets}")

    def test_packs_the_released_weights_into_the_transposed_npu_layout(self) -> None:
        """The layout the replacement really builds: ``[experts, in, out]``, values and all."""
        assert self.replacement._transpose_source_weights, \
            (f"the released per-expert source is always the Transformers layout, so a transpose "
             f"is expected: _transpose_source_weights={self.replacement._transpose_source_weights}")

        converted = _convert(self.replacement.make_transforms(), self.checkpoint)

        expected = _expected_packed(self.checkpoint, self.num_experts, transposed=True)
        assert sorted(converted) == sorted(expected), \
            (f"conversion produced the wrong model tensors: "
             f"expected={sorted(expected)}, got={sorted(converted)}")
        for name, want in expected.items():
            got = converted[name]
            assert tuple(got.shape) == tuple(want.shape), \
                (f"packed {name} has the wrong shape: "
                 f"expected={tuple(want.shape)}, got={tuple(got.shape)}")
            assert tuple(got.shape) == tuple(getattr(self.replacement, name.rsplit(".", 1)[1]).shape), \
                (f"packed {name} does not fit the parameter it loads into: "
                 f"parameter={tuple(getattr(self.replacement, name.rsplit('.', 1)[1]).shape)}, "
                 f"converted={tuple(got.shape)}")
            assert torch.equal(got, want), \
                (f"packed {name} holds the wrong released values: "
                 f"expected={want.tolist()}, got={got.tolist()}")

    def test_packs_without_a_transpose_when_the_target_layout_already_matches(self) -> None:
        """The transpose is the pre-existing layout step, not something the packing implies.

        ``GroupedExperts`` decides it from the source shapes; forcing it off here keeps that
        decision and the packing independent, so the packing is pinned on its own.
        """
        self.replacement._transpose_source_weights = False

        converted = _convert(self.replacement.make_transforms(), self.checkpoint)

        expected = _expected_packed(self.checkpoint, self.num_experts, transposed=False)
        for name, want in expected.items():
            got = converted[name]
            assert torch.equal(got, want), \
                (f"untransposed {name} is not the plain packed released layout: "
                 f"expected={want.tolist()}, got={got.tolist()}")

    def test_collects_the_experts_in_index_order_past_ten(self) -> None:
        """Expert 2 before expert 10: the glob collects keys in the loader's natural-key order.

        Eleven experts is the smallest count at which a plain lexicographic key order would
        place expert 10 second and silently permute every packed expert row.
        """
        num_experts = 11
        experts, checkpoint = _released_experts(num_experts)
        replacement = self.replacement_class(module=experts, module_fqn=_SCOPE)

        converted = _convert(replacement.make_transforms(), checkpoint)

        expected = _expected_packed(checkpoint, num_experts, transposed=True)
        for name, want in expected.items():
            got = converted[name]
            assert torch.equal(got, want), \
                (f"packed {name} rows are not in expert-index order: "
                 f"expected first elements="
                 f"{[row.flatten()[0].item() for row in want]}, "
                 f"got first elements={[row.flatten()[0].item() for row in got]}")


if __name__ == "__main__":
    unittest.main()
