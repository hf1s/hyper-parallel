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

"""Opt-in, independent MF precision hypotheses and first-forward observations."""
from dataclasses import replace
import json
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
import torch_npu
from torch import nn
from torch.nn import functional as F

from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager, _WrapModuleInfo
from hyper_parallel.models.jt_deepseek_v3 import modeling_jt_deepseek_v3 as jt


class PrecisionFSDP(FSDP2Manager):
    """Retain selected small precision-sensitive units in FP32."""

    def _find_wrap_modules(self, model, metadata_by_parameter=None):
        units = super()._find_wrap_modules(model, metadata_by_parameter)
        selected = {id(unit.module) for unit in units}
        for name, module in dict(model.named_modules()).items():
            if getattr(module, "keep_compute_in_fp32", False) and id(module) not in selected:
                units.append(_WrapModuleInfo(name, module))
                selected.add(id(module))
        return units

    def _parallelize_child_units(self, wrap_modules, owner_by_parameter, metadata_by_parameter,
                                replicate_params, dense_fsdp_kwargs):
        precision = dict(dense_fsdp_kwargs)
        precision["mp_policy"] = replace(dense_fsdp_kwargs["mp_policy"], param_dtype=torch.float32,
                                          cast_forward_inputs=False, output_dtype=None)
        embedding = dict(precision)
        embedding["mp_policy"] = replace(precision["mp_policy"],
                                          output_dtype=dense_fsdp_kwargs["mp_policy"].param_dtype)
        count = 0
        for unit in sorted(wrap_modules, key=lambda item: item.fqn.count("."), reverse=True):
            options = precision if getattr(unit.module, "keep_compute_in_fp32", False) else dense_fsdp_kwargs
            if options is precision and isinstance(unit.module, nn.Embedding):
                options = embedding
            count += super()._parallelize_child_units(
                [unit], owner_by_parameter, metadata_by_parameter, replicate_params, options)
        return count


class MFGraphCrossEntropy(torch.autograd.Function):
    """Apply the measured MF log-softmax/NLL formula to the live HP logits."""

    @staticmethod
    def forward(ctx, logits, labels, denominator):
        ctx.input_shape, ctx.input_dtype = logits.shape, logits.dtype
        values = logits.reshape(-1, logits.shape[-1]).float()
        targets = labels.reshape(-1)
        mask = targets >= 0
        targets = targets.clamp_min(0)
        centered = values - values.amax(-1, keepdim=True)
        log_probabilities = centered - centered.exp().sum(-1, keepdim=True).log()
        denominator = denominator.float()
        ctx.save_for_backward(log_probabilities, targets, mask, denominator)
        losses = -log_probabilities.gather(1, targets[:, None]).squeeze(-1)
        return (losses * mask).sum() / denominator

    @staticmethod
    def backward(ctx, gradient):
        log_probabilities, targets, mask, denominator = ctx.saved_tensors
        derivative = log_probabilities.exp()
        derivative.scatter_add_(1, targets[:, None], -torch.ones_like(targets[:, None], dtype=torch.float32))
        sensitivity = (gradient / denominator) * mask.float()
        result = derivative * sensitivity[:, None]
        return result.to(ctx.input_dtype).reshape(ctx.input_shape), None, None


def mf_graph_cross_entropy(*, logits, labels, vocab_size, shift_labels, num_items_in_batch,
                           ignore_index=-100, **kwargs):
    """Expose the same masked mean-loss contract used by the output projection."""
    del labels, kwargs
    if ignore_index != -100 or logits.shape[-1] != vocab_size:
        raise ValueError('Unexpected CE vocabulary or ignore index')
    return MFGraphCrossEntropy.apply(logits, shift_labels, num_items_in_batch)


def _dvm_sumx(values: torch.Tensor) -> torch.Tensor:
    """Reproduce measured SIMD64 SumX ordering for an aligned final axis."""
    if values.shape[-1] % 64:
        raise ValueError('DVM SumX replay requires a final axis divisible by 64')
    work = values.reshape(*values.shape[:-1], -1, 64)
    while work.shape[-2] > 1:
        length = work.shape[-2]
        half = length // 2
        reduced = work[..., :half, :] + work[..., half:2 * half, :]
        if length % 2:
            reduced[..., 0, :] = reduced[..., 0, :] + work[..., -1, :]
        work = reduced
    work = work.squeeze(-2)
    while work.shape[-1] > 1:
        work = work[..., ::2] + work[..., 1::2]
    return work.squeeze(-1)


def _sequential_sum(values: torch.Tensor) -> torch.Tensor:
    """Accumulate the last axis in its observed iteration order."""
    result = values[..., 0].clone()
    for index in range(1, values.shape[-1]):
        result = result + values[..., index]
    return result


def _dvm_ce_sum(exponents: torch.Tensor) -> torch.Tensor:
    """Reproduce the measured 910B3 DVM schedule for the reference CE shape."""
    if exponents.shape != (4096, 152064):
        raise ValueError('Measured DVM CE schedule requires 4096 rows and vocabulary 152064')
    partials = _dvm_sumx(exponents.reshape(4096, 22, 6912))
    result = _sequential_sum(partials)
    # Tail rows match the measured sum of even-index and odd-index block accumulations.
    result[4080:] = _sequential_sum(partials[4080:, ::2]) + _sequential_sum(partials[4080:, 1::2])
    return result[:, None]


class MFDVMCrossEntropy(MFGraphCrossEntropy):
    """Evaluate live HP logits with the measured MF DVM reduction ordering."""

    @staticmethod
    def forward(ctx, logits, labels, denominator):
        ctx.input_shape, ctx.input_dtype = logits.shape, logits.dtype
        values = logits.reshape(-1, logits.shape[-1]).float()
        targets = labels.reshape(-1)
        mask = targets >= 0
        targets = targets.clamp_min(0)
        centered = values - values.amax(-1, keepdim=True)
        log_probabilities = centered - _dvm_ce_sum(centered.exp()).log()
        denominator = denominator.float()
        ctx.save_for_backward(log_probabilities, targets, mask, denominator)
        losses = -log_probabilities.gather(1, targets[:, None]).squeeze(-1)
        return _dvm_sumx(losses * mask) / denominator


def mf_dvm_cross_entropy(*, logits, labels, vocab_size, shift_labels, num_items_in_batch,
                         ignore_index=-100, **kwargs):
    """Keep the masked mean-loss contract while selecting the measured DVM schedule."""
    del labels, kwargs
    if ignore_index != -100 or logits.shape[-1] != vocab_size:
        raise ValueError('Unexpected CE vocabulary or ignore index')
    return MFDVMCrossEntropy.apply(logits, shift_labels, num_items_in_batch)


class MFReferenceCE(torch.autograd.Function):
    """Diagnostic only: replay measured CE values without retaining them on NPU."""

    @staticmethod
    def forward(ctx, logits, labels, denominator):
        if logits.numel() != 4096 * 152064:
            raise ValueError('Reference CE probe only supports the recorded 4K batch')
        values = logits.reshape(4096, 152064).float()
        observed = values[:1, :16].cpu().numpy()
        matches = []
        root = Path('/state/mf-deepcechunks58/deep')
        for head, factor in [('model.output_layer', '1.0'), ('model.mtp.output_layer', '0.3')]:
            source = np.load(next(root.glob(head + '_logits_chunk_0000_float32*')), mmap_mode='r')
            if np.array_equal(observed, source[:1, :16]):
                matches.append(factor)
        if len(matches) != 1:
            raise ValueError('Reference CE probe requires the exact recorded initial forward')
        ctx.reference_path = next(Path('/state/ce-native-logsoftmax').glob('logp-' + matches[0] + '_float32*'))
        ctx.targets = labels.reshape(-1).detach()
        ctx.denominator = denominator.float().detach()
        ctx.input_shape, ctx.input_dtype = logits.shape, logits.dtype
        centered = values - values.amax(-1, keepdim=True)
        log_probabilities = centered - centered.exp().sum(-1, keepdim=True).log()
        losses = -log_probabilities.gather(1, ctx.targets.clamp_min(0)[:, None]).squeeze(-1)
        print('[DIAGNOSTIC-ONLY] Observed MF CE log-probability replay, factor=' + matches[0], flush=True)
        return (losses * (ctx.targets >= 0)).sum() / ctx.denominator

    @staticmethod
    def backward(ctx, gradient):
        reference = np.load(ctx.reference_path, mmap_mode='r').reshape(4096, 152064)
        result = torch.empty((4096, 152064), device=gradient.device, dtype=ctx.input_dtype)
        for start in range(0, 4096, 64):
            targets = ctx.targets[start:start + 64]
            values = torch.tensor(np.array(reference[start:start + 64]), device=gradient.device)
            derivative = values.exp()
            derivative.scatter_add_(1, targets.clamp_min(0)[:, None],
                                    -torch.ones_like(targets[:, None], dtype=torch.float32))
            sensitivity = (gradient / ctx.denominator) * (targets >= 0).float()
            result[start:start + 64].copy_(derivative * sensitivity[:, None])
        return result.reshape(ctx.input_shape), None, None


def mf_reference_cross_entropy(*, logits, labels, vocab_size, shift_labels, num_items_in_batch,
                               ignore_index=-100, **kwargs):
    """Use only in the single-update diagnostic; never a training candidate."""
    del labels, vocab_size, kwargs
    if ignore_index != -100:
        raise ValueError('Unexpected ignore index')
    return MFReferenceCE.apply(logits, shift_labels, num_items_in_batch)


class MFNorm(jt.DeepseekV32RMSNorm):
    """Compute native fused RMSNorm with FP32 input and affine weight, then cast."""
    keep_compute_in_fp32 = True

    def forward(self, hidden_states):
        value = torch_npu.npu_rms_norm(hidden_states.float(), self.weight.float(),
                                     epsilon=self.variance_epsilon)[0]
        return value.to(getattr(self, "alignment_output_dtype", hidden_states.dtype))


class MFNormResidual(torch.autograd.Function):
    """Test one final cast for the norm-input and residual-path gradients."""

    @staticmethod
    def forward(ctx, values, weight, epsilon):
        inputs = values.float()
        normalized, rstd = torch_npu.npu_rms_norm(inputs, weight.float(), epsilon=epsilon)
        ctx.save_for_backward(inputs, weight, rstd)
        ctx.input_dtype = values.dtype
        return normalized.to(values.dtype), values

    @staticmethod
    def backward(ctx, norm_gradient, residual_gradient):
        inputs, weight, rstd = ctx.saved_tensors
        input_gradient, weight_gradient = torch.ops.npu.npu_rms_norm_backward(
            norm_gradient.float(), inputs, weight.float(), rstd)
        return (input_gradient + residual_gradient.float()).to(ctx.input_dtype), weight_gradient, None


class MFDualNorm(MFNorm):
    """Expose the residual identity at the same autograd boundary as RMSNorm."""

    def forward(self, hidden_states):
        return MFNormResidual.apply(hidden_states, self.weight, self.variance_epsilon)


class MFResidualGradientDecoder(jt.JTDeepseekV3Decoder):
    """Preserve BF16 forward values while testing fused residual backward casts."""

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
        if past_key_values is not None or use_cache:
            raise ValueError('JT does not support cached decoding')
        normalized, residual = self.input_layernorm(hidden_states)
        branch, _ = self.self_attn(normalized, attention_mask=attention_mask,
                                   position_ids=position_ids, position_embeddings=position_embeddings, **kwargs)
        hidden_states = residual + branch
        normalized, residual = self.post_attention_layernorm(hidden_states)
        branch = self.mlp(normalized)
        result = (residual + branch).to(hidden_states.dtype)
        return result.float() if getattr(self, "alignment_return_fp32", False) else result


class MFRouterWeights(torch.autograd.Function):
    """Preserve the native graph's normalization derivatives and auxiliary merge."""

    @staticmethod
    def forward(ctx, scores, indices, scale, coefficient):
        tokens, experts = scores.shape
        selected = scores.gather(1, indices)
        selected_denominator = selected.sum(-1, keepdim=True) + 1e-20
        denominator = scores.sum(-1, keepdim=True) + 1e-20
        counts = torch.bincount(indices.reshape(-1), minlength=experts).float()
        load = counts / indices.numel()
        normalized = selected / selected_denominator
        auxiliary = ((scores / denominator).mean(0) * load).mean() * (coefficient * experts ** 2)
        ctx.save_for_backward(scores, indices, normalized, selected_denominator, denominator, load)
        ctx.scale, ctx.coefficient = scale, coefficient
        return normalized * scale, auxiliary

    @staticmethod
    def backward(ctx, probability_gradient, auxiliary_gradient):
        scores, indices, normalized, selected_denominator, denominator, load = ctx.saved_tensors
        tokens, experts = scores.shape
        scaled = probability_gradient * ctx.scale
        selected_sum = (-scaled * (normalized / selected_denominator)).sum(-1, keepdim=True)
        selected_gradient = selected_sum + scaled / selected_denominator
        routed = torch.zeros_like(scores).scatter_add(1, indices, selected_gradient)
        auxiliary_base = (((auxiliary_gradient * (ctx.coefficient * experts ** 2)) / experts) * load) / tokens
        auxiliary_sum = (-auxiliary_base * ((scores / denominator) / denominator)).sum(-1, keepdim=True)
        derivative = (routed + auxiliary_sum) + auxiliary_base / denominator
        return derivative, None, None, None


class MFSigmoid(torch.autograd.Function):
    """Keep the unfused graph order of the sigmoid derivative."""

    @staticmethod
    def forward(ctx, value):
        result = value.sigmoid()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, gradient):
        value, = ctx.saved_tensors
        return (1 - value) * (value * gradient)


class MFRouterMatMul(torch.autograd.Function):
    """Match the native FP32 router's materialized-transpose weight derivative."""

    @staticmethod
    def forward(ctx, values, weight):
        if values.dtype != torch.float32 or weight.dtype != torch.float32:
            raise ValueError('Native router gradient requires FP32 inputs and weights')
        ctx.save_for_backward(values, weight)
        return F.linear(values, weight)

    @staticmethod
    def backward(ctx, gradient):
        values, weight = ctx.saved_tensors
        dx = torch.mm(gradient, weight) if ctx.needs_input_grad[0] else None
        dw = torch.mm(values.T.contiguous(), gradient).T if ctx.needs_input_grad[1] else None
        return dx, dw


class MFRouter(jt.JTDeepseekV3Gate):
    """Compute MF's FP32 routing logits without BF16 weight rounding."""
    keep_compute_in_fp32 = True

    def forward(self, hidden_states):
        values, weight = hidden_states.reshape(-1, self.hidden_dim).float(), self.weight.float()
        self.router_logits = (MFRouterMatMul.apply(values, weight)
                              if getattr(self, 'alignment_contiguous_gradient', False) else F.linear(values, weight))
        return self.router_logits


class MFRotary(jt.JTDeepseekV3RotaryEmbedding):
    """Delay rounding until both FP32 rotary products have been added."""

    def forward(self, values, cos, sin):
        ordered = torch.cat((values[..., ::2], values[..., 1::2]), dim=-1).float()
        first, second = ordered.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return (ordered * cos.float().unsqueeze(1) + rotated * sin.float().unsqueeze(1)).to(values.dtype)


class MFGraphSwiGLU(torch.autograd.Function):
    """Follow the observed GraphKernel SiLUGrad expression and cast boundaries."""

    @staticmethod
    def forward(ctx, pair, round_up):
        ctx.save_for_backward(pair)
        ctx.round_up = round_up
        return torch_npu.npu_swiglu(pair, dim=-1)

    @staticmethod
    def backward(ctx, gradient):
        pair, = ctx.saved_tensors
        gate, up = pair.chunk(2, dim=-1)
        denominator = 1 + torch.exp(-gate.float())
        sigmoid = 1 / denominator
        silu = gate.float() / denominator
        derivative = sigmoid + silu - sigmoid * silu
        gate_gradient = (derivative * (gradient.float() * up.float())).to(gate.dtype)
        up_gradient = (gradient * F.silu(gate) if ctx.round_up
                       else (gradient.float() * F.silu(gate.float())).to(up.dtype))
        return torch.cat((gate_gradient, up_gradient), dim=-1), None


class MFSharedSwiGLU(torch.autograd.Function):
    """Preserve native fused forward and the measured shared-MLP backward casts."""

    @staticmethod
    def forward(ctx, pair):
        ctx.save_for_backward(pair)
        return torch_npu.npu_swiglu(pair, dim=-1)

    @staticmethod
    def backward(ctx, gradient):
        pair, = ctx.saved_tensors
        gate, up = pair.chunk(2, dim=-1)
        gate_gradient = torch.ops.aten.silu_backward(
            gradient.float() * up.float(), gate.float()).to(gate.dtype)
        up_gradient = gradient * F.silu(gate)
        return torch.cat((gate_gradient, up_gradient), dim=-1)


class FusedDenseMLP(jt.DeepseekV32MLP):
    """Test one fused SwiGLU rounding instead of separate SiLU and multiplication."""

    def forward(self, hidden_states):
        hidden_states = hidden_states.to(self.gate_proj.weight.dtype)
        if getattr(self, 'alignment_interleaved', False):
            weight = torch.stack((self.gate_proj.weight, self.up_proj.weight), dim=1).flatten(0, 1)
            projected = F.linear(hidden_states, weight)
            pair = torch.cat((projected[..., ::2], projected[..., 1::2]), dim=-1)
        else:
            pair = torch.cat((self.gate_proj(hidden_states), self.up_proj(hidden_states)), dim=-1)
        if getattr(self, 'alignment_ir_gradient', False):
            activation = MFGraphSwiGLU.apply(pair, getattr(self, 'alignment_shared_gradient', False))
        else:
            activation = (MFSharedSwiGLU.apply(pair) if getattr(self, "alignment_shared_gradient", False)
                          else torch_npu.npu_swiglu(pair, dim=-1))
        directory = getattr(self, 'alignment_backward_directory', None)
        if directory is not None:
            prefix = directory / self.alignment_backward_name
            def record_projection(gradient):
                np.save(str(prefix) + '_fc1_gradient.npy', gradient.detach().float().cpu().numpy())
                return gradient
            def record_activation(gradient):
                np.save(str(prefix) + '_activation_gradient.npy', gradient.detach().float().cpu().numpy())
                return gradient
            projected.register_hook(record_projection)
            activation.register_hook(record_activation)
        return self.down_proj(activation)


class MFKeyExpand(torch.autograd.Function):
    """Reduce shared-key gradients in explicit FP32 before the BF16 boundary."""

    @staticmethod
    def forward(ctx, value, heads, native_tree=False):
        ctx.native_tree = native_tree
        return value.expand(-1, heads, -1, -1)

    @staticmethod
    def backward(ctx, gradient):
        ordered = gradient.permute(2, 0, 1, 3).contiguous().float()
        if ctx.native_tree:
            if ordered.shape[2] != 128:
                raise ValueError('The verified DVM SumY schedule requires 128 heads')
            while ordered.shape[2] > 1:
                half = ordered.shape[2] // 2
                ordered = ordered[:, :, :half] + ordered[:, :, half:]
            reduced = ordered.to(gradient.dtype)
        else:
            reduced = ordered.sum(2, keepdim=True).to(gradient.dtype)
        return reduced.permute(1, 2, 0, 3), None, None


class MFKeyReductionMLAAttention(jt.JTDeepseekV3MLAAttention):
    """Keep native projection values and the measured shared-key reduction dtype."""

    def _project_attention_inputs(self, hidden_states: torch.Tensor, position_embeddings: Any,
                                  past_key_values: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states to queries, keys and values.

        Unlike the parent, the rotary key passes through ``key_rope_gather``, where
        the recipe gathers the sequence-parallel shards, and both rotary parts use
        :class:`JTDeepseekV3RotaryEmbedding`.

        Args:
            hidden_states: Input hidden states, ``[batch, sequence, hidden]``.
            position_embeddings: Rotary ``(cos, sin)`` tables.
            past_key_values: Must be ``None``; cached decoding is unsupported.

        Returns:
            Query, key and value states, each ``[batch, heads, sequence, head_dim]``.

        Raises:
            ValueError: If ``position_embeddings`` is missing or a cache is given.
        """
        if past_key_values is not None or position_embeddings is None:
            raise ValueError("JT MLA attention requires position embeddings and no KV cache")
        query_latent, kv_latent, key_rope = self.project_latent_inputs(hidden_states)
        batch, sequence = query_latent.shape[:2]
        query = self.q_b_proj(query_latent).reshape(batch, sequence, self.num_heads, self.qk_head_dim)
        query_pass, query_rope = query.split((self.qk_nope_head_dim, self.qk_rope_head_dim), dim=-1)
        kv_latent = kv_latent.reshape(batch, 1, sequence, self.kv_lora_rank)
        kv_states = self.kv_b_proj(kv_latent).view(
            batch, sequence, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        key_pass, value = kv_states.split((self.qk_nope_head_dim, self.v_head_dim), dim=-1)
        cos, sin = position_embeddings
        query = torch.cat((query_pass.transpose(1, 2),
                           self.explicit_rotary(query_rope.transpose(1, 2), cos, sin)), dim=-1)
        key_rope = self.explicit_rotary(key_rope.unsqueeze(1), cos, sin)
        key = torch.cat((key_pass, MFKeyExpand.apply(key_rope, self.num_heads,
                                  getattr(self, 'alignment_key_dvm_sum', False))), dim=-1)
        return query, key, value


class SeparateLatentProjection(nn.Linear):
    """Keep the checkpoint's fused layout while using MF's two matmul shapes."""

    def forward(self, values):
        query = F.linear(values, self.weight[:self.query_rank])
        key_value = F.linear(values, self.weight[self.query_rank:])
        return torch.cat((query, key_value), dim=-1)


class MFCombine(torch.autograd.Function):
    """Match native MF fused forward and BF16 probability-gradient reductions."""

    @staticmethod
    def forward(ctx, outputs, probabilities, round_probabilities):
        ctx.round_probabilities = round_probabilities
        ctx.save_for_backward(outputs, probabilities)
        return (outputs.float() * probabilities.float().unsqueeze(-1)).sum(1).to(outputs.dtype)

    @staticmethod
    def backward(ctx, gradient):
        outputs, probabilities = ctx.saved_tensors
        coefficients = probabilities.to(outputs.dtype) if ctx.round_probabilities else probabilities
        token_gradient = (gradient.float().unsqueeze(1) * coefficients.float().unsqueeze(-1)).to(outputs.dtype)
        product = gradient.unsqueeze(1) * outputs
        probability_gradient = product.float().sum(-1).to(outputs.dtype).to(probabilities.dtype)
        return token_gradient, probability_gradient, None


class MFDVMCombine(MFCombine):
    """Reduce live probability-gradient products using the measured MF SumX tree."""

    @staticmethod
    def backward(ctx, gradient):
        outputs, probabilities = ctx.saved_tensors
        coefficients = probabilities.to(outputs.dtype) if ctx.round_probabilities else probabilities
        token_gradient = (gradient.float().unsqueeze(1) * coefficients.float().unsqueeze(-1)).to(outputs.dtype)
        product = gradient.unsqueeze(1) * outputs
        probability_gradient = _dvm_sumx(product.float()).to(outputs.dtype).to(probabilities.dtype)
        return token_gradient, probability_gradient, None


class MFOutputHeadMatMul(torch.autograd.Function):
    """Keep live BF16 linear math while orienting weight gradients like native MF."""

    @staticmethod
    def forward(ctx, values, weight):
        ctx.save_for_backward(values, weight)
        return F.linear(values, weight)

    @staticmethod
    def backward(ctx, gradient):
        values, weight = ctx.saved_tensors
        dy = gradient.reshape(-1, gradient.shape[-1])
        inputs = values.reshape(-1, values.shape[-1])
        dx = torch.mm(dy, weight).reshape(values.shape) if ctx.needs_input_grad[0] else None
        dw = torch.mm(dy.transpose(0, 1), inputs) if ctx.needs_input_grad[1] else None
        return dx, dw


class MFOutputHead(nn.Linear):
    """Cast per projection and accumulate the two head gradients in FP32 masters."""
    keep_compute_in_fp32 = True

    def forward(self, values):
        weight = self.weight.to(values.dtype)
        result = (MFOutputHeadMatMul.apply(values, weight)
                  if getattr(self, 'alignment_native_gradient', False) else F.linear(values, weight))
        directory = getattr(self, 'alignment_head_trace_directory', None)
        if directory is not None:
            call = getattr(self, 'alignment_head_trace_call', 0)
            self.alignment_head_trace_call = call + 1
            if call < 2:
                prefix = Path(directory) / f'head-{call}'
                np.save(str(prefix) + '-inputs.npy', values.detach().float().cpu().numpy())
                metadata = {name: {'shape': list(value.shape), 'stride': list(value.stride()),
                                   'dtype': str(value.dtype), 'format': torch_npu.get_npu_format(value)}
                            for name, value in [('inputs', values), ('weight', weight), ('logits', result)]}
                Path(str(prefix) + '-metadata.json').write_text(json.dumps(metadata, indent=2))

                def observe(suffix):
                    def hook(gradient):
                        np.save(str(prefix) + suffix, gradient.detach().float().cpu().numpy())
                        return gradient
                    return hook

                result.register_hook(observe('-dlogits.npy'))
                weight.register_hook(observe('-weight-gradient.npy'))
        return result


class MFResidualDecoder(jt.JTDeepseekV3Decoder):
    """Round the fused MoE/shared/residual sum once, as observed in MF Graph O1."""

    def forward(self, hidden_states, *args, **kwargs):
        result = super().forward(hidden_states, *args, **kwargs)
        result = result.to(hidden_states.dtype)
        return result.float() if getattr(self, "alignment_return_fp32", False) else result


class ObservedMoE(jt.JTDeepseekV3MoE):
    """Independently switch expert kernels and the weighted reduction precision."""

    def forward(self, hidden_states):
        if not getattr(self, 'alignment_residual_fp32', False):
            return super().forward(hidden_states)
        if getattr(self, 'alignment_input_grad_fp32', False):
            hidden_states = hidden_states.float()
        hidden = hidden_states
        if self.padding:
            hidden = torch.cat((hidden.new_zeros(1, self.padding, hidden.shape[-1]), hidden), dim=1)
        routed = self.ep_compute(hidden)[:, self.padding:]
        shared = self.shared_experts(hidden_states)
        return routed.float() + shared.float()

    def route(self, hidden):
        if not (getattr(self, 'alignment_teacher_routes', False) or
                getattr(self, 'alignment_shared_router_sigmoid', False) or
                getattr(self, 'alignment_router_ir_gradient', False)):
            return super().route(hidden)
        padding, config = self.padding, self.config
        self.gate(hidden[:, padding:])
        logits = self.gate.pop_router_logits().float()
        scores = MFSigmoid.apply(logits) if getattr(self, 'alignment_router_ir_gradient', False) else logits.sigmoid()
        if getattr(self, 'alignment_teacher_routes', False):
            path = Path('/alignment/runs/mf-routing/routes') / f'{self.alignment_route_step:06d}.npz'
            with np.load(path, allow_pickle=False) as archive:
                index_array = archive[self.alignment_route_name + '.indices'].reshape(-1, config.num_experts_per_tok)
            indices = torch.from_numpy(index_array.astype(np.int64)).to(hidden.device)
            self.alignment_route_step += 1
        else:
            if config.n_group != 1:
                raise ValueError('Shared-sigmoid probe requires the reference single router group')
            selection = scores + self.gate.e_score_correction_bias
            indices = selection.topk(config.num_experts_per_tok, dim=-1, sorted=True).indices
        if getattr(self, 'alignment_router_ir_gradient', False):
            if not config.norm_topk_prob:
                raise ValueError('Router IR probe requires normalized top-k weights')
            selected, self.auxiliary_loss = MFRouterWeights.apply(
                scores, indices, config.routed_scaling_factor, config.moe_aux_loss_coeff)
        else:
            selected = scores.gather(1, indices)
            if config.norm_topk_prob:
                selected = selected / (selected.sum(-1, keepdim=True) + 1e-20)
            selected = selected * config.routed_scaling_factor
            self.auxiliary_loss = self.routing_statistics(indices, scores)
        flat_indices = indices.reshape(-1)
        counts = flat_indices.new_zeros(config.n_routed_experts, dtype=torch.float32).scatter_add_(
            0, flat_indices, flat_indices.new_ones(flat_indices.shape, dtype=torch.float32))
        self.tokens_per_expert = counts if self.tokens_per_expert is None else self.tokens_per_expert + counts
        if padding:
            pad_ids = torch.arange(padding * config.num_experts_per_tok, device=indices.device)
            indices = torch.cat((pad_ids.reshape(padding, config.num_experts_per_tok) % padding, indices))
            selected = torch.cat((selected.new_zeros(padding, selected.shape[-1]), selected))
        return indices, selected

    def local_routed_forward(self, hidden):
        if getattr(self, 'alignment_route_input_grad_fp32', False):
            hidden = hidden.float()
        indices, probabilities = self.route(hidden)
        flat = hidden.reshape(-1, hidden.shape[-1])
        expert_flat = flat if getattr(self, 'alignment_cast_after_gather', False) else flat.to(self.experts.gate_up_proj.dtype)
        if self.alignment_grouped:
            if getattr(self, 'alignment_slot_major', False):
                count, top_k = indices.shape
                slot_order = indices.transpose(0, 1).reshape(-1).argsort(stable=True)
                order = (slot_order % count) * top_k + slot_order // count
            else:
                order = indices.reshape(-1).argsort(stable=True)
            counts = torch.bincount(indices.reshape(-1), minlength=self.experts.num_experts)
            grouped = self.experts.forward_expert_major(
                expert_flat[order // indices.shape[1]].to(self.experts.gate_up_proj.dtype), counts)
            directory = getattr(self, 'alignment_backward_directory', None)
            if directory is not None:
                prefix = directory / self.alignment_backward_name
                np.save(str(prefix) + '_expert_major_input.npy', flat[order // indices.shape[1]].detach().float().cpu().numpy())
                np.save(str(prefix) + '_expert_counts.npy', counts.detach().cpu().numpy())
                np.save(str(prefix) + '_expert_major_output.npy', grouped.detach().float().cpu().numpy())
                def record_expert_gradient(gradient):
                    np.save(str(prefix) + '_expert_major_gradient.npy', gradient.detach().float().cpu().numpy())
                    return gradient
                grouped.register_hook(record_expert_gradient)
            restored = torch.empty_like(grouped).index_copy(0, order, grouped)
            outputs = restored.reshape(indices.shape[0], indices.shape[1], flat.shape[-1])
        else:
            outputs = flat.new_zeros(indices.shape[0], indices.shape[1], flat.shape[-1])
            for expert in range(self.experts.num_experts):
                tokens, slots = torch.where(indices == expert)
                pair = F.linear(flat[tokens], self.experts.gate_up_proj[expert])
                if self.alignment_swiglu:
                    values = torch_npu.npu_swiglu(pair, dim=-1)
                else:
                    gate, up = pair.chunk(2, dim=-1)
                    values = F.silu(gate) * up
                values = F.linear(values, self.experts.down_proj[expert])
                outputs = outputs.index_put((tokens, slots), values)
        if getattr(self, 'alignment_combine_gradient', False):
            combine_type = MFDVMCombine if getattr(self, 'alignment_combine_dvm', False) else MFCombine
            result = combine_type.apply(outputs, probabilities, self.alignment_gradient_round_probs)
        elif self.alignment_combine:
            result = (outputs.float() * probabilities.float().unsqueeze(-1)).sum(1).to(hidden.dtype)
        else:
            result = (outputs * probabilities.to(outputs.dtype).unsqueeze(-1)).sum(1)
        return result.reshape(hidden.shape)


def configure_precision(model, factors):
    """Select factors before FSDP creates its compute parameters."""
    allowed = {"norm", "router", "embedding", "rope", "mtp-global", "swiglu", "grouped", "combine",
               "qkv-split", "dense-interleaved", "teacher-routes", "sync-head", "jit", "moe-residual", "head-fp32-grad", "combine-grad", "combine-native-grad", "slot-major", "shared-grad", "residual-norm-grad", "lm-mtp-norm-grad", "moe-input-grad", "graph-context-grad", "silu-ir-grad", "route-input-grad", "ce-mf", "ce-dvm", "teacher-ce-logp", "key-reduce-fp32", "router-shared-sigmoid", "router-ir-grad", "norm-dvm", "head-native-grad", "router-contiguous-grad", "key-dvm-sum", "combine-dvm"}
    if set(factors) - allowed:
        raise ValueError(f"Unknown precision factors: {set(factors) - allowed}")
    if 'jit' in factors:
        torch_npu.npu.set_compile_mode(jit_compile=True)
    if 'ce-mf' in factors:
        model.loss_function = mf_graph_cross_entropy
    if 'ce-dvm' in factors:
        model.loss_function = mf_dvm_cross_entropy
    if 'teacher-ce-logp' in factors:
        model.loss_function = mf_reference_cross_entropy
    if 'head-fp32-grad' in factors:
        model.lm_head.__class__ = MFOutputHead
        model.lm_head.alignment_native_gradient = 'head-native-grad' in factors
    model.alignment_rope_fp32 = "rope" in factors
    model.alignment_mtp_global = "mtp-global" in factors
    for module in model.modules():
        if "norm" in factors and type(module) is jt.DeepseekV32RMSNorm:
            module.__class__ = MFNorm
        if "router" in factors and type(module) is jt.JTDeepseekV3Gate:
            module.__class__ = MFRouter
            module.alignment_contiguous_gradient = 'router-contiguous-grad' in factors
        if "embedding" in factors and isinstance(module, nn.Embedding):
            module.keep_compute_in_fp32 = True
        if "rope" in factors and type(module) is jt.JTDeepseekV3RotaryEmbedding:
            module.__class__ = MFRotary
        if "swiglu" in factors and type(module) is jt.DeepseekV32MLP:
            module.__class__ = FusedDenseMLP
            module.alignment_interleaved = "dense-interleaved" in factors
            module.alignment_ir_gradient = "silu-ir-grad" in factors
        if "qkv-split" in factors and isinstance(module, jt.JTDeepseekV3MLAAttention):
            module.linear_qkv.__class__ = SeparateLatentProjection
            module.linear_qkv.query_rank = module.q_lora_rank
        if "key-reduce-fp32" in factors and isinstance(module, jt.JTDeepseekV3MLAAttention):
            module.__class__ = MFKeyReductionMLAAttention
            module.alignment_key_dvm_sum = "key-dvm-sum" in factors
        if set(factors) & {"swiglu", "grouped", "combine"} and type(module) is jt.JTDeepseekV3MoE:
            module.__class__ = ObservedMoE
            module.alignment_grouped = "grouped" in factors
            module.alignment_shared_router_sigmoid = "router-shared-sigmoid" in factors
            module.alignment_router_ir_gradient = "router-ir-grad" in factors
            module.alignment_input_grad_fp32 = "moe-input-grad" in factors
            module.alignment_route_input_grad_fp32 = "route-input-grad" in factors
            module.alignment_slot_major = "slot-major" in factors
            module.alignment_swiglu = "swiglu" in factors
            module.alignment_combine = "combine" in factors
            module.alignment_combine_dvm = "combine-dvm" in factors
            module.alignment_combine_gradient = bool(set(factors) & {"combine-grad", "combine-native-grad"})
            module.alignment_gradient_round_probs = "combine-native-grad" in factors
            module.ep_compute = module.local_routed_forward
    if 'shared-grad' in factors:
        for name, module in model.named_modules():
            if isinstance(module, FusedDenseMLP) and name.endswith('shared_experts'):
                module.alignment_shared_gradient = True
    if 'graph-context-grad' in factors:
        for name, module in model.named_modules():
            if isinstance(module, ObservedMoE):
                module.alignment_gradient_round_probs = name.startswith('mtp.')
            if isinstance(module, FusedDenseMLP) and name.endswith('shared_experts'):
                module.alignment_shared_gradient = name.startswith('mtp.')
    if 'moe-residual' in factors:
        for module in model.modules():
            if type(module) is jt.JTDeepseekV3Decoder:
                module.__class__ = MFResidualDecoder
            if isinstance(module, ObservedMoE):
                module.alignment_residual_fp32 = True
    if 'residual-norm-grad' in factors:
        for module in model.modules():
            if isinstance(module, jt.JTDeepseekV3Decoder):
                module.__class__ = MFResidualGradientDecoder
                module.input_layernorm.__class__ = MFDualNorm
                module.post_attention_layernorm.__class__ = MFDualNorm
    if 'lm-mtp-norm-grad' in factors:
        if len(model.mtp.layers) != 1:
            raise ValueError('The branch-merge probe requires the reference single MTP depth')
        model.model.layers[-1].alignment_return_fp32 = True
        model.model.norm.alignment_output_dtype = torch.bfloat16
        model.mtp.layers[0].hnorm.alignment_output_dtype = torch.bfloat16
    if 'teacher-routes' in factors:
        for name, module in model.named_modules():
            if isinstance(module, ObservedMoE):
                module.alignment_teacher_routes = True
                module.alignment_route_step = 200
                prefix = name.replace('model.layers.', 'model.decoder.layers.')
                if prefix.startswith('mtp.'):
                    prefix = 'model.' + prefix
                module.alignment_route_name = prefix + '.router'
    if 'sync-head' in factors:
        def synchronize(module, inputs, output):
            del module, inputs, output
            torch.npu.synchronize()
        model.model.norm.register_forward_hook(synchronize)
        for layer in model.mtp.layers:
            layer.final_layernorm.register_forward_hook(synchronize)


def attach_observations(model, directory):
    """Save the same first-forward row samples used by the MF TensorDump taps."""
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=False)
    calls = {}
    inventory = []
    backward = output.name.startswith("backward")
    deep = output.name == "deep"
    deep_all = deep and output.parent.name.endswith("deepall")

    def save(name, value):
        if not isinstance(value, torch.Tensor) or value.ndim < 2:
            return
        if backward and 'norm' in name and value.requires_grad:
            def record_gradient(gradient):
                np.save(output / (name + '_gradient.npy'), gradient.detach().float().cpu().numpy())
                return gradient
            value.register_hook(record_gradient)
        rows = value.detach().reshape(-1, value.shape[-1])
        sample = torch.cat((rows[:128], rows[2240:2260])).float().cpu().numpy()
        np.save(output / (name + ".npy"), sample)
        if (name.endswith('.gate') or name in {'model.norm', 'mtp.layers.0.final_layernorm'}
                or (deep and (name.startswith('model.layers.0.') or deep_all))):
            np.save(output / (name + '_full.npy'), value.detach().float().cpu().numpy())

    def observer(name):
        def record(module, inputs, result):
            del module
            count = calls.get(name, 0)
            calls[name] = count + 1
            if count:
                return
            if inputs:
                save(name + "_input", inputs[0])
            save(name, result[0] if isinstance(result, tuple) else result)
        return record

    for name, module in model.named_modules():
        inventory.append({"name": name, "class": type(module).__name__})
        if name and name != "lm_head" and not isinstance(module, (nn.ModuleList, nn.ModuleDict)):
            module.register_forward_hook(observer(name))
        if isinstance(module, FusedDenseMLP) and output.name == 'backwarddense':
            module.alignment_backward_directory = output
            module.alignment_backward_name = name
        if isinstance(module, jt.JTDeepseekV3MoE):
            if output.name == 'backwardmoe':
                module.alignment_backward_directory = output
                module.alignment_backward_name = name
            original_route = module.route

            def capture_route(owner, hidden, native=original_route, key=name):
                del owner
                indices, probabilities = native(hidden)
                if backward and probabilities.requires_grad:
                    def record_probability(gradient):
                        np.save(output / (key + '_probability_gradient.npy'), gradient.detach().float().cpu().numpy())
                        return gradient
                    probabilities.register_hook(record_probability)
                if not (output / (key + '_indices.npy')).exists():
                    np.save(output / (key + '_indices.npy'), indices.detach().cpu().numpy())
                    np.save(output / (key + '_probabilities.npy'), probabilities.detach().float().cpu().numpy())
                return indices, probabilities

            module.route = MethodType(capture_route, module)
        if isinstance(module, jt.JTDeepseekV3MLAAttention):
            original_projection = module._project_attention_inputs

            def capture_attention(owner, *args, native=original_projection, key=name, **kwargs):
                del owner
                projected = native(*args, **kwargs)
                for label, tensor in zip(('query', 'key', 'value'), projected):
                    path = output / (key + '_' + label + '.npy')
                    if not path.exists():
                        sample = tensor.detach().permute(2, 0, 1, 3)[:128].float().cpu().numpy()
                        np.save(path, sample)
                        if deep and (key.startswith("model.layers.0.") or deep_all):
                            full = tensor.detach().permute(2, 0, 1, 3).float().cpu().numpy()
                            np.save(output / (key + "_" + label + "_full.npy"), full)
                return projected

            module._project_attention_inputs = MethodType(capture_attention, module)
    (output / "modules.json").write_text(json.dumps(inventory, indent=2))
