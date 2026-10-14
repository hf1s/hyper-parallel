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
"""One alignment step 301 with both known fixes plus operator-boundary taps on the MTP chain.

The two fixes are the ones :mod:`entry_bwd` established: ``CLOSE_MATMUL_K_SHIFT`` is popped
after the trainer has been constructed, and the named losses are re-associated to MindFormers'
``(lm + aux) + mtp``.  Nothing else about the run differs from a stock step 301; the whole
difference is :mod:`mtp_taps`, which records the forward tensors and backward gradients on the
MTP operator boundaries and then publishes them for the MF/HP comparison.
"""

import json
import math
import os
import sys
from pathlib import Path

WORKTREE = Path('/workspace/worktree/align-l2-f')
EXAMPLE = WORKTREE / 'examples' / 'training_demo' / 'align-l2-f'
OVERLAY = Path('/alignment/tools')
MINE = Path('/workspace/align-work/MTPChainHP')
SWITCH = 'CLOSE_MATMUL_K_SHIFT'
MF_ORDER = ('lm', 'aux', 'mtp')
PUBLISH = '/home/zth/tmp/align-cmp/bwdtaps/hp'


def _configure_import_path():
    """Resolve this directory, the overlay, the example and the worktree first."""
    for entry in (str(MINE), str(OVERLAY), str(EXAMPLE), str(WORKTREE)):
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)


def _install_mf_loss_association():
    """Sum the named losses in MindFormers' order, without touching framework source."""
    import hyper_parallel.trainer.base as trainer_base  # pylint: disable=C0415

    original = trainer_base.BaseTrainer.postforward

    def postforward(self, outputs, labels):
        """Keep every value the trainer produced and only re-associate their sum."""
        _, loss_dict = original(self, outputs, labels)
        values = []
        for token in MF_ORDER:
            matches = [value for name, value in loss_dict.items() if token in name]
            if len(matches) != 1:
                print(f'[MTPTAP] association fallback: {token} matched {len(matches)} names',
                      flush=True)
                return original(self, outputs, labels)[0], loss_dict
            values.append(matches[0])
        return (values[0] + values[1]) + values[2], loss_dict

    trainer_base.BaseTrainer.postforward = postforward
    print('[MTPTAP] loss association overridden to (lm + aux) + mtp', flush=True)


def _describe(model):
    """Print the MTP module inventory and the head/loss types as evidence."""
    lines = []
    for name, module in model.named_modules():
        if 'mtp' in name:
            epsilon = getattr(module, 'variance_epsilon', getattr(module, 'eps', None))
            lines.append({'name': name, 'class': type(module).__name__,
                          'eps': None if epsilon is None else float(epsilon)})
    norms = {}
    for name in ('model.norm', 'mtp.layers.0.hnorm', 'mtp.layers.0.enorm',
                 'mtp.layers.0.final_layernorm'):
        try:
            module = model.get_submodule(name)
        except AttributeError:
            continue
        norms[name] = {'class': type(module).__name__,
                       'mro': [klass.__name__ for klass in type(module).__mro__],
                       'keep_compute_in_fp32': getattr(module, 'keep_compute_in_fp32', None),
                       'alignment_output_dtype': str(
                           getattr(module, 'alignment_output_dtype', None))}
    payload = {'mtp_modules': lines,
               'norms': norms,
               'same_class_hnorm_enorm': (
                   type(model.mtp.layers[0].hnorm) is type(model.mtp.layers[0].enorm)),
               'lm_head_class': type(model.lm_head).__name__,
               'loss_function': getattr(model.loss_function, '__name__',
                                        type(model.loss_function).__name__),
               'loss_chunk_size': model.loss_chunk_size,
               'mtp_depths': len(model.mtp.layers)}
    run_dir = Path(os.environ['JT_RUN_DIR'])
    (run_dir / 'mtp_inventory.json').write_text(json.dumps(payload, indent=2) + '\n')
    for line in lines:
        print(f'[MTPTAP] module {line["name"]:44s} {line["class"]:22s} eps={line["eps"]}',
              flush=True)
    for name, info in norms.items():
        print(f'[MTPTAP] norm {name:32s} class={info["class"]:16s} '
              f'fp32={info["keep_compute_in_fp32"]} out_dtype={info["alignment_output_dtype"]} '
              f'mro={info["mro"][:4]}', flush=True)
    print(f'[MTPTAP] hnorm/enorm same class = {payload["same_class_hnorm_enorm"]}', flush=True)
    print(f'[MTPTAP] {payload["lm_head_class"]} / {payload["loss_function"]} / '
          f'chunk={payload["loss_chunk_size"]} depths={payload["mtp_depths"]}', flush=True)


def _grad_candidates(trainer):
    """Return the parameter iterators that can hold gradients in this trainer."""
    holders = []
    for attribute in ('hsdp_model_parts', 'model_parts', 'model'):
        value = getattr(trainer, attribute, None)
        if value is None:
            continue
        holders.extend(value if isinstance(value, (list, tuple)) else [value])
    return holders


def _global_norm_fp64(trainer):
    """Return the fp64 L2 norm of every visible parameter gradient, in blocks."""
    total = 0.0
    count = 0
    seen = set()
    sources = {}
    for holder in _grad_candidates(trainer):
        for name, parameter in holder.named_parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            gradient = getattr(parameter, 'grad', None)
            slot = 'grad'
            if gradient is None:
                gradient = getattr(parameter, 'main_grad', None)
                slot = 'main_grad'
            if gradient is None or not hasattr(gradient, 'numel'):
                sources.setdefault('without_grad', 0)
                sources['without_grad'] += 1
                continue
            sources[slot] = sources.get(slot, 0) + 1
            flat = mtp_taps._local(gradient).detach().float().reshape(-1)  # pylint: disable=protected-access
            for offset in range(0, flat.numel(), 1 << 22):
                block = flat[offset:offset + (1 << 22)].double()
                total += float((block * block).sum())
            count += flat.numel()
    return math.sqrt(total), count, sources


def _install_grad_norm_record(trainer, pending):
    """Record the fp64 gradient norm next to the trainer's own device norm."""
    import hyper_parallel.trainer.base as trainer_base  # pylint: disable=C0415

    original = trainer_base.BaseTrainer.prepare_optimizer_step

    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            norm, count, sources = _global_norm_fp64(self)
            pending['norm_fp64'] = norm
            pending['numel_fp64'] = count
            pending['norm_fp64_sources'] = sources
            print(f'[MTPTAP] fp64 grad norm = {norm!r} over {count} elements, sources={sources}',
                  flush=True)
        except Exception as error:  # pylint: disable=broad-except
            print(f'[MTPTAP] fp64 grad norm failed: {type(error).__name__}: {error}', flush=True)
        return result

    trainer_base.BaseTrainer.prepare_optimizer_step = wrapped
    return trainer


def _metrics_callback(trainer, pending):
    """Record the loss, the named losses and both gradient norms of the step."""
    from hyper_parallel.trainer.callbacks.base import Callback  # pylint: disable=C0415

    class Metrics(Callback):
        """Persist what the step produced, without touching it."""

        def on_step_end(self, state, *, loss, loss_dict, grad_norm, **kwargs):
            """Append one row per completed step."""
            del kwargs
            row = {'step': state.global_step, 'loss': loss,
                   'loss_17g': '%.17g' % loss,
                   'grad_norm': grad_norm, 'grad_norm_17g': '%.17g' % grad_norm,
                   'norm_fp64': pending.get('norm_fp64'),
                   'norm_fp64_17g': ('%.17g' % pending['norm_fp64']
                                     if 'norm_fp64' in pending else None),
                   'numel_fp64': pending.get('numel_fp64'),
                   'loss_dict_17g': {name: '%.17g' % value
                                     for name, value in loss_dict.items()}}
            run_dir = Path(os.environ['JT_RUN_DIR'])
            with (run_dir / 'metrics-mtpchain.jsonl').open('a') as stream:
                stream.write(json.dumps(row, default=str) + '\n')
            (run_dir / 'metrics-mtpchain.json').write_text(
                json.dumps(row, indent=2, default=str) + '\n')
            print(f'[MTPTAP] step {state.global_step} loss={row["loss_17g"]} '
                  f'grad_norm={row["grad_norm_17g"]} norm_fp64={row["norm_fp64_17g"]}', flush=True)

    return Metrics(trainer)


def _pin_mtp_scale(model):
    """Optionally pin the MTP loss factor to the float32 nearest 0.3.

    HP's ``mtp_layer_loss__dy`` is ``0.29999998211860657`` - exactly one float32 ulp *below*
    ``0.30000001192092896 = float32(0.3)`` - while MindFormers hands the CE backward
    ``0.30000001192092896``.  The multiplier is ``cfg.mtp_loss_factor / depths``
    (``modeling_jt_deepseek_v3.py:774``), so replacing the config value with the
    float32-representable 0.3 aligns the two constants without touching the repo.
    """
    if os.environ.get('MTPTAP_SCALE_FP32') != '1':
        return None
    import numpy as np  # pylint: disable=C0415
    old = model.config.mtp_loss_factor
    model.config.mtp_loss_factor = float(np.float32(old))
    print(f'[MTPTAP] mtp_loss_factor pinned: {old!r} -> {model.config.mtp_loss_factor!r} '
          f'(float32 = {float(np.float32(old))!r})', flush=True)
    return model.config.mtp_loss_factor


def main():
    """Run one step 301 with the two fixes and the MTP operator-boundary taps."""
    _configure_import_path()
    import hyper_parallel

    resolved = Path(hyper_parallel.__file__).resolve()
    if not resolved.is_relative_to(WORKTREE):
        raise SystemExit(f'hyper_parallel resolved outside the worktree: {resolved}')
    print(f'[MTPTAP] hyper_parallel={resolved}', flush=True)

    _install_mf_loss_association()
    from hyper_parallel.trainer.config.parser import parse_training_args  # pylint: disable=C0415
    from hyper_parallel.trainer.text_trainer import TextTrainer  # pylint: disable=C0415

    trainer = TextTrainer(parse_training_args())
    if os.environ.get('MTPTAP_KEEP_KSHIFT') != '1':
        os.environ.pop(SWITCH, None)
    print(f'[MTPTAP] {SWITCH} cleared before the first matmul = {os.environ.get(SWITCH)!r}',
          flush=True)
    for name in sorted(os.environ):
        if name.startswith(('MTPTAP_', 'BWSW_')) or 'DETERMINIS' in name:
            print(f'[MTPTAP] env {name}={os.environ[name]}', flush=True)

    run_dir = Path(os.environ['JT_RUN_DIR'])
    model = trainer.base.model
    _describe(model)

    import mtp_taps

    store = mtp_taps.attach_trainer(trainer, run_dir / 'bwd')
    mtp_taps.install_alignment(model)
    _pin_mtp_scale(model)
    pending = {}
    _install_grad_norm_record(trainer.base, pending)
    trainer.base._callbacks.insert(0, _metrics_callback(trainer.base, pending))
    try:
        trainer.train()
    finally:
        store.write()
        print(f'[MTPTAP] taps written: {len(store.index)} tensors into {run_dir / "bwd"}',
              flush=True)
        try:
            import publish_mtp

            if os.environ.get('MTPTAP_PUBLISH', '1') == '0':
                print('[MTPTAP] publish skipped (MTPTAP_PUBLISH=0); taps stay in the run dir',
                      flush=True)
            else:
                publish_mtp.publish(run_dir, PUBLISH,
                                    tag=os.environ.get('MTPTAP_TAG', 'mtpchain'))
        except Exception as error:  # pylint: disable=broad-except
            print(f'[MTPTAP] publish failed: {type(error).__name__}: {error}', flush=True)
    print(f'[MTPTAP] finished, run dir {run_dir}', flush=True)


if __name__ == '__main__':
    main()
