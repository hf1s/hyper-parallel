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
"""Alignment run with Fix 1 + Fix 2 + Fix 3 (PR #180), plus the MTP operator-boundary taps.

Fix 1 — clear ``CLOSE_MATMUL_K_SHIFT``: ``enable_full_determinism()`` exports it while the trainer
is constructed, which closes the CANN MatMul K-shift tiling and reorders the K accumulation of the
fused kv projection.  It is popped right after the trainer is built and every later write is
suppressed, so the first matmul compiles the MindFormers kernel.

Fix 2 — MindFormers' loss association: the trainer sums the named losses as ``(lm + mtp) + aux``
(``torch.stack(list(loss_dict.values())).sum()``), MindFormers assembles ``(lm + aux) + mtp``.
The override recomputes ``loss`` from the very same ``loss_dict``, so only the addition order
changes.

Fix 3 — PR #180 (``pr180.py``): the backward scale is formed from the token counts first
(``backward_scale = cur_token_len / all_reduced_len * dp_size * cp_size``) instead of dividing the
already-weighted loss, so a full single-rank microbatch delivers an exact 1.0 gradient to every
named loss.  Applied as a runtime monkey patch; no repository file is touched.

The ``norm-dvm`` factor patch (``mf_dvm_clip.attach``) is attached exactly as the stock entries do;
it is part of the alignment configuration, not one of the fixes.

This copy runs against the patched ``hyper_parallel`` package in ``MINE``
(``/workspace/align-work/MuonFp32UpdateSwitch``), which adds the *config* switches
``optimizer.muon_config.fp32_update``, ``optimizer.muon_config.ns_per_param`` and
``optimizer.adamw_config.adamw_wd_all_params``.  ``fix5`` (the monkey-patched predecessor of
``fp32_update``) is off, so ``PR180_FIX5=0`` selects HP's stock bfloat16 update buffer.

Environment switches (all optional):
  * ``PR180_PATCH=0``  — control run: Fix 1 + Fix 2 only, no PR #180.
  * ``PR180_TAPS=0``   — do not attach the MTP operator-boundary taps (used for the 300-step run).
  * ``PR180_FP64=0``   — do not record the fp64 gradient norm.
  * ``PR180_ASSOC=0``  — do not install the loss association (Fix 2).
  * ``PR180_KSHIFT=keep`` — do not pop ``CLOSE_MATMUL_K_SHIFT`` (Fix 1).
  * ``PR180_FIX5=0``   — do not install the monkey-patched fp32 update (default: installed).
  * ``PR180_TAG``      — label recorded in the metrics file and the launcher output.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

WORKTREE = Path('/workspace/worktree/align-l2-f')
EXAMPLE = WORKTREE / 'examples' / 'training_demo' / 'align-l2-f'
OVERLAY = Path('/alignment/tools')
MINE = Path('/workspace/align-work/MuonFp32UpdateSwitch')
TAPDIR = Path('/workspace/align-work/MTPChainHP')
SWITCH = 'CLOSE_MATMUL_K_SHIFT'
MF_ORDER = ('lm', 'aux', 'mtp')
MF_TAPS = Path('/home/zth/tmp/align-cmp/bwdtaps/mf')


def _enabled(name: str, default: str = '1') -> bool:
    """Return whether an environment switch is on (default: on)."""
    return os.environ.get(name, default) not in ('0', '', 'false', 'False')


def _configure_import_path() -> None:
    """Resolve this directory, the tap helpers, the overlay and the example before the worktree.

    ``MINE`` holds the patched ``hyper_parallel`` package (the Muon alignment switches), so it
    has to be searched before the read-only shared worktree.
    """
    for entry in (str(WORKTREE), str(EXAMPLE), str(OVERLAY), str(TAPDIR), str(MINE)):
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)


def _report_optimizer_switches(trainer, label: str = '') -> dict:
    """Report the Muon/AdamW alignment switches the built optimizer actually carries."""
    state = {'fp32_update': None, 'ns_per_param': None, 'adamw_weight_decay': []}
    chain = getattr(trainer.base, 'optimizer', None)
    for optimizer in getattr(chain, 'chained_optimizers', None) or [chain]:
        if optimizer is None:
            continue
        name = type(optimizer).__name__
        if hasattr(optimizer, 'fp32_update'):
            state['fp32_update'] = bool(optimizer.fp32_update)
            state['ns_per_param'] = bool(getattr(optimizer, 'ns_per_param', None))
            state['update_dtype'] = str(optimizer._update_dtype)  # pylint: disable=W0212
        if hasattr(optimizer, 'param_groups') and 'adamw' in name.lower():
            state['adamw_weight_decay'] = sorted({
                float(group['weight_decay']) for group in optimizer.param_groups
            })
    print(f'[MFS] optimizer switches{label}: {state}', flush=True)
    return state


def _install_mf_loss_association() -> None:
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
                print(f'[PR180] association fallback: {token} matched {len(matches)} names',
                      flush=True)
                return original(self, outputs, labels)[0], loss_dict
            values.append(matches[0])
        loss = (values[0] + values[1]) + values[2]
        return loss, loss_dict

    trainer_base.BaseTrainer.postforward = postforward
    print('[PR180] loss association overridden to (lm + aux) + mtp', flush=True)


def _clear_k_shift(suppress: bool) -> list:
    """Pop the K-shift switch after trainer construction; optionally guard later writes."""
    os.environ.pop(SWITCH, None)
    suppressed = []
    if suppress:
        original_setitem = os._Environ.__setitem__  # pylint: disable=protected-access

        def guarded_setitem(environ, key, value):
            """Keep ``CLOSE_MATMUL_K_SHIFT`` unset for the whole process."""
            if key == SWITCH:
                suppressed.append(value)
                return
            original_setitem(environ, key, value)

        os._Environ.__setitem__ = guarded_setitem  # pylint: disable=protected-access
    print(f'[PR180] {SWITCH}={os.environ.get(SWITCH)!r} popped, '
          f'later writes {"suppressed" if suppress else "allowed"}', flush=True)
    return suppressed


def _configured_factors() -> list:
    """Read the factor list straight from the configuration file as a fallback."""
    import yaml  # pylint: disable=C0415

    return list(yaml.safe_load(Path(sys.argv[1]).read_text())['model'].get('factors', []))


def _attach_norm_dvm(trainer) -> None:
    """Attach the measured single-card MF DVM norm clipping, as the stock entries do."""
    factors = list(getattr(trainer.base.config.model, 'factors', None) or _configured_factors())
    print(f'[PR180] factors={json.dumps(factors)}', flush=True)
    if 'norm-dvm' in factors:
        from mf_dvm_clip import attach  # pylint: disable=C0415

        attach(trainer)
        print('[PR180] attached the measured single-card MF DVM norm clipping', flush=True)


def _install_pr180() -> list:
    """Apply PR #180 only when the exact token-scale switch is explicitly enabled."""
    if not _enabled('PR180_PATCH'):
        print('[PR180] Fix 3 DISABLED: upstream mean_global_loss is used unchanged', flush=True)
        return []
    import pr180  # pylint: disable=C0415

    print(f'[PR180] patch diff: {json.dumps(pr180.describe(), indent=2)}', flush=True)
    return pr180.apply()


def _install_shift_alignment(model) -> list:
    """Fix 4 — MindFormers' MTP document shift: drop ``sequence_end_mask`` on the MTP module only.

    HP shifts the MTP input ids with the same ``shift_mtp_sequence`` MindFormers uses, but it then
    additionally zeroes the shifted ids at every document tail
    (``components/modules/mtp.py:135-137``, ``input_ids.masked_fill(sequence_end_mask, 0)``), so
    those positions embed token 0 instead of the real next token.  MindFormers' roll leaves them
    alone.  Only the mask handed to the *module* is dropped here; the loss path
    (``compute_jt_losses`` / ``iter_mtp_targets``) keeps its own mask, so the targets of those
    positions stay masked at ``-100`` and the loss denominator is untouched.
    """
    if not _enabled('PR180_SHIFT_ALIGN'):
        print('[PR180] Fix 4 DISABLED: the MTP input keeps the document-tail zeroing', flush=True)
        return []
    handles = []
    calls = {'n': 0}

    def drop_mask(module, args, kwargs):  # pylint: disable=unused-argument
        if kwargs.get('sequence_end_mask') is not None:
            calls['n'] += 1
            if calls['n'] <= 2:
                print(f'[PR180] Fix 4: dropping sequence_end_mask on '
                      f'{type(module).__name__} (call {calls["n"]})', flush=True)
            kwargs = dict(kwargs, sequence_end_mask=None)
        return args, kwargs

    targets = [(name, module) for name, module in model.named_modules()
               if name.replace('_checkpoint_wrapped_module.', '') == 'mtp']
    if not targets:
        fallback = getattr(model, 'mtp', None)
        if fallback is None:
            raise SystemExit('Fix 4: no mtp module found on the model')
        targets = [('mtp (attribute)', fallback)]
    for name, module in targets:
        handles.append(module.register_forward_pre_hook(drop_mask, with_kwargs=True))
        print(f'[PR180] Fix 4: pre-hook on {name} ({type(module).__name__})', flush=True)
    return handles


def _grad_candidates(trainer):
    """Return every holder that may own parameters of the trainer (those exposing parameters)."""
    holders = []
    for attribute in ('model', 'optimizer', 'scheduler', 'modules', '_modules', 'engine',
                      'model_wrapper', '_model'):
        value = getattr(trainer, attribute, None)
        if value is None:
            continue
        for item in (value if isinstance(value, (list, tuple)) else [value]):
            if hasattr(item, 'named_parameters'):
                holders.append(item)
            else:
                print(f'[PR180] grad holder {attribute} has no named_parameters '
                      f'({type(item).__name__})', flush=True)
    return holders


def _global_norm_fp64(trainer):
    """Return the fp64 L2 norm of every visible parameter gradient, in blocks."""
    import mtp_taps  # pylint: disable=C0415

    total = 0.0
    count = 0
    seen = set()
    sources = {}
    for holder in _grad_candidates(trainer):
        for _, parameter in holder.named_parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            gradient = getattr(parameter, 'grad', None)
            slot = 'grad'
            if gradient is None:
                gradient = getattr(parameter, 'main_grad', None)
                slot = 'main_grad'
            if gradient is None or not hasattr(gradient, 'numel'):
                sources['without_grad'] = sources.get('without_grad', 0) + 1
                continue
            sources[slot] = sources.get(slot, 0) + 1
            flat = mtp_taps._local(gradient).detach().float().reshape(-1)  # pylint: disable=W0212
            for offset in range(0, flat.numel(), 1 << 22):
                block = flat[offset:offset + (1 << 22)].double()
                total += float((block * block).sum())
            count += flat.numel()
    return math.sqrt(total), count, sources


def _install_grad_norm_record(pending) -> None:
    """Record the fp64 gradient norm before and after the optimizer step's clipping.

    ``clip_grad_norm_`` multiplies every gradient by ``1 / max(total, 1)``, so the *post* norm is
    1.0 by construction; the number that can be compared with MindFormers is the **pre-clip** one,
    which is measured only on the first step (the whole 2.7e9-element gradient in fp64 is not free).
    """
    import hyper_parallel.trainer.base as trainer_base  # pylint: disable=C0415

    original = trainer_base.BaseTrainer.prepare_optimizer_step

    def wrapped(self, *args, **kwargs):
        if 'norm_fp64_preclip' not in pending:
            try:
                norm, count, sources = _global_norm_fp64(self)
                pending['norm_fp64_preclip'] = norm
                pending['numel_fp64_preclip'] = count
                pending['norm_fp64_preclip_sources'] = sources
                print(f'[PR180] fp64 PRE-clip grad norm = {norm!r} over {count} elements, '
                      f'sources={sources}', flush=True)
            except Exception as error:  # pylint: disable=broad-except
                print(f'[PR180] fp64 pre-clip norm failed: {type(error).__name__}: {error}',
                      flush=True)
        result = original(self, *args, **kwargs)
        if 'norm_fp64' not in pending:
            try:
                norm, count, sources = _global_norm_fp64(self)
                pending['norm_fp64'] = norm
                pending['numel_fp64'] = count
                pending['norm_fp64_sources'] = sources
                print(f'[PR180] fp64 POST-clip grad norm = {norm!r} over {count} elements, '
                      f'sources={sources}', flush=True)
            except Exception as error:  # pylint: disable=broad-except
                print(f'[PR180] fp64 post-clip norm failed: {type(error).__name__}: {error}',
                      flush=True)
        return result

    trainer_base.BaseTrainer.prepare_optimizer_step = wrapped


def _metrics_callback(trainer, pending) -> object:
    """Record the loss, the loss components and both gradient norms of each step."""
    from hyper_parallel.trainer.callbacks.base import Callback  # pylint: disable=C0415

    class Metrics(Callback):
        """Persist the values the run is judged on, without touching them."""

        def on_step_end(self, state, *, loss, loss_dict, grad_norm, **kwargs):
            """Append one row per completed step."""
            del kwargs
            values = [loss, grad_norm, *loss_dict.values()]
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f'Nonfinite result at step {state.global_step}')
            row = {'step': state.global_step, 'loss': float(loss),
                   'loss_17g': '%.17g' % float(loss),
                   'grad_norm': float(grad_norm), 'grad_norm_17g': '%.17g' % float(grad_norm),
                   'norm_fp64': pending.get('norm_fp64'),
                   'norm_fp64_17g': ('%.17g' % pending['norm_fp64']
                                     if 'norm_fp64' in pending else None),
                   'norm_fp64_preclip': pending.get('norm_fp64_preclip'),
                   'norm_fp64_preclip_17g': ('%.17g' % pending['norm_fp64_preclip']
                                             if 'norm_fp64_preclip' in pending else None),
                   'numel_fp64': pending.get('numel_fp64'),
                   'loss_dict': {name: float(value) for name, value in loss_dict.items()},
                   'loss_dict_17g': {name: '%.17g' % float(value)
                                     for name, value in loss_dict.items()}}
            run_dir = Path(os.environ['JT_RUN_DIR'])
            with (run_dir / 'metrics-pr180.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            (run_dir / 'metrics-pr180-last.json').write_text(json.dumps(row, indent=2) + '\n')
            print(f'[PR180] step {state.global_step} loss={row["loss_17g"]} '
                  f'grad_norm={row["grad_norm_17g"]} norm_fp64={row["norm_fp64_17g"]}', flush=True)

    return Metrics(trainer)


def main() -> None:
    """Run the session with Fix 1 + Fix 2 + Fix 3 and the optional taps."""
    _configure_import_path()
    import hyper_parallel  # pylint: disable=C0415

    resolved = Path(hyper_parallel.__file__).resolve()
    if not resolved.is_relative_to(MINE):
        raise SystemExit(f'hyper_parallel resolved outside the overlay: {resolved}')
    print(f'[PR180] hyper_parallel={resolved}', flush=True)
    if _enabled('PR180_ASSOC'):
        _install_mf_loss_association()
    from hyper_parallel.trainer.config.parser import parse_training_args  # pylint: disable=C0415
    from hyper_parallel.trainer.text_trainer import TextTrainer  # pylint: disable=C0415

    trainer = TextTrainer(parse_training_args())
    dy_tap_dir = os.environ.get('DY_TAP_DIR')
    if dy_tap_dir:
        import dy_tap  # pylint: disable=C0415
        dy_tap.install(trainer.base.model, dy_tap_dir,
                       step=int(os.environ.get('DY_TAP_STEP', '301')))
    if os.environ.get('PR180_KSHIFT') == 'keep':
        print(f'[PR180] Fix 1 DISABLED: {SWITCH}={os.environ.get(SWITCH)!r} kept', flush=True)
        suppressed = []
    else:
        suppressed = _clear_k_shift(True)
    _attach_norm_dvm(trainer)
    patched_modules = _install_pr180()
    shift_handles = _install_shift_alignment(trainer.base.model)
    fix5_state = {}
    if _enabled('PR180_FIX5'):
        import fix5  # pylint: disable=C0415

        fix5.apply(xret_numel=fix5.xret_numel_from_env())
        fix5_state = fix5
    else:
        print('[PR180] Fix 5 DISABLED: the Muon scale stays in bfloat16 (upstream behaviour)',
              flush=True)
    param_dump = os.environ.get('PR180_PARAM_DUMP')
    if param_dump:
        import param_native  # pylint: disable=C0415

        param_native.install(trainer.base, Path(param_dump), step=os.environ.get('PR180_TAG'))
    run_dir = Path(os.environ['JT_RUN_DIR'])
    store = None
    if _enabled('PR180_TAPS'):
        import mtp_taps  # pylint: disable=C0415

        store = mtp_taps.attach_trainer(trainer, run_dir / 'bwd')
    pending = {}
    if _enabled('PR180_FP64'):
        _install_grad_norm_record(pending)
    grad_dump = os.environ.get('PR180_GRAD_DUMP')
    if grad_dump:
        import grad_native  # pylint: disable=C0415

        grad_native.install(trainer.base, Path(grad_dump),
                            save_raw=os.environ.get('PR180_GRAD_RAW') == '1')
    trainer.base._callbacks.insert(0, _metrics_callback(trainer.base, pending))
    switch_state = _report_optimizer_switches(trainer)
    (run_dir / 'entry_state.json').write_text(json.dumps({
        'tag': os.environ.get('PR180_TAG', 'pr180'),
        'patch_applied': bool(patched_modules),
        'patched_modules': patched_modules,
        'fix1_k_shift_cleared': os.environ.get('PR180_KSHIFT') != 'keep',
        'fix2_association': _enabled('PR180_ASSOC'),
        'fix4_shift_alignment': bool(shift_handles),
        'fix5_fp32_update': bool(fix5_state),
        'fix5_captured': (fix5_state.captured_summary() if fix5_state else None),
        'muon_switches': switch_state,
        'taps': store is not None,
        'fp64_norm': _enabled('PR180_FP64'),
        'norm_dvm_attached': True,
        'mf_taps_dir': str(MF_TAPS),
    }, indent=2) + '\n')
    try:
        trainer.train()
    finally:
        if store is not None:
            store.write()
            print(f'[PR180] taps written: {len(store.index)} tensors into {run_dir / "bwd"}',
                  flush=True)
        if fix5_state:
            summary = fix5_state.captured_summary()
            (run_dir / 'fix5_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            print(f'[PR180] Fix 5 summary: {summary}', flush=True)
            xret_dir = os.environ.get('PR180_XRET_DUMP')
            if xret_dir:
                fix5_state.dump_captured(Path(xret_dir))
        print(f'[PR180] suppressed writes of {SWITCH}: {suppressed}', flush=True)
        _report_optimizer_switches(trainer, ' after training')
    print(f'[PR180] finished, run dir {run_dir}', flush=True)


if __name__ == '__main__':
    main()
