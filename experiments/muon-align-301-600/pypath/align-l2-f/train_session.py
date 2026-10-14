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
"""Run the alignment session from this worktree and report its factor replacements.

The shared experiment entry runs the plain Trainer. This entry first counts how many
modules every factor of the alignment overlay rewrites, prints that inventory, and
then starts the identical training loop.
"""

import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

WORKTREE = Path(__file__).resolve().parents[3]
OVERLAY = Path('/alignment/tools')


def _configure_import_path() -> None:
    """Resolve this worktree and the alignment overlay ahead of installed packages."""
    for entry in (str(OVERLAY), str(WORKTREE)):
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)


def _require_worktree_runtime() -> None:
    """Fail fast when ``hyper_parallel`` does not come from this worktree.

    Hugging the editable install pointed at another checkout is the usual reason a
    session silently trains code that is not the reviewed revision.
    """
    import hyper_parallel

    resolved = Path(hyper_parallel.__file__).resolve()
    if not resolved.is_relative_to(WORKTREE):
        raise SystemExit(f'hyper_parallel resolved outside the worktree: {resolved}')
    print(f'[L2] hyper_parallel={resolved}', flush=True)


def _factor_changes(configure_precision, model, factors):
    """Count the modules every factor rewrites by replaying it on a model copy.

    The overlay replaces module classes in place, so applying growing prefixes of the
    factor list to a copy shows exactly which modules each factor claims. The copy is
    made while the model still lives on the meta device, and the trained model is only
    configured afterwards by the original single call.

    Args:
        configure_precision: The overlay's factor selection function.
        model: Meta-device model the overlay is about to configure.
        factors: Configured factor names, in file order.

    Returns:
        One row per factor with the number and identity of its replaced modules.
    """
    probe = copy.deepcopy(model)
    pristine = {name: type(module).__name__ for name, module in probe.named_modules()}
    previous = dict(pristine)
    rows = []
    for index, factor in enumerate(factors):
        configure_precision(probe, list(factors)[:index + 1])
        current = {name: type(module).__name__ for name, module in probe.named_modules()}
        changes = {name: f'{pristine[name]}->{current[name]}'
                   for name in current if current[name] != previous[name]}
        rows.append({'factor': factor, 'replaced': len(changes), 'modules': changes})
        previous = current
    return rows


def _report(rows) -> None:
    """Print the per-factor inventory and persist it next to the run log."""
    counts = {row['factor']: row['replaced'] for row in rows}
    print(f'[L2-FACTORS] {json.dumps(counts)}', flush=True)
    for row in rows:
        print(f"[L2-FACTOR] {row['factor']}: {row['replaced']} modules "
              f"{json.dumps(row['modules'], sort_keys=True)}", flush=True)
    total = sum(row['replaced'] for row in rows)
    swapped_nothing = [row['factor'] for row in rows if not row['replaced']]
    print(f'[L2-FACTORS-TOTAL] replaced={total} '
          f'no_module_swap={json.dumps(swapped_nothing)}', flush=True)
    run_dir = os.environ.get('JT_RUN_DIR')
    if run_dir:
        Path(run_dir, 'factor-report.json').write_text(
            json.dumps({'factors': rows, 'total_replaced': total,
                        'no_module_swap': swapped_nothing}, indent=2) + '\n')


def _install_factor_report() -> None:
    """Wrap the overlay's factor selection so it reports its replacements once."""
    import precision_probe

    original = precision_probe.configure_precision

    def configure_precision(model, factors):
        """Report the factor inventory on rank zero, then configure the real model."""
        factors = list(factors)
        if os.environ.get('RANK', '0') == '0':
            try:
                _report(_factor_changes(original, model, factors))
            except Exception as error:  # pylint: disable=broad-except
                print(f'[L2-FACTORS-ERROR] {type(error).__name__}: {error}', flush=True)
        return original(model, factors)

    precision_probe.configure_precision = configure_precision


def _configured_factors() -> list:
    """Read the factor list straight from the configuration file as a fallback."""
    import yaml

    return list(yaml.safe_load(Path(sys.argv[1]).read_text())['model'].get('factors', []))


def _metrics_callback(trainer, pending: dict) -> Any:
    """Save full precision loss and reject a nonfinite training result."""
    from hyper_parallel.trainer.callbacks.base import Callback

    class Metrics(Callback):
        """Record the values the run is judged on."""

        def on_step_end(self, state, *, loss, loss_dict, grad_norm, **kwargs):
            """Observe the completed step without altering model or optimizer values."""
            del kwargs
            values = {'step': state.global_step, 'loss': float(loss), 'grad_norm': float(grad_norm),
                      'loss_dict': {name: float(value) for name, value in loss_dict.items()}}
            if 'norm_fp64' in pending:
                values['grad_norm_fp64'] = pending['norm_fp64']
                values['grad_norm_fp64_17g'] = '%.17g' % pending['norm_fp64']
            if not all(math.isfinite(value) for value in [values['loss'], values['grad_norm'],
                                                         *values['loss_dict'].values()]):
                raise ValueError(f'Nonfinite result at step {state.global_step}')
            rank = os.environ.get('RANK', '0')
            path = Path(os.environ['JT_RUN_DIR']) / f'metrics-rank{rank}.jsonl'
            with path.open('a') as stream:
                stream.write(json.dumps(values) + '\n')

    return Metrics(trainer)


def _report_embedding(model) -> None:
    """Report the embedding unit's class, parameter dtype and precision attributes."""
    embedding = dict(model.named_modules()).get('model.embed_tokens')
    if embedding is None:
        print('[L2-EMBEDDING] model.embed_tokens is missing', flush=True)
        return
    parameter = next(embedding.parameters(), None)
    attributes = {name: str(value) for name, value in vars(embedding).items()
                  if any(token in name for token in ('dtype', 'fp32', 'unit', 'embedding'))}
    print(f'[L2-EMBEDDING] class={type(embedding).__name__} '
          f'param_dtype={getattr(parameter, "dtype", None)} '
          f'keep_compute_in_fp32={getattr(embedding, "keep_compute_in_fp32", "UNSET")} '
          f'output_dtype={getattr(embedding, "output_dtype", "UNSET")} '
          f'attributes={json.dumps(attributes, sort_keys=True)}', flush=True)


def main() -> None:
    """Report the factor inventory and start the ordinary HP training loop."""
    _configure_import_path()
    _require_worktree_runtime()
    _install_factor_report()

    import alignment_builder
    from hyper_parallel.models.jt_deepseek_v3.adapter.runtime import mf_muon_parity
    from hyper_parallel.trainer.config.parser import parse_training_args
    from hyper_parallel.trainer.text_trainer import TextTrainer

    print(f'[L2] alignment_builder={Path(alignment_builder.__file__).resolve()}', flush=True)
    print(f'[L2] mf_muon_parity={Path(mf_muon_parity.__file__).resolve()}', flush=True)
    trainer = TextTrainer(parse_training_args())
    factors = list(getattr(trainer.base.config.model, 'factors', None) or _configured_factors())
    print(f'[L2] factors={json.dumps(factors)}', flush=True)
    if 'norm-dvm' in factors:
        from mf_dvm_clip import attach

        attach(trainer)
        print('[L2] attached the measured single-card MF DVM norm clipping', flush=True)
    import gradient_hashes

    pending = gradient_hashes.attach(trainer, Path(os.environ['JT_RUN_DIR']))
    print('[L2] attached the raw-gradient and post-step parameter fingerprints', flush=True)
    import activation_anchors

    activation_anchors.attach(trainer, Path(os.environ['JT_RUN_DIR']))
    import detail_anchors

    detail_anchors.attach(trainer, Path(os.environ['JT_RUN_DIR']))
    gemm_directory = os.environ.get('GEMM_VARIANT_DIR')
    if gemm_directory:
        import gemm_variants

        gemm_variants.attach(trainer, gemm_directory,
                             os.environ.get('MF_BITS_DIR', '/home/zth/tmp/align-cmp/mf'))
        print('[L2] attached the kv GEMM variant experiment', flush=True)
    _report_embedding(trainer.base.model)
    trainer.base._callbacks.insert(0, _metrics_callback(trainer.base, pending))
    trainer.train()


if __name__ == '__main__':
    main()
