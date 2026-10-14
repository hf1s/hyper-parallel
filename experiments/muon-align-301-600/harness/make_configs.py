#!/usr/bin/env python3
"""Materialise the four alignment configs used by the Muon switch A/B runs.

``cfg-base.yaml`` is the alignment config with all three switches written out explicitly and
turned off; ``cfg-a.yaml`` / ``cfg-b.yaml`` / ``cfg-c.yaml`` turn on exactly one of them, so the
four files differ by single boolean lines.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'train-301.yaml'

MUON_TAIL = """    matched_adamw_rms: 0.2
"""
MUON_TAIL_WITH_SWITCHES = """    matched_adamw_rms: 0.2
    fp32_update: %s
    ns_per_param: %s
"""
ADAMW_TAIL = """    adamw_eps: 1.0e-08
"""
ADAMW_TAIL_WITH_SWITCH = """    adamw_eps: 1.0e-08
    adamw_wd_all_params: %s
"""

VARIANTS = {
    'cfg-base.yaml': (False, False, False),
    'cfg-a.yaml': (True, False, False),
    'cfg-b.yaml': (False, False, True),
    'cfg-c.yaml': (False, True, False),
}


def render(fp32_update: bool, ns_per_param: bool, wd_all: bool) -> str:
    """Return the alignment config with the three switch values written in."""
    def flag(value: bool) -> str:
        """Return the YAML spelling of a boolean flag."""
        return 'true' if value else 'false'

    text = SOURCE.read_text()
    for old, new in (
        (MUON_TAIL, MUON_TAIL_WITH_SWITCHES % (flag(fp32_update), flag(ns_per_param))),
        (ADAMW_TAIL, ADAMW_TAIL_WITH_SWITCH % flag(wd_all)),
    ):
        if text.count(old) != 1:
            raise SystemExit(f'anchor found {text.count(old)} times: {old!r}')
        text = text.replace(old, new)
    return text


def main() -> None:
    """Write the four config files next to this script."""
    for name, values in VARIANTS.items():
        path = HERE / name
        path.write_text(render(*values))
        print(f'{path}: fp32_update={values[0]} ns_per_param={values[1]} wd_all={values[2]}')


if __name__ == '__main__':
    main()
