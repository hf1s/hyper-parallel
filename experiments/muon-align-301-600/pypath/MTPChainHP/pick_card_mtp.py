"""Print the index of an idle NPU with at least ``need`` MB of free HBM.

``npu-smi info``'s HBM column can lag behind reality: a chip can report tens of GB free while a
process that never released its device memory still holds them, and a launch then fails with
"NPU out of memory".  So the chip is only accepted when ``npu-smi info -t proc-mem`` also reports
no process on it.

Usage: ``pick_card.py [needed_MB] [wait_seconds]``
"""

import re
import subprocess
import sys
import time

# ``npu-smi info`` prints two lines per device: ``| <npu>  <name> | health | ...`` followed by
# ``| <chip> | <bus-id> | <aicore> | <mem> | <hbm-used> / <hbm-total> |``.  The chip column is 0
# for every device, so the NPU index has to come from the first line.
NPU = re.compile(r'^\|\s*(\d+)\s+\S+\s*\|')
HBM = re.compile(r'^\|\s*\d+\s*\|\s*[0-9A-Fa-f:.]+\s*\|\s*\d+\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)')


def _run(args):
    return subprocess.run(args, capture_output=True, text=True, timeout=60).stdout


def hbm_free():
    """Return ``{npu: free_MB}`` from ``npu-smi info``."""
    result = {}
    current = None
    for line in _run(['npu-smi', 'info']).splitlines():
        match = NPU.match(line)
        if match:
            current = int(match.group(1))
            continue
        match = HBM.match(line)
        if match and current is not None:
            result[current] = int(match.group(2)) - int(match.group(1))
    return result


def busy(chip):
    """Return True when a process still holds device memory on ``chip``."""
    return 'No process in device' not in _run(['npu-smi', 'info', '-t', 'proc-mem', '-i', str(chip)])


def main():
    """Print the best idle chip, waiting until one has the requested free memory."""
    need = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    wait = float(sys.argv[2]) if len(sys.argv) > 2 else 3600.0
    deadline = time.time() + wait
    while True:
        free = hbm_free()
        candidates = sorted((chip for chip, size in free.items() if size >= need),
                            key=lambda chip: -free[chip])
        for chip in candidates:
            if not busy(chip):
                print(f'[pick] chip={chip} free={free[chip]}MB need={need}MB '
                      f'(idle check passed; all={sorted(free.items())})', file=sys.stderr, flush=True)
                print(chip)
                return 0
        print(f'[pick] no idle chip with {need}MB free; free={sorted(free.items())} '
              f'busy={[chip for chip in candidates]}', file=sys.stderr, flush=True)
        if time.time() >= deadline:
            raise SystemExit(f'no idle chip has {need}MB free after {wait}s')
        time.sleep(60)


if __name__ == '__main__':
    sys.exit(main())
