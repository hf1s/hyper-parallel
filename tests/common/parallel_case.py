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
"""Wrappers that spawn ``torchrun`` ST workers.

Like :mod:`tests.common.distributed_launcher`, this module must stay free of
``torch`` / ``hyper_parallel`` imports so pytest parent launchers only pay the
cost of forking the distributed runner. Worker bodies live in non-launcher
modules and import frameworks as usual.
"""
import os
import signal
import multiprocessing as mp
from typing import Optional

from tests.common.port_utils import allocate_port


class TorchCase:
    """torch case messages"""

    def __init__(self, file_name: str, case_name: str, master_port: Optional[int] = None, num_proc: int = 1) -> None:
        """Initialize TorchCase with file path, case name, optional port, and process count."""
        self.file_name = file_name
        self.case_name = case_name
        self.master_port = master_port
        self.num_proc = num_proc


def _parallel_run_context():
    """Use spawn for CPU shards to avoid inheriting fork-only state."""
    device_type = os.environ.get("HYPER_PARALLEL_TEST_DEVICE_TYPE", "").strip().lower()
    if device_type == "cpu":
        return mp.get_context("spawn")
    return mp.get_context()


def run_case(visible_devices: list, case: TorchCase) -> None:
    """Run a single test case in a child process with device visibility set.

    Args:
        visible_devices: List of device indices to expose via ASCEND_RT_VISIBLE_DEVICES.
        case: The test case descriptor.
    """
    # become the leader of a new process group so that os.killpg on timeout
    # kills torchrun worker sub-processes as well as this wrapper
    os.setsid()
    if os.environ.get("HYPER_PARALLEL_TEST_DEVICE_TYPE", "").strip().lower() == "cpu":
        os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    else:
        # set visible devices for current case
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = ','.join(map(str, visible_devices))
    # Import the thin launcher only — never tests.torch.utils (imports torch /
    # torch_npu) in this wrapper process.
    # pylint: disable=C0415
    from tests.common.distributed_launcher import torchrun_case
    torchrun_case(case.file_name, case.case_name, case.master_port, case.num_proc)


def _auto_assign_ports(cases: list) -> None:
    """Assign unique ports to every case whose :attr:`master_port` is ``None``.

    Ports are allocated in the parent process so that concurrent children
    never race — each child receives a globally-unique, pre-allocated port.
    """
    for case in cases:
        if case.master_port is None:
            case.master_port = allocate_port()


def _visible_device_pool(global_num_proc: int) -> list[int]:
    """Read the inherited NPU selection without importing a device runtime."""
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    device_type = os.environ.get("HYPER_PARALLEL_TEST_DEVICE_TYPE", "").strip().lower()
    if device_type == "cpu" or visible is None:
        return list(range(global_num_proc))
    fields = visible.split(",")
    if any(not field.isascii() or not field.isdecimal() for field in fields):
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES must contain comma-separated non-negative device IDs")
    devices = [int(field) for field in fields]
    if devices != sorted(set(devices)):
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES must contain unique device IDs in ascending order")
    return devices[:global_num_proc]


def _device_assignments(cases: list[TorchCase], global_num_proc: int) -> list[list[int]]:
    """Validate the budget before slicing the parent's visible device IDs."""
    if isinstance(global_num_proc, bool) or not isinstance(global_num_proc, int) or global_num_proc < 1:
        raise ValueError("global_num_proc must be a positive integer")
    for case in cases:
        if isinstance(case.num_proc, bool) or not isinstance(case.num_proc, int) or case.num_proc < 1:
            raise ValueError(f"num_proc for {case.case_name!r} must be a positive integer")
    devices = _visible_device_pool(global_num_proc)
    required = sum(case.num_proc for case in cases)
    if required > len(devices):
        raise ValueError(
            f"Requested {required} processes, but only {len(devices)} devices are available "
            f"within global_num_proc={global_num_proc}"
        )
    assignments = []
    offset = 0
    for case in cases:
        assignments.append(devices[offset:offset + case.num_proc])
        offset += case.num_proc
    return assignments


def parallel_run(cases: list[TorchCase], global_num_proc: int = 8) -> None:
    """Run a group of test cases in parallel, assigning disjoint device slices to each.

    Args:
        cases: List of TorchCase descriptors to run concurrently.
            The sum of all ``num_proc`` values must not exceed ``global_num_proc``.
        global_num_proc: Maximum device budget for this group. Defaults to 8.
            When ASCEND_RT_VISIBLE_DEVICES is set, NPU cases receive disjoint
            slices of those IDs, capped by this budget. Unset visibility and CPU
            cases retain the zero-based allocation.

    Raises:
        ValueError: If process counts are not positive integers, the inherited
            device selection is malformed, or the cases exceed the available budget.
        AssertionError: If any case times out (420 s join timeout), or if any
            child process exits with a non-zero return code.
    """
    assignments = _device_assignments(cases, global_num_proc)

    # auto-assign ports before spawning children (avoids cross-process races)
    unassigned = [c for c in cases if c.master_port is None]
    if unassigned:
        _auto_assign_ports(unassigned)

    # create child process (run_case calls os.setsid to own a process group,
    # so os.killpg on timeout kills torchrun workers too)
    processes = []
    ctx = _parallel_run_context()
    for _, (case, devices) in enumerate(zip(cases, assignments)):
        p = ctx.Process(target=run_case, args=(devices, case))
        p.start()
        processes.append(p)

    # wait child process terminates (timeout=420s to prevent infinite hang on distributed deadlock)
    timed_out = []
    for i, p in enumerate(processes):
        p.join(timeout=420)
        if p.is_alive():
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.join()
            timed_out.append(cases[i].case_name)

    # check results for all cases
    if timed_out:
        raise AssertionError(f"Cases timed out (possible collective deadlock): {timed_out}")
    failed = [cases[i].case_name for i, p in enumerate(processes) if p.exitcode != 0]
    assert not failed, f"List cases failed: {failed}"
