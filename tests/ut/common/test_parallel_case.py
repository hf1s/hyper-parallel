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
"""Device allocation regressions for the framework-free ST launcher."""
import os
import subprocess
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from tests.common import distributed_launcher, parallel_case
from tests.common.parallel_case import TorchCase


class TestParallelCase(unittest.TestCase):
    """Exercise the public launcher without starting device processes."""

    def setUp(self) -> None:
        """Isolate device settings and replace only the process boundary."""
        self._stack = ExitStack()
        self.addCleanup(self._stack.close)
        self._stack.enter_context(patch.dict(os.environ))
        os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
        os.environ["HYPER_PARALLEL_TEST_DEVICE_TYPE"] = "npu"
        self.context = self._stack.enter_context(patch.object(parallel_case, "_parallel_run_context")).return_value
        self.context.Process.return_value.is_alive.return_value = False
        self.context.Process.return_value.exitcode = 0
        self.allocate_port = self._stack.enter_context(patch.object(parallel_case, "allocate_port", return_value=24567))

    def _assigned_devices(self) -> list[list[int]]:
        return [call.kwargs["args"][0] for call in self.context.Process.call_args_list]

    def test_inherited_devices_are_split_without_renumbering(self) -> None:
        """Two groups keep the parent's sparse device IDs and do not overlap."""
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "1,3,5,7"
        cases = [TorchCase("worker.py", "first", num_proc=2), TorchCase("worker.py", "second", num_proc=2)]
        parallel_case.parallel_run(cases)
        self.assertEqual(self._assigned_devices(), [[1, 3], [5, 7]])
        self.assertEqual(os.environ["ASCEND_RT_VISIBLE_DEVICES"], "1,3,5,7")
        self.assertEqual(self.allocate_port.call_count, 2)

    def test_single_group_inherits_selected_four_devices(self) -> None:
        """A four-rank case fits a four-device selection with the default budget."""
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "4,5,6,7"
        parallel_case.parallel_run([TorchCase("worker.py", "case", 23456, 4)])
        self.assertEqual(self._assigned_devices(), [[4, 5, 6, 7]])
        self.allocate_port.assert_not_called()

    def test_unset_visibility_keeps_zero_based_allocation(self) -> None:
        """Existing callers without a visible-device restriction keep their IDs."""
        parallel_case.parallel_run([TorchCase("worker.py", "first", num_proc=1),
                                    TorchCase("worker.py", "second", num_proc=2)])
        self.assertEqual(self._assigned_devices(), [[0], [1, 2]])

    def test_insufficient_visible_devices_fail_before_side_effects(self) -> None:
        """A budget larger than the inherited selection cannot widen that selection."""
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "4,6"
        with self.assertRaisesRegex(ValueError, "Requested 3 processes, but only 2 devices"):
            parallel_case.parallel_run([TorchCase("worker.py", "case", num_proc=3)])
        self.context.Process.assert_not_called()
        self.allocate_port.assert_not_called()

    def test_explicit_budget_still_limits_a_larger_selection(self) -> None:
        """The inherited list does not override a smaller caller-specified budget."""
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "4,5,6,7"
        with self.assertRaisesRegex(ValueError, "global_num_proc=2"):
            parallel_case.parallel_run([TorchCase("worker.py", "case", num_proc=3)], global_num_proc=2)
        self.context.Process.assert_not_called()
        self.allocate_port.assert_not_called()

    def test_malformed_visibility_fails_before_side_effects(self) -> None:
        """Reject ambiguous selections instead of silently enabling other devices."""
        for visible in ("", "4,", "4,,6", "-1", "4, 6", "a", "4,4", "4,04", "6,4", "４"):
            with self.subTest(visible=visible):
                os.environ["ASCEND_RT_VISIBLE_DEVICES"] = visible
                with self.assertRaisesRegex(ValueError, "ASCEND_RT_VISIBLE_DEVICES"):
                    parallel_case.parallel_run([TorchCase("worker.py", "case")])
        self.context.Process.assert_not_called()
        self.allocate_port.assert_not_called()

    def test_invalid_process_counts_fail_before_side_effects(self) -> None:
        """Non-positive and non-integer sizes cannot produce overlapping slices."""
        for count in (0, -1, True, 1.5):
            with self.subTest(num_proc=count):
                with self.assertRaisesRegex(ValueError, "num_proc.*positive integer"):
                    parallel_case.parallel_run([TorchCase("worker.py", "case", num_proc=count)])
            with self.subTest(global_num_proc=count):
                with self.assertRaisesRegex(ValueError, "global_num_proc.*positive integer"):
                    parallel_case.parallel_run([TorchCase("worker.py", "case")], global_num_proc=count)
        self.context.Process.assert_not_called()
        self.allocate_port.assert_not_called()

    def test_cpu_allocation_ignores_npu_visibility(self) -> None:
        """CPU process groups are not limited by inherited NPU device IDs."""
        os.environ["HYPER_PARALLEL_TEST_DEVICE_TYPE"] = "cpu"
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "invalid-for-npu"
        parallel_case.parallel_run([TorchCase("worker.py", "case", num_proc=3)])
        self.assertEqual(self._assigned_devices(), [[0, 1, 2]])

    def test_worker_receives_assigned_ids_and_cpu_clears_visibility(self) -> None:
        """The wrapper forwards the selected IDs without loading a device runtime."""
        observed = []
        launcher = Mock(side_effect=lambda *_: observed.append(os.environ.get("ASCEND_RT_VISIBLE_DEVICES")))
        case = TorchCase("worker.py", "case", 23456, 2)
        with patch.object(parallel_case.os, "setsid"), patch.object(distributed_launcher, "torchrun_case", launcher):
            parallel_case.run_case([4, 6], case)
            os.environ["HYPER_PARALLEL_TEST_DEVICE_TYPE"] = "cpu"
            parallel_case.run_case([0, 1], case)
        self.assertEqual(observed, ["4,6", None])
        self.assertEqual(launcher.call_count, 2)

    def test_launcher_import_does_not_load_training_frameworks(self) -> None:
        """Check the import contract in a fresh interpreter without pytest plugins."""
        script = (
            "import sys; import tests.common.parallel_case; "
            "print(','.join(name for name in ('torch', 'torch_npu', 'hyper_parallel') if name in sys.modules))"
        )
        result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[3],
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "")
