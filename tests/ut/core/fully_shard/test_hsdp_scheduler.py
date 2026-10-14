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
"""CPU regressions for source-parameter ownership during FSDP initialization."""

import gc
import unittest
import weakref
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, clear_device_mesh_cache
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2
from hyper_parallel.core.fully_shard.hsdp_state import HSDPStateV2
from hyper_parallel.core.fully_shard.utils import (
    DDPMeshInfo,
    FSDPMeshInfo,
    MixedPrecisionPolicy,
    OffloadPolicy,
    SourceShardMetaInfo,
)


class TestSourceParameterLifetime(unittest.TestCase):
    """Keep source-layout metadata while releasing replaced parameter storage."""

    def tearDown(self) -> None:
        """Isolate the topology-only meshes created by this test."""
        clear_device_mesh_cache()

    @patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=0)
    def test_source_parameters_are_released_after_sharding(self, _mock_rank):
        """Single and grouped roots must not keep pre-sharding parameters alive."""
        for grouped in (False, True):
            for mode in ("source", "replicate", "both"):
                with self.subTest(grouped=grouped, mode=mode):
                    self._check_source_lifetime(grouped, mode)

    @staticmethod
    def _mesh_info(mesh, replicate):
        """Provide topology metadata without creating a process group."""
        mesh_info = object.__new__(DDPMeshInfo if replicate else FSDPMeshInfo)
        mesh_info.mesh = mesh
        mesh_info.shard_mesh_dim = None if replicate else 0
        mesh_info.replicate_mesh_dim = 0 if replicate else None
        if replicate:
            mesh_info.replicate_mesh_rank = 0
            mesh_info.replicate_mesh_size = 2
            mesh_info.replicate_process_group = None
        else:
            mesh_info.shard_mesh_rank = 0
            mesh_info.shard_mesh_size = 2
            mesh_info.shard_process_group = None
        return mesh_info

    def _check_source_lifetime(self, grouped, mode):
        root = DeviceMesh("cpu", [[0], [1]], mesh_dim_names=("fsdp", "tp"), _init_backend=False)
        fsdp, source = root["fsdp"], root["tp"]
        replicate = mode != "source"
        mesh_info = self._mesh_info(fsdp, replicate)
        module = nn.Linear(4, 4, bias=False)
        original = weakref.ref(module.weight)
        expected = module.weight.detach().clone()[:4 if replicate else 2]
        metadata = SourceShardMetaInfo(source, (Replicate(),)) if mode != "replicate" else None
        source_map = {module.weight: metadata} if metadata is not None else None
        replicate_params = {module.weight} if replicate else None
        roots = module
        if grouped:
            peer = nn.Linear(4, 4, bias=False)
            peer.weight = module.weight
            roots = (module, peer)

        with patch.object(HSDPStateV2, "_build_param_mesh_info", return_value=mesh_info):
            scheduler = HSDPSchedulerV2(
                roots, fsdp, True, None, MixedPrecisionPolicy(), OffloadPolicy(),
                None, replicate_params, torch.device("cpu"), False, source_shard_infos=source_map,
            )

        if source_map is not None:
            self.assertEqual(len(source_map), 1)
            self.assertIs(source_map[original()], metadata)
        if replicate_params is not None:
            self.assertEqual(replicate_params, {original()})
        self.assertIs(scheduler.hsdp_state.hsdp_params[0].source_shard_info, metadata)
        self.assertNotEqual(module.weight.to_local().data_ptr(), original().data_ptr())
        del source_map, replicate_params
        gc.collect()

        self.assertIsNone(original(), "FSDP retained the full source Parameter after replacing it")
        torch.testing.assert_close(module.weight.to_local(), expected)
        if grouped:
            self.assertIs(roots[0].weight, roots[1].weight)
