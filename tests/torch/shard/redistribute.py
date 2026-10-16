# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""test torch redistribute"""
import numpy as np
import torch
import torch.distributed as dist
import pytest
from hyper_parallel import DTensor, distribute_tensor, init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Partial, Shard, Replicate
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device


def test_shard_to_replicate():
    '''
    Feature: shard to replicate.
    Description:
    Expectation: Run success.
    '''
    init_backend(_DEVICE_TYPE)

    # Create DeviceMesh
    mesh = init_device_mesh(
        device_type=_DEVICE_TYPE,
        mesh_shape=(1, 2),
        mesh_dim_names=("dp", "tp")
    )

    # Define placements using Placement format
    x_placements = (Replicate(), Shard(1))
    dst_placements = (Replicate(), Replicate())

    dist_x = DTensor.from_local(to_device(torch.ones(8, 4), _DEVICE_TYPE), mesh, x_placements)
    out = dist_x.redistribute(mesh, dst_placements)

    expect_out = torch.ones(8, 8)

    assert np.allclose(expect_out.cpu().detach().numpy(),
                       out.to_local().cpu().detach().numpy(),  # use to_local()
                       0.001, 0.001)


def test_replicate_to_shard():
    '''
    Feature: replicate to shard.
    Description:
    Expectation: Run success.
    '''
    init_backend(_DEVICE_TYPE)

    # Create DeviceMesh
    mesh = init_device_mesh(
        device_type=_DEVICE_TYPE,
        mesh_shape=(1, 2),
        mesh_dim_names=("dp", "tp")
    )

    # Define placements using Placement format
    dst_placements = (Replicate(), Shard(1))
    x_placements = (Replicate(), Replicate())

    dist_x = DTensor.from_local(to_device(torch.ones(8, 8), _DEVICE_TYPE), mesh, x_placements)
    out = dist_x.redistribute(mesh, dst_placements)

    expect_out = torch.ones(8, 4)

    assert np.allclose(expect_out.cpu().detach().numpy(),
                       out.to_local().cpu().detach().numpy(),  # use to_local()
                       0.001, 0.001)


def test_different_mesh():
    '''
    Feature: different mesh, unsupported.
    Description:
    Expectation: Run success.
    '''
    init_backend(_DEVICE_TYPE)

    # Create DeviceMesh with 3 dimensions
    mesh_3d = init_device_mesh(
        device_type=_DEVICE_TYPE,
        mesh_shape=(1, 2, 1),
        mesh_dim_names=("dp", "tp", "sp")
    )

    # Note: For 3D mesh, "tp" maps to dim 1
    dst_placements = (Replicate(), Shard(1), Replicate())

    # Create DeviceMesh with 2 dimensions
    mesh_2d = init_device_mesh(
        device_type=_DEVICE_TYPE,
        mesh_shape=(1, 2),
        mesh_dim_names=("dp", "tp")
    )

    x_placements = (Replicate(), Replicate())

    dist_x = DTensor.from_local(to_device(torch.ones(8, 8), _DEVICE_TYPE), mesh_2d, x_placements)

    # Redistributing between different meshes should raise RuntimeError
    with pytest.raises(RuntimeError):
        dist_x.redistribute(mesh_3d, dst_placements)


def test_non_contiguous_redistribute():
    '''
    Feature: redistribute a DTensor whose local tensor is non-contiguous.

    A transpose makes the local storage non-contiguous.  The redistribution
    path must internally ensure contiguity before calling communication ops.
    Description:
    Expectation: Run success — the redistributed result matches the expected full tensor.
    '''
    init_backend(_DEVICE_TYPE)

    mesh = init_device_mesh(
        device_type=_DEVICE_TYPE,
        mesh_shape=(1, 2),
        mesh_dim_names=("dp", "tp")
    )

    src_placements = (Replicate(), Shard(1))
    dst_placements = (Replicate(), Replicate())

    # Non-contiguous local shard via transpose.
    data = torch.ones(2, 2)
    non_contig_local = data.T  # shape (2, 2), non-contiguous
    assert not non_contig_local.is_contiguous()

    dist_x = DTensor.from_local(to_device(non_contig_local, _DEVICE_TYPE), mesh, src_placements)
    out = dist_x.redistribute(mesh, dst_placements)

    expect_out = torch.ones(2, 4)
    assert np.allclose(expect_out.cpu().detach().numpy(),
                       out.to_local().cpu().detach().numpy(),
                       0.001, 0.001)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_uneven_shard_reconstruction_rejected(dtype: torch.dtype) -> None:
    """Reject unsupported uneven shards on both ranks without poisoning cached transforms."""
    init_backend(_DEVICE_TYPE)
    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2,), mesh_dim_names=("shard",))

    # Interleave even and uneven shapes to exercise already cached transforms.
    for rows in (2, 1, 3, 4):
        reference = to_device(torch.arange(rows * 64, dtype=torch.float32).reshape(rows, 64).to(dtype), _DEVICE_TYPE)
        sharded = distribute_tensor(reference, mesh, (Shard(0),))
        if rows % 2:
            replicated = distribute_tensor(reference, mesh, (Replicate(),))
            expected = (
                "DTensor redistribution requires equal-sized shards: "
                f"global shape ({rows}, 64), dimension 0 with size {rows} "
                "cannot be evenly split across 2 shards."
            )
            for _ in range(2):
                with pytest.raises(NotImplementedError) as error:
                    sharded.full_tensor()
                assert str(error.value) == expected, f"Expected {expected!r}, got {str(error.value)!r}"
                with pytest.raises(NotImplementedError) as error:
                    replicated.redistribute(mesh, (Shard(0),))
                assert str(error.value) == expected, f"Expected {expected!r}, got {str(error.value)!r}"
        else:
            for _ in range(2):
                torch.testing.assert_close(sharded.full_tensor(), reference, rtol=0, atol=0)
        dist.barrier()


def test_uneven_partial_reduction_rejected() -> None:
    """Explicit partial reduction must not turn uneven logical shapes into rank-local guesses."""
    init_backend(_DEVICE_TYPE)
    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("shard", "partial"))
    for rows in (2, 1, 3, 4):
        reference = to_device(torch.arange(rows * 64, dtype=torch.float32).reshape(rows, 64), _DEVICE_TYPE)
        sharded = distribute_tensor(reference, mesh, (Shard(0), Replicate()))
        partial = DTensor.from_local(sharded.to_local(), mesh, (Shard(0), Partial()), shape=reference.shape)
        if rows % 2:
            expected = (
                "DTensor redistribution requires equal-sized shards: "
                f"global shape ({rows}, 64), dimension 0 with size {rows} "
                "cannot be evenly split across 2 shards."
            )
            for _ in range(2):
                with pytest.raises(NotImplementedError) as error:
                    partial.reduce_partial()
                assert str(error.value) == expected, f"Expected {expected!r}, got {str(error.value)!r}"
        else:
            reduced = partial.reduce_partial()
            torch.testing.assert_close(reduced.full_tensor(), reference * 2, rtol=0, atol=0)
        dist.barrier()
