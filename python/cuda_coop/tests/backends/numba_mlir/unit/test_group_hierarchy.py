# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check physical group descriptors against exact launch facts."""

import pytest

import cuda.coop.numba_mlir as coop
from cuda.coop._core import LaunchFactOrigin, LaunchFacts, resolve_thread_group

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


_GROUP_KINDS = (
    ("this_thread", "thread"),
    ("this_warp", "warp"),
    ("this_block", "block"),
    ("this_cluster", "cluster"),
    ("this_grid", "grid"),
)


@pytest.mark.parametrize(("constructor_name", "kind"), _GROUP_KINDS)
def test_core_resolves_every_physical_group_from_exact_launch_facts(
    constructor_name,
    kind,
):
    launch = LaunchFacts(
        exact_block_dim=(8, 4, 2),
        exact_grid_dim=(6, 4, 2),
        exact_cluster_dim=(2, 2, 1),
        cluster_launch=True,
        provenance=LaunchFactOrigin(
            fact="cluster_launch",
            source="test",
            verified=True,
        ),
    )
    group = getattr(coop, constructor_name)()
    resolved = resolve_thread_group(group, launch).require_supported()

    assert resolved.kind == kind
    if kind == "thread":
        assert resolved.hierarchy.implicit
        return
    assert resolved.hierarchy.block_dim == (8, 4, 2)
    if kind in {"cluster", "grid"}:
        assert resolved.hierarchy.cluster_dim == (2, 2, 1)
    if kind == "grid":
        assert resolved.hierarchy.grid_dim == (3, 2, 2)
