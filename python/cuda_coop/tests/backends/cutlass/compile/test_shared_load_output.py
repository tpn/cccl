# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reject shared Load destinations that cannot supply disjoint typed rows."""

import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr
from cutlass.memory import SmemAllocator

from cuda import coop
from cuda.coop import cutlass as cutlass_coop

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


@pytest.mark.parametrize(
    "api", [coop, cutlass_coop], ids=["common", "qualified"]
)
@pytest.mark.parametrize(
    "case,message",
    [
        ("global", "shared tensor"),
        ("register", "shared tensor"),
        ("dtype", "dtype"),
        ("capacity", "provides 127"),
        ("rank", "one-dimensional"),
        ("stride", "contiguous"),
        ("alignment", "alignment"),
        ("zero", "positive"),
        ("dynamic", "integer"),
        ("payload", "output or items_per_thread"),
    ],
)
def test_invalid_shared_output(api, case, message):
    @cute.kernel
    def kernel(source: cute.Pointer):
        count = 4
        capacity = 127 if case == "capacity" else 256
        scalar_type = cutlass.Float32 if case == "dtype" else cutlass.Int32
        shared = SmemAllocator().allocate_tensor(
            scalar_type, cute.make_layout(capacity), byte_alignment=4
        )
        output = shared
        if cutlass.const_expr(case == "global"):
            output = cute.make_tensor(source, cute.make_layout(128))
        elif cutlass.const_expr(case == "register"):
            output = cute.make_rmem_tensor(128, cutlass.Int32)
        elif cutlass.const_expr(case == "rank"):
            output = cute.make_tensor(
                shared.iterator, cute.make_layout((32, 4))
            )
        elif cutlass.const_expr(case == "stride"):
            output = cute.make_tensor(
                shared.iterator, cute.make_layout(128, stride=2)
            )
        elif cutlass.const_expr(case == "payload"):
            output = api.ThreadData(items_per_thread=4)
        elif cutlass.const_expr(case == "zero"):
            count = 0
        elif cutlass.const_expr(case == "dynamic"):
            count = cute.arch.thread_idx()[0] + 1
        api.load(
            api.this_block(),
            source,
            output,
            items_per_thread=count,
            alignment=16 if case == "alignment" else None,
        )

    @cute.jit
    def launch(source: cute.Pointer):
        kernel(source).launch(grid=1, block=32)

    pointer = make_ptr(
        cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=4
    )
    with pytest.raises(Exception, match=message):
        cute.compile[(GPUArch("sm_80"),)](launch, pointer)
