# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check asynchronous shared-copy lifetime and operand diagnostics."""

import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr

from cuda import coop

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


def _compile(launch):
    source = make_ptr(
        cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    return cute.compile[(GPUArch("sm_80"),)](launch, source, cutlass.Int32(1))


@pytest.mark.parametrize(
    "case,diagnostic",
    [
        ("missing", "requires a matching wait"),
        ("twice", "waited on exactly once"),
        ("branch_issue", r"kernel\s+entry\s+block"),
        (
            "branch_wait",
            r"kernel\s+entry\s+block|TYPE_DYNAMIC_EXPR_UNSUPPORTED",
        ),
        ("loop_issue", r"kernel\s+entry\s+block"),
        ("small", "insufficient padded capacity"),
        ("alignment", "16-byte alignment"),
        ("global_destination", "smem memory"),
    ],
)
def test_invalid_shared_copy(case, diagnostic):
    @cute.kernel
    def kernel(source: cute.Pointer, control: cutlass.Int32):
        block = coop.this_block()
        capacity = 8 if case == "small" else 12
        alignment = 4 if case == "alignment" else 16
        shared = coop.TempStorage().reserve(
            capacity, cutlass.Int32, alignment=alignment
        )
        if cutlass.const_expr(case == "global_destination"):
            shared = cute.make_tensor(source, cute.make_layout(12))
        if cutlass.const_expr(case == "branch_issue"):
            if control > 0:
                coop.load_to_shared_async(block, source, shared, num_items=8)
        elif cutlass.const_expr(case == "loop_issue"):
            for _ in range(control):
                coop.load_to_shared_async(block, source, shared, num_items=8)
        else:
            pending = coop.load_to_shared_async(
                block, source, shared, num_items=8
            )
            if cutlass.const_expr(case == "branch_wait"):
                if control > 0:
                    pending.wait()
            elif cutlass.const_expr(case != "missing"):
                alias = pending
                pending.wait()
                if cutlass.const_expr(case == "twice"):
                    alias.wait()

    @cute.jit
    def launch(source: cute.Pointer, control: cutlass.Int32):
        kernel(source, control).launch(grid=1, block=64)

    with pytest.raises(Exception, match=diagnostic):
        _compile(launch)
