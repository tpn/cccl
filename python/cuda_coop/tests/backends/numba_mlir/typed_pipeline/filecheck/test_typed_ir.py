# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""What inference and call resolution leave in the typed Numba IR.

The dump is taken after the last rewrite, so it shows the ``ThreadData``
types with their evidence and the ``cuda.shared.array`` backing the storage
pass inserted.
"""

from __future__ import annotations

from numba_cuda_mlir import cuda

from cuda import coop
from cuda.coop.numba_mlir._pipeline import testing

from .._support import BLOCK, CHIP, f32_in_out, requires_typed_pipeline

pytestmark = requires_typed_pipeline


def test_thread_data_takes_its_dtype_from_the_load():
    @cuda.jit
    def k1(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items)
        coop.store(block, output, items)

    testing.check(
        k1,
        f32_in_out(),
        """
        CHECK-LABEL: typed-ir k1
        CHECK:         block = call {{.*}} :: coop.group[block]
        CHECK:         items = call
        CHECK-SAME:      :: coop.ThreadData(float32, items=4, producer)
        CHECK-NOT:     $coop_backing
        """,
        dump="typed_ir",
        block=BLOCK,
        chip=CHIP,
    )


def test_explicit_dtype_is_the_strongest_evidence():
    @cuda.jit
    def explicit(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4, dtype=output.dtype)
        coop.load(block, source, items)
        coop.store(block, output, items)

    testing.check(
        explicit,
        f32_in_out(),
        """
        CHECK-LABEL: typed-ir explicit
        CHECK:         items = call
        CHECK-SAME:      :: coop.ThreadData(float32, items=4, explicit)
        """,
        dump="typed_ir",
        block=BLOCK,
        chip=CHIP,
    )


def test_store_destination_types_a_payload_without_a_load():
    @cuda.jit
    def fill(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        for i in range(4):
            items[i] = 1
        coop.store(block, output, items)

    testing.check(
        fill,
        f32_in_out(),
        """
        CHECK-LABEL: typed-ir fill
        CHECK:         items = call
        CHECK-SAME:      :: coop.ThreadData(float32, items=4, store-dst)
        """,
        dump="typed_ir",
        block=BLOCK,
        chip=CHIP,
    )


def test_a_load_through_a_tuple_alias_types_the_payload():
    @cuda.jit
    def packed(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        alias = (items,)[0]
        coop.load(block, source, alias)
        coop.store(block, output, items)

    testing.check(
        packed,
        f32_in_out(),
        """
        CHECK-LABEL: typed-ir packed
        CHECK:         items = call
        CHECK-SAME:      :: coop.ThreadData(float32, items=4, producer)
        CHECK:         = build_tuple(items=[Var(items,
        CHECK-SAME:      :: UniTuple(coop.ThreadData(float32, items=4,
        CHECK-SAME:      producer) x 1)
        CHECK:         alias = static_getitem(
        CHECK-SAME:      :: coop.ThreadData(float32, items=4, producer)
        CHECK:         _coop_store_wrapper_{{.*}}_value = items
        CHECK-SAME:      :: coop.ThreadData(float32, items=4, producer)
        """,
        dump="typed_ir",
        block=BLOCK,
        chip=CHIP,
    )


def test_transposing_algorithm_gets_a_static_shared_backing():
    @cuda.jit
    def kt(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items, algorithm="transpose")
        coop.store(block, output, items, algorithm="transpose")

    testing.check(
        kt,
        f32_in_out(),
        """
        CHECK-LABEL: typed-ir kt
        CHECK:         $coop_shared_array = global(shared_array:
        CHECK-NEXT:    $coop_backing_size = const(int, [[SIZE:[0-9]+]])
        CHECK-NEXT:    $coop_backing_dtype = global(uint8: uint8)
        CHECK-NEXT:    $coop_backing_align = const(int, 16)
        CHECK-NEXT:    $coop_backing = call $coop_shared_array(
        CHECK-SAME:      kws=(('alignment', {{.*}} :: array(uint8, 1d, C)
        CHECK:         $coop_region_hi = const(int, [[SIZE]])
        CHECK:         $coop_region_implicit = getitem(value=$coop_backing,
        CHECK-SAME:      :: array(uint8, 1d, C)
        CHECK:         ('__coop_scratch', Var($coop_region_implicit
        CHECK:         ('__coop_scratch', Var($coop_region_implicit
        """,
        dump="typed_ir",
        block=BLOCK,
        chip=CHIP,
    )
