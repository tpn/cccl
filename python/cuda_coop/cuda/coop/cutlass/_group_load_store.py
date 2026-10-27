# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validate block and warp Load/Store calls for CuTe lowering.

Exact launch facts and shared group planning determine participation and
algorithm support. Argument bindings mark each control as omitted, as a
constant embedded in generated C++, or as a runtime argument. Block calls can
use explicit scratch descriptors; physical and logical warps use
compiler-managed scratch when their algorithm needs it.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

from cuda.coop._core import ArgumentBinding, GroupLoadStoreAlgorithm
from cuda.coop._core.api._dispatch import (
    _common_root_operation_name,
    _common_root_operation_scope,
)
from cuda.coop._core.api._payload import _validate_common_temp_storage
from cuda.coop._core.thread_group import ThreadGroup as CommonThreadGroup
from cuda.coop._typing import (
    BlockLoadStoreAlgorithm,
    CommonThreadDataLike,
    IntegerValue,
    TempStorageLike,
    ThreadDataLike,
    ValidItems,
    WarpLoadStoreAlgorithm,
    WritableArrayLike,
    _CommonNumericT,
)

from .._core.api.thread_group import BlockGroup, WarpGroup
from ._thread_data import ThreadData
from ._thread_group import (
    _require_complete_warp_partition,
    _resolve_primitive_group_from_launch,
)

_SCOPE = "cuda.coop.cutlass"
_MAX_STATIC_OFFSET = (1 << 63) - 1


def _resolve_group(group, algorithm, temp_storage, operation):
    """Resolve block or warp groups against exact launch dimensions.

    Physical and logical warps require a block made only of complete 32-thread
    warps, and they reject explicit scratch descriptors. Normalize the
    algorithm and validate any explicit block scratch descriptor. Shared
    planning checks the logical width and algorithm support. Return the group,
    launch facts, and normalized algorithm for lowering.
    """

    if not isinstance(group, CommonThreadGroup):
        raise TypeError(f"{_SCOPE}.{operation} group must be a ThreadGroup")
    if group.kind not in {"block", "warp", "threads_within_warp"}:
        raise NotImplementedError(
            f"{_SCOPE}.{operation} requires a block, physical warp, "
            "or logical warp group"
        )
    if group.kind != "block" and temp_storage is not None:
        raise NotImplementedError(
            f"{_SCOPE}.{operation} explicit TempStorage is supported only "
            "for block groups"
        )
    algorithm = _normalize_algorithm(algorithm)
    if temp_storage is not None:
        _validate_common_temp_storage(operation, temp_storage)
        from ._compiler._lifetime import register_use
        from ._temp_storage import TempStorage

        if isinstance(temp_storage, TempStorage):
            register_use(temp_storage, operation)
    from ._compiler._launch import current_kernel_launch_facts

    launch = current_kernel_launch_facts()
    resolved = _resolve_primitive_group_from_launch(
        group, launch, feature=operation
    )
    _require_complete_warp_partition(
        resolved, feature=operation, exact_block_dim=launch.exact_block_dim
    )
    return resolved, launch, algorithm


def load(
    group: BlockGroup | WarpGroup,
    source: object,
    output: ThreadDataLike[_CommonNumericT] | WritableArrayLike | None = None,
    /,
    *,
    items_per_thread: int | None = None,
    alignment: int | None = None,
    algorithm: BlockLoadStoreAlgorithm | WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    oob_default: _CommonNumericT | float | None = None,
    offset: IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData | None:
    """Load a contiguous group tile into per-thread values.

    Shared parameters and participation follow :func:`cuda.coop.load`. This
    implementation accepts blocks and physical or logical warps. Supply a
    positive compile-time ``items_per_thread`` to return fresh CUTLASS
    ThreadData with the source dtype; this is the preferred form.
    Alternatively, supply a CUTLASS ThreadData ``output`` to fill it in place
    and return ``None``. Its dtype is inferred from the source, or must agree
    with it when already declared.
    Supply ``items_per_thread`` without ``output`` to return a payload, or
    supply ThreadData without ``items_per_thread`` to fill that payload.

    To load directly into shared memory, supply a native CuTe shared tensor
    as ``output`` together with ``items_per_thread``. The tensor must be
    contiguous, one-dimensional, have the source dtype, and contain at least
    ``block_threads * items_per_thread`` elements. Each physical block thread
    writes its own consecutive row, starting at
    ``linear_thread_rank * items_per_thread``. A native view selects the
    destination origin; ``offset`` still applies only to the source.
    All participating threads must use the same destination view.
    This form returns ``None``. Call ``block.sync()`` before another thread
    reads these rows. No asynchronous operation or extra padding is involved.

    With ``items_per_thread``, ``alignment`` requests a minimum power-of-two
    byte alignment for the returned payload's materialized storage, as in
    ``ThreadData``. An existing ``output`` retains its own alignment and
    requires ``alignment=None``. This option does not describe the source
    memory's alignment or change the selected item layout.

    To update an existing native CuTe register
    tensor, load a payload and call
    ``tensor.store(payload.to_tensor_ssa())`` with compatible shape and item
    order. The ThreadData forms materialize a temporary register tensor and
    place its scalar values in ThreadData. The shared-output form binds CUB's
    output array directly to each thread's shared row.

    Load populates the payload in the selected algorithm's layout. Beyond
    ``valid_items``, slots have unspecified values unless ``oob_default`` is
    supplied, even if initialized before Load. A default requires
    ``valid_items``. A runtime default must have the memory dtype. A Python or
    NumPy default must be finite and within the memory dtype's range; a float
    default requires a floating-point dtype.

    The count ranges from zero through the group's full tile size. ``offset``
    is a nonnegative element offset. Counts, offsets, and supplied defaults
    must agree within the group. Physical and logical warps receive
    consecutive tiles in linear block-thread order; the compiler adds that
    group's tile origin to the offset. Different groups can use different
    controls. The caller must provide enough accessible memory for the
    selected prefix at the resulting offset.

    For ``this_warp().group_by(width)``, ``width`` must be 1, 2, 4, 8, 16, or
    32. Each logical tile contains ``width * items_per_thread`` elements. The
    enclosing block must contain complete 32-thread physical warps.

    The source must expose a raw pointer and a provably compact layout. A bare
    pointer object with no shape or stride metadata is also accepted, but its
    capacity is not checked. Register or local-memory tensors are rejected.
    Load reads addressable memory, such as global or shared memory. Shared
    tensors from ``SmemAllocator`` and ``TempStorage.reserve`` are supported
    when their dtype and compact layout meet these requirements. Synchronize
    the participating threads after shared-memory writes and before Load
    reads values produced by other threads. DIRECT, STRIPED, and VECTORIZE
    need no shared scratch or
    reuse barrier; an
    accepted block descriptor does not change that.

    Transpose algorithms use shared scratch. With no ``temp_storage``, the
    compiler allocates it and inserts a trailing reuse barrier. An explicit
    :class:`cuda.coop.cutlass.TempStorage` is supported only for block calls.
    It sets size, alignment, sharing, and synchronization policy; its default
    ``auto_sync=False`` requires a barrier before reuse. Each physical or
    logical warp uses an independent scratch slice and a reuse barrier that
    covers only its own lanes.

    Shared output must not overlap the source range or primitive scratch.
    A detected output/scratch overlap triggers a device trap and leaves the
    CUDA context unusable.
    Separate descriptors or ``TempStorage(sharing="exclusive")`` keep
    reservations and explicit scratch in disjoint storage. DIRECT and
    TRANSPOSE produce blocked rows, giving a flat copy of each source tile.
    STRIPED preserves CUB's striped ownership, so concatenating the rows
    permutes the tile. A scratch reuse barrier does not replace the caller's
    synchronization before cross-thread consumption.

    Examples
    --------
    Copy a partial tile between different source and destination offsets.
    The launcher accepts device pointers and a compile-time
    ``items_per_thread`` value.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/cutlass/runtime/test_qualified_load_store_examples.py
        :language: python
        :start-after: # qualified-load-store-example-begin
        :end-before: # qualified-load-store-example-end
        :dedent: 4
    """

    returns_output = output is None
    shared_output = not returns_output and items_per_thread is not None
    if returns_output:
        if items_per_thread is None:
            raise TypeError(
                f"{_SCOPE}.load requires output or items_per_thread"
            )
        if _common_root_operation_name() == "load":
            with _common_root_operation_scope("ThreadData"):
                output = ThreadData(items_per_thread, alignment=alignment)
        else:
            output = ThreadData(items_per_thread, alignment=alignment)
    elif shared_output and isinstance(output, ThreadData):
        raise TypeError(
            f"{_SCOPE}.load accepts ThreadData output or items_per_thread, "
            "not both"
        )
    if shared_output:
        if isinstance(items_per_thread, bool) or not isinstance(
            items_per_thread, Integral
        ):
            raise TypeError(
                f"{_SCOPE}.load items_per_thread must be a compile-time integer"
            )
        if items_per_thread <= 0:
            raise ValueError(f"{_SCOPE}.load items_per_thread must be positive")
    elif not isinstance(output, ThreadData):
        raise TypeError(f"{_SCOPE}.load output must be ThreadData")
    if not returns_output and alignment is not None:
        raise TypeError(
            f"{_SCOPE}.load alignment requires items_per_thread and no output"
        )
    if oob_default is not None and valid_items is None:
        raise ValueError(f"{_SCOPE}.load oob_default requires valid_items")
    group, launch, algorithm = _resolve_group(
        group, algorithm, temp_storage, "load"
    )
    from ._lowering._load_store import provider_load

    provider_load(
        group=group,
        launch=launch,
        source=source,
        output=output,
        algorithm=algorithm,
        valid_items=valid_items,
        valid_items_binding=_classify_integer_binding(
            valid_items, name="valid_items"
        ),
        oob_default=oob_default,
        oob_default_binding=_classify_oob_default(oob_default),
        offset=offset,
        offset_binding=_classify_integer_binding(offset, name="offset"),
        temp_storage=temp_storage,
        items_per_thread=items_per_thread if shared_output else None,
    )
    return output if returns_output else None


def store(
    group: BlockGroup | WarpGroup,
    destination: object,
    value: _CommonNumericT | CommonThreadDataLike[_CommonNumericT],
    /,
    *,
    algorithm: BlockLoadStoreAlgorithm | WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    offset: IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> None:
    """Store per-thread values into a contiguous group tile.

    Shared parameters and participation follow :func:`cuda.coop.store`. This
    implementation accepts blocks and physical or logical warps. Each thread
    supplies a scalar or an initialized CUTLASS ThreadData payload whose dtype
    matches the destination. As with :func:`cuda.coop.store`, do not rely on
    the payload's contents after a transpose Store; copy values needed later.

    ``valid_items`` selects a prefix from zero through the group's full tile
    size. ``offset`` is a nonnegative element offset. Both must agree within
    the group. Physical and logical warp widths, tile origins, and prefix
    bounds follow :func:`load`. Different groups can use different controls.
    The caller must provide enough accessible destination memory for the
    prefix at the resulting offset. Items outside it are not written. The
    destination has the same pointer and layout requirements as :func:`load`.

    Transpose algorithms use shared scratch. With no ``temp_storage``, the
    compiler allocates it and inserts a trailing reuse barrier. An explicit
    descriptor is supported only for block calls and controls allocation and
    reuse. The caller must synchronize before reuse unless ``auto_sync=True``.
    Physical and logical warps use independent scratch slices and a barrier
    that covers only the group's lanes. DIRECT, STRIPED, and VECTORIZE need no
    scratch or reuse barrier.

    Examples
    --------
    Copy a partial tile between different source and destination offsets.
    The launcher accepts device pointers and a compile-time
    ``items_per_thread`` value.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/cutlass/runtime/test_qualified_load_store_examples.py
        :language: python
        :start-after: # qualified-load-store-example-begin
        :end-before: # qualified-load-store-example-end
        :dedent: 4
    """

    group, launch, algorithm = _resolve_group(
        group, algorithm, temp_storage, "store"
    )
    from ._lowering._load_store import provider_store

    provider_store(
        group=group,
        launch=launch,
        destination=destination,
        value=value,
        algorithm=algorithm,
        valid_items=valid_items,
        valid_items_binding=_classify_integer_binding(
            valid_items, name="valid_items"
        ),
        offset=offset,
        offset_binding=_classify_integer_binding(offset, name="offset"),
        temp_storage=temp_storage,
    )


def _normalize_algorithm(algorithm: Any) -> GroupLoadStoreAlgorithm:
    """Resolve an enum or normalized Load/Store algorithm name.

    Strip surrounding whitespace and accept case and hyphen variants before
    matching the shared selector. Report available choices for unknown names.
    """

    token = getattr(algorithm, "value", algorithm)
    if isinstance(token, str):
        token = token.strip().lower().replace("-", "_")
    try:
        return GroupLoadStoreAlgorithm(token)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(item.value for item in GroupLoadStoreAlgorithm)
        raise ValueError(
            f"{_SCOPE}.load/store algorithm must be one of {choices}"
        ) from exc


def _is_boolean(value: Any) -> bool:
    """Recognize Python, NumPy, and DSL booleans before integer checks."""

    if isinstance(value, bool):
        return True
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        if isinstance(value, np.bool_):
            return True
    from cutlass.base_dsl.typing import Boolean

    return isinstance(value, Boolean)


def _classify_integer_binding(value: Any, *, name: str) -> ArgumentBinding:
    """Separate omitted, embedded, and runtime integer controls.

    Reject booleans. Validate static offset bounds here; shared planning
    checks static valid counts against the tile size. DSL integer values
    remain runtime bindings for the provider ABI.
    """

    if value is None:
        return ArgumentBinding.omitted()
    if _is_boolean(value):
        raise TypeError(f"{_SCOPE}.load/store {name} must be an integer")
    if isinstance(value, Integral):
        normalized = int(value)
        if name == "offset" and normalized < 0:
            raise ValueError(f"{_SCOPE}.load/store offset must be non-negative")
        if name == "offset" and normalized > _MAX_STATIC_OFFSET:
            raise ValueError(
                f"{_SCOPE}.load/store offset must fit a signed 64-bit integer"
            )
        return ArgumentBinding.static(normalized)
    from cutlass.base_dsl.typing import Integer

    if isinstance(value, Integer):
        return ArgumentBinding.runtime()
    raise TypeError(
        f"{_SCOPE}.load/store {name} must be an integer, "
        f"not {type(value).__name__}"
    )


def _classify_oob_default(value: Any) -> ArgumentBinding:
    """Embed host numeric defaults and retain DSL inputs for runtime.

    Embed Python and NumPy numbers as plain int or float constants, and keep
    DSL values as runtime inputs. Reject booleans and non-finite floats here.
    The provider later checks range and float-to-integer compatibility with
    the memory dtype.
    """

    if value is None:
        return ArgumentBinding.omitted()
    if _is_boolean(value):
        raise TypeError(
            f"{_SCOPE}.load oob_default must be numeric, not boolean"
        )
    if isinstance(value, Integral):
        return ArgumentBinding.static(int(value))
    if isinstance(value, Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{_SCOPE}.load oob_default must be finite")
        return ArgumentBinding.static(normalized)
    from cutlass.base_dsl.typing import Numeric

    if isinstance(value, Numeric):
        return ArgumentBinding.runtime()
    raise TypeError(
        f"{_SCOPE}.load oob_default must be a numeric scalar, not "
        f"{type(value).__name__}"
    )


__all__ = [
    "_is_boolean",
    "load",
    "store",
]
