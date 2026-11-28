# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check static shapes and strides for contiguous memory access."""

from __future__ import annotations

from typing import Any

from cutlass import cute


def _optional_attr(value: Any, name: str) -> Any:
    """Treat unavailable foreign metadata as missing evidence."""

    try:
        return getattr(value, name, None)
    except Exception:  # noqa: BLE001
        # Optional foreign metadata may reject access.
        return None


def _static_layout_int(value: Any) -> int | None:
    """Accept only a statically inspectable, non-Boolean layout integer."""

    if isinstance(value, bool):
        return None
    try:
        normalized = value.__index__()
    except Exception:  # noqa: BLE001
        # Dynamic DSL extents may reject integer conversion.
        return None
    if isinstance(normalized, bool):
        return None
    return int(normalized)


def _layout_leaf_pairs(
    shape: Any,
    strides: Any,
) -> tuple[tuple[Any, Any], ...] | None:
    """Flatten congruent shape/stride trees into scalar leaf pairs."""

    shape_is_tree = isinstance(shape, (tuple, list))
    strides_is_tree = isinstance(strides, (tuple, list))
    if shape_is_tree != strides_is_tree:
        return None
    if not shape_is_tree:
        return ((shape, strides),)
    if len(shape) != len(strides):
        return None

    leaves: list[tuple[Any, Any]] = []
    for shape_child, strides_child in zip(shape, strides):
        child_leaves = _layout_leaf_pairs(shape_child, strides_child)
        if child_leaves is None:
            return None
        leaves.extend(child_leaves)
    return tuple(leaves)


def contiguous_layout_extent(value: Any) -> tuple[int | None, str | None]:
    """Prove compactness and return element capacity or a diagnostic.

    Missing shape and stride leave the extent unknown. The caller must then
    require a direct raw pointer. Unit extents do not constrain stride.
    """

    if isinstance(_optional_attr(value, "layout"), cute.ComposedLayout):
        return None, "has a composed layout"

    shape = _optional_attr(value, "shape")
    strides_value = _optional_attr(value, "strides")
    if strides_value is None:
        strides_value = _optional_attr(value, "stride")
    if shape is None and strides_value is None:
        return None, None
    if shape is None or strides_value is None:
        return None, "does not expose both shape and stride metadata"
    leaves = _layout_leaf_pairs(shape, strides_value)
    if leaves is None:
        return None, "has incongruent shape and stride layouts"

    static_shape = tuple(_static_layout_int(extent) for extent, _ in leaves)
    static_strides = tuple(_static_layout_int(stride) for _, stride in leaves)
    if any(value is None for value in (*static_shape, *static_strides)):
        return None, "is not statically provable as compact"
    normalized_shape = tuple(int(extent) for extent in static_shape)
    normalized_strides = tuple(int(stride) for stride in static_strides)
    if any(extent <= 0 for extent in normalized_shape):
        return None, "has a non-positive static extent"
    if any(
        stride <= 0
        for stride, extent in zip(normalized_strides, normalized_shape)
        if extent != 1
    ):
        return None, "has a non-positive static stride"

    expected_stride = 1
    for stride, extent in sorted(
        zip(normalized_strides, normalized_shape),
        key=lambda entry: entry[0],
    ):
        if extent == 1:
            continue
        if stride != expected_stride:
            return None, "is not a compact contiguous layout"
        expected_stride *= extent
    return expected_stride, None
