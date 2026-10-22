# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, overload

from typing_extensions import TypeVar

from cuda.coop._typing import (
    BlockLoadStoreAlgorithm,
    CommonNumericScalar,
    IntegerValue,
    TempStorageLike,
    ThreadDataLike,
    ValidItems,
    WarpLoadStoreAlgorithm,
)

from .._core.api.load_store import store as store
from .._core.api.thread_group import BlockGroup, WarpGroup
from ._thread_data import ThreadData

_CommonNumericT = TypeVar("_CommonNumericT", bound=CommonNumericScalar)

@overload
def load(
    group: BlockGroup,
    source: object,
    output: None = None,
    /,
    *,
    items_per_thread: int,
    algorithm: BlockLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    oob_default: None = None,
    offset: IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData[Any]:
    """Return a fresh payload containing one cooperative block tile."""

@overload
def load(
    group: BlockGroup,
    source: object,
    output: None = None,
    /,
    *,
    items_per_thread: int,
    algorithm: BlockLoadStoreAlgorithm = "direct",
    valid_items: ValidItems,
    oob_default: CommonNumericScalar | float,
    offset: IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData[Any]:
    """Return a partial block tile with invalid items filled."""

@overload
def load(
    group: WarpGroup,
    source: object,
    output: None = None,
    /,
    *,
    items_per_thread: int,
    algorithm: WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    oob_default: None = None,
    offset: IntegerValue | None = None,
    temp_storage: None = None,
) -> ThreadData[Any]:
    """Return a fresh payload containing one physical or logical warp tile."""

@overload
def load(
    group: WarpGroup,
    source: object,
    output: None = None,
    /,
    *,
    items_per_thread: int,
    algorithm: WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems,
    oob_default: CommonNumericScalar | float,
    offset: IntegerValue | None = None,
    temp_storage: None = None,
) -> ThreadData[Any]:
    """Return a partial warp tile with invalid items filled."""

@overload
def load(
    group: BlockGroup,
    source: object,
    output: ThreadDataLike[_CommonNumericT],
    /,
    *,
    items_per_thread: None = None,
    algorithm: BlockLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    oob_default: None = None,
    offset: IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> None:
    """Populate ``output`` in place with one cooperative block tile."""

@overload
def load(
    group: BlockGroup,
    source: object,
    output: ThreadDataLike[_CommonNumericT],
    /,
    *,
    items_per_thread: None = None,
    algorithm: BlockLoadStoreAlgorithm = "direct",
    valid_items: ValidItems,
    oob_default: _CommonNumericT | float,
    offset: IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> None:
    """Populate a partial block tile and fill invalid items."""

@overload
def load(
    group: WarpGroup,
    source: object,
    output: ThreadDataLike[_CommonNumericT],
    /,
    *,
    items_per_thread: None = None,
    algorithm: WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    oob_default: None = None,
    offset: IntegerValue | None = None,
    temp_storage: None = None,
) -> None:
    """Populate ``output`` in place with one physical or logical warp tile."""

@overload
def load(
    group: WarpGroup,
    source: object,
    output: ThreadDataLike[_CommonNumericT],
    /,
    *,
    items_per_thread: None = None,
    algorithm: WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems,
    oob_default: _CommonNumericT | float,
    offset: IntegerValue | None = None,
    temp_storage: None = None,
) -> None:
    """Load a partial warp tile and fill invalid output items."""
