# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Identify cooperative markers and provider factories by callable identity.

The typed pipeline recognizes public marker aliases through the marker registry.
Provider factories retain their storage and thread-scope metadata for source
specialization. Registration does not compile or execute device code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from cuda.coop._core import SynchronizationScope

_CallableT = TypeVar("_CallableT", bound=Callable[..., Any])


class StorageABI(str, Enum):
    """Describe the provider's temporary-storage calling convention.

    ``NONE`` needs no storage operand. ``LEADING_POINTER`` reserves the first
    provider argument for scratch memory, supplied separately from the
    operation's ordinary operands.
    """

    NONE = "none"
    LEADING_POINTER = "leading_pointer"


@dataclass(frozen=True)
class FactoryOperation:
    """Declare the call contract of a registered provider factory.

    Factories use this record to select storage and thread scopes during
    specialization. Registration declares the contract; it does not inspect
    generated device code.

    Attributes
    ----------
    operation : str
        Operation identifier used to select its provider.
    namespace : str
        Provider namespace, such as ``"block"`` or ``"warp"``.
    storage_abi : StorageABI
        Whether the generated call receives a leading scratch pointer.
    execution_scope : SynchronizationScope
        Threads that execute the primitive together.
    synchronization_scope : SynchronizationScope
        Declared synchronization scope: ``NONE`` or the execution scope.
        Storage planning decides which reuse barriers to emit.
    """

    operation: str
    namespace: str
    storage_abi: StorageABI
    execution_scope: SynchronizationScope
    synchronization_scope: SynchronizationScope

    def __post_init__(self) -> None:
        for name in ("operation", "namespace"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "storage_abi", StorageABI(self.storage_abi))
        object.__setattr__(
            self,
            "execution_scope",
            SynchronizationScope(self.execution_scope),
        )
        object.__setattr__(
            self,
            "synchronization_scope",
            SynchronizationScope(self.synchronization_scope),
        )
        if self.synchronization_scope not in {
            SynchronizationScope.NONE,
            self.execution_scope,
        }:
            raise ValueError(
                "synchronization_scope must be NONE or match execution_scope"
            )


_GROUP_OPERATIONS: dict[Callable[..., Any], str] = {}
_FACTORY_OPERATIONS: dict[Callable[..., Any], FactoryOperation] = {}


def group_operation(operation: str) -> Callable[[_CallableT], _CallableT]:
    """Register a public marker without wrapping it, preserving aliases."""

    def decorate(function: _CallableT) -> _CallableT:
        existing = _GROUP_OPERATIONS.get(function)
        if existing is not None and existing != operation:
            raise RuntimeError(
                f"group marker {function!r} is already registered as "
                f"{existing!r}"
            )
        _GROUP_OPERATIONS[function] = operation
        return function

    return decorate


def group_operation_name(function: Any) -> str | None:
    """Return the operation for an exactly registered group marker."""

    return _GROUP_OPERATIONS.get(function)


def register_factory(
    function: _CallableT,
    *,
    operation: str,
    namespace: str,
    storage_abi: StorageABI,
    execution_scope: SynchronizationScope,
    synchronization_scope: SynchronizationScope,
) -> _CallableT:
    """Register a provider factory and its storage and thread scopes.

    Factories use this metadata to validate storage choices during
    specialization. Their import paths and names do not determine their ABI.
    Registration only declares metadata; the factory and source emitter must
    implement the declared storage and synchronization behavior.

    Parameters
    ----------
    function : callable
        Host-side callable that builds an ``Algorithm`` from specialization
        keywords. See the module overview for the distinction from
        public group markers.
    operation : str
        Non-empty operation identifier, such as ``"load"`` or ``"store"``.
    namespace : str
        Non-empty provider namespace, such as ``"block"`` or ``"warp"``.
    storage_abi : StorageABI
        Whether provider calls take a leading scratch pointer.
    execution_scope : SynchronizationScope
        Scope of threads executing the cooperative operation.
    synchronization_scope : SynchronizationScope
        Declared synchronization scope: ``NONE`` or the execution scope.

    Returns
    -------
    callable
        The original factory after registry insertion. Identical repeated
        registration is accepted.

    Raises
    ------
    TypeError
        ``function`` is not callable.
    ValueError
        Names, enum values, or the scope relationship are invalid.
    RuntimeError
        This exact factory is already registered with different metadata.
    """

    if not callable(function):
        raise TypeError("lowering factory must be callable")
    metadata = FactoryOperation(
        operation=operation,
        namespace=namespace,
        storage_abi=storage_abi,
        execution_scope=execution_scope,
        synchronization_scope=synchronization_scope,
    )
    existing = _FACTORY_OPERATIONS.get(function)
    if existing is not None and existing != metadata:
        raise RuntimeError(
            f"lowering factory {function!r} is already registered as "
            f"{existing!r}"
        )
    _FACTORY_OPERATIONS[function] = metadata
    return function


def factory_operation(function: Any) -> FactoryOperation | None:
    """Return metadata for an exactly registered lowering factory."""

    return _FACTORY_OPERATIONS.get(function)


__all__ = [
    "FactoryOperation",
    "StorageABI",
    "factory_operation",
    "group_operation",
    "group_operation_name",
    "register_factory",
]
