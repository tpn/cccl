# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing rules for ``ThreadData`` payloads, registered with ``_typing_rules``.

A ``ThreadData`` without ``dtype=`` takes its dtype from the strongest
evidence, and a ``load`` into it outranks an indexed write. Type variables
only widen, so a read typed while only a write was known would keep the
write's dtype after a later ``load`` refines the payload. Two rules order
typing instead:

* gate: a call that reads a payload whose dtype comes from evidence weaker
  than a producer waits, unless it is a cooperative operation or a witness,
  which are the evidence. A waiting call types once propagation is
  otherwise stuck, against the payload's dtype at that point.
* check: a call released that way must still agree with its payload at the
  last fixed point; a producer that arrived only after the release (its
  source depended on the read) is a stale read, and an error.
* check: a payload whose dtype comes from indexed writes alone takes it from
  the first write typed, and converts the others silently; so all of its
  writes must have that dtype.

Indexed writes are ``setitem`` constraints, not calls, so they never wait.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from . import _bridges as nb
from . import _typing_rules as rules
from ._errors import TypedPipelineError
from ._types import PRODUCER, WRITE, ThreadDataType

_WITNESS_KEYS = frozenset(
    {"coop.thread_data.refine_load", "coop.thread_data.refine_store"}
)


def _payloads(ty: Any) -> Iterable[ThreadDataType]:
    if isinstance(ty, ThreadDataType):
        yield ty
    elif isinstance(ty, nb.types.BoundFunction):
        yield from _payloads(ty.this)
    elif isinstance(ty, nb.types.BaseTuple):
        for item in ty.types:
            yield from _payloads(item)


def _unsettled(ty: Any) -> bool:
    """Whether ``ty`` holds a payload a producer may still change."""

    return any(
        p.dtype.is_precise() and p.strength < PRODUCER for p in _payloads(ty)
    )


def _is_evidence(fnty: Any) -> bool:
    from ._templates import OPERATIONS

    key = getattr(fnty, "typing_key", None)
    try:
        return key in OPERATIONS or key in _WITNESS_KEYS
    except TypeError:
        return False


def read_after_producers(fnty, arg_types, constraint, released):
    """Gate: a payload's consumers type after its producers."""

    if released or _is_evidence(fnty):
        return None
    if any(_unsettled(ty) for ty in (*arg_types, fnty)):
        return rules.WAIT
    return None


def released_reads_agree(typeinfer_, matching, released):
    """Check: a released read typed against its payload's final dtype."""

    errors = []
    for constraint, (seen, now) in released.items():
        if now is None:
            continue
        for before, after in zip(seen, now):
            for old, new in zip(_payloads(before), _payloads(after)):
                if old.dtype == new.dtype:
                    continue
                errors.append(
                    TypedPipelineError(
                        f"coop.ThreadData: a read at line "
                        f"{constraint.loc.line} typed against {old}, but a "
                        f"producer that depends on it refined the payload to "
                        f"{new}. Pass dtype={new.dtype} to coop.ThreadData.",
                        loc=constraint.loc,
                    )
                )
    return errors


def writes_agree(typeinfer_, matching, released):
    """Check: a payload typed by indexed writes alone has one dtype."""

    written: dict[str, dict] = {}
    for name, value, loc in rules.writes(typeinfer_):
        payload = matching.get(name)
        if payload is None or payload.strength != WRITE or value is None:
            continue
        written.setdefault(name, {}).setdefault(nb.unliteral(value), loc)
    errors = []
    for name, dtypes in written.items():
        if set(dtypes) == {matching[name].dtype}:
            continue
        found = ", ".join(sorted(map(str, dtypes)))
        errors.append(
            TypedPipelineError(
                f"coop.ThreadData {name!r}: its writes disagree on the dtype "
                f"({found}). Pass dtype= or make the writes agree.",
                loc=next(iter(dtypes.values())),
            )
        )
    return errors


_registered = False


def register() -> None:
    """Register the payload rules once."""

    global _registered
    if _registered:
        return
    rules.register(ThreadDataType, rules.GATE, read_after_producers)
    rules.register(ThreadDataType, rules.CHECK, released_reads_agree)
    rules.register(ThreadDataType, rules.CHECK, writes_agree)
    _registered = True


__all__ = [
    "read_after_producers",
    "register",
    "released_reads_agree",
    "writes_agree",
]
