# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing rules that an extension registers with Numba-CUDA-MLIR type inference.

The coop-side form of a fourth registry bucket, next to templates, attributes
and globals: rules keyed by a Numba type class, which type inference runs at
fixed points of its own propagation. A rule never sees a constraint whose
types do not hold an instance of its key, and nothing here knows which
types an extension registers.

Three kinds of rule, each called with the types that matched its key:

=========  ==================================================================
Kind       Called as, and what it may do
=========  ==================================================================
``gate``   ``fn(fnty, arg_types, constraint, released)`` when a call's
           arguments are typed and one of them, or the callee's receiver,
           matches. Returning ``WAIT`` leaves the call untyped for now; it
           is retried on every round, and once propagation is otherwise
           stuck it is released: ``released`` is then true and the call
           types whatever the rule returns.
``settle`` ``fn(typeinfer, matching, waiting)`` at each fixed point, with the
           matching type variables and the waiting calls. It may only widen
           types, through ``typeinfer.add_type``, and returns a list of
           conflict errors or ``None``.
``check``  ``fn(typeinfer, matching, released)`` once, at the last fixed
           point. ``released`` maps each released call to its argument types
           (and callee) when it typed, and now. It returns a list of
           conflict errors or ``None``; it changes nothing.
=========  ==================================================================

Type inference then runs in strata: propagate to a fixed point; settle; if
that changed a type, propagate again; else release the waiting calls and
propagate again; else check. Each step either widens a type or releases a
call that was never released before, so the loop ends. With no rule
registered, type inference is the compiler's own.

For audit, ``RULES.describe()`` lists the registered rules, and inside
``with RULES.tracing() as events:`` each decision a rule makes is recorded
as an ``Event``: a gate's call waits, opens or is released; a settle or
check rule ran, or reported a conflict. ``with RULES.without(rule):`` runs
type inference as if ``rule`` were not registered.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from functools import reduce
from typing import Any, NamedTuple

from . import _bridges as nb
from ._ncm_patches import Patch, Pin, typeinfer

GATE, SETTLE, CHECK = "gate", "settle", "check"
KINDS = (GATE, SETTLE, CHECK)


class _Wait:
    def __repr__(self) -> str:
        return "WAIT"


WAIT = _Wait()


class Event(NamedTuple):
    """One decision of a rule; ``line`` is the kernel source line, if any."""

    kind: str
    rule: str
    line: int | None
    what: str


class RuleSet:
    """Typing rules by kind and by the Numba type class they are keyed on.

    ``lookups`` counts how often type inference consulted the set, and
    ``calls`` how often a rule ran.
    """

    def __init__(self) -> None:
        self._rules: dict[str, dict[type, list[Callable]]] = {
            kind: {} for kind in KINDS
        }
        self.lookups = 0
        self.calls = 0
        self._trace: list[Event] | None = None

    def describe(self) -> list[tuple[str, str, str, str]]:
        """``(kind, key, rule, summary)`` for every registered rule."""

        rows = []
        for kind in KINDS:
            for key, fns in self._rules[kind].items():
                for fn in fns:
                    doc = (fn.__doc__ or "").strip().splitlines()
                    rows.append(
                        (kind, key.__name__, fn.__name__, doc[0] if doc else "")
                    )
        return rows

    @contextmanager
    def tracing(self) -> Iterator[list[Event]]:
        """Record the rules' decisions while the block runs."""

        saved, self._trace = self._trace, []
        try:
            yield self._trace
        finally:
            self._trace = saved

    @contextmanager
    def without(self, *fns: Callable) -> Iterator[None]:
        """Unregister ``fns`` while the block runs, to show what they do."""

        saved = {kind: dict(table) for kind, table in self._rules.items()}
        for table in self._rules.values():
            for key in list(table):
                table[key] = [fn for fn in table[key] if fn not in fns]
                if not table[key]:
                    del table[key]
        try:
            yield
        finally:
            self._rules = saved

    def note(self, kind: str, fn: Callable, loc: Any, what: str) -> None:
        if self._trace is not None:
            line = getattr(loc, "line", None)
            self._trace.append(Event(kind, fn.__name__, line, what))

    def register(self, key: type, kind: str, fn: Callable) -> Callable:
        if kind not in KINDS:
            raise ValueError(f"unknown typing rule kind {kind!r}")
        if not (isinstance(key, type) and issubclass(key, nb.types.Type)):
            raise TypeError(f"a typing rule key is a Numba type class: {key}")
        self._rules[kind].setdefault(key, []).append(fn)
        return fn

    def __bool__(self) -> bool:
        return any(self._rules.values())

    def has(self, kind: str) -> bool:
        return bool(self._rules[kind])

    def matching(self, kind: str, tys: Iterable[Any]) -> tuple[Callable, ...]:
        """The rules of ``kind`` whose key matches any of ``tys``."""

        self.lookups += 1
        table = self._rules[kind]
        found: dict[Callable, None] = {}
        for ty in tys:
            for key in _keys(table, ty):
                found.update(dict.fromkeys(table[key]))
        return tuple(found)


def _keys(table: dict, ty: Any) -> Iterable[type]:
    """The keys of ``table`` that ``ty`` matches, by the MRO of its class.

    A key matches an instance of it, a tuple holding one, and a bound
    function whose receiver matches.
    """

    if ty is None:
        return
    for cls in type(ty).__mro__:
        if cls in table:
            yield cls
    if isinstance(ty, nb.types.BoundFunction):
        yield from _keys(table, ty.this)
    elif isinstance(ty, nb.types.BaseTuple):
        for item in ty.types:
            yield from _keys(table, item)


RULES = RuleSet()
register = RULES.register


class _State:
    """The calls of one type inference that wait or were released.

    ``waiting`` maps a waiting call to the gate rule that holds it.
    ``released`` maps a released call to the types it typed with, which are
    recorded the first time it types after its release.
    """

    def __init__(self) -> None:
        self.waiting: dict[Any, Callable] = {}
        self.released: dict[Any, tuple | None] = {}


def _state(typeinfer_) -> _State:
    state = vars(typeinfer_).get("_typing_rules")
    if state is None:
        state = typeinfer_._typing_rules = _State()
    return state


def _call_types(constraint, typevars, fnty) -> tuple | None:
    folded = typeinfer.fold_arg_vars(
        typevars, constraint.args, constraint.vararg, constraint.kws
    )
    if folded is None:
        return None
    pos_args, kw_args = folded
    return (*pos_args, *kw_args.values(), fnty)


def _gate(constraint, typeinfer_, typevars, fnty) -> bool:
    """Whether a gate rule makes ``constraint`` wait."""

    seen = _call_types(constraint, typevars, fnty)
    if seen is None:
        return False
    arg_types = seen[:-1]
    # As the compiler: wait for imprecise arguments other than arrays.
    if not all(
        a.is_precise() or isinstance(a, nb.types.Array) for a in arg_types
    ):
        return False
    rules = RULES.matching(GATE, seen)
    if not rules:
        return False
    state = _state(typeinfer_)
    released = constraint in state.released
    for fn in rules:
        RULES.calls += 1
        if fn(fnty, arg_types, constraint, released) is WAIT and not released:
            if constraint not in state.waiting:
                RULES.note(GATE, fn, constraint.loc, "wait")
            state.waiting[constraint] = fn
            return True
    held = state.waiting.pop(constraint, None)
    if held is not None:
        RULES.note(GATE, held, constraint.loc, "open")
    if released and state.released[constraint] is None:
        state.released[constraint] = seen
    return False


def _matching_vars(kind: str, typeinfer_) -> dict[Callable, dict[str, Any]]:
    by_rule: dict[Callable, dict[str, Any]] = {}
    for name, tv in typeinfer_.typevars.items():
        if tv.type is None:
            continue
        for fn in RULES.matching(kind, (tv.type,)):
            by_rule.setdefault(fn, {})[name] = tv.type
    return by_rule


def _run(kind: str, typeinfer_, context: Any) -> list:
    errors = []
    for fn, matching in _matching_vars(kind, typeinfer_).items():
        RULES.calls += 1
        found = list(fn(typeinfer_, matching, context) or ())
        for error in found:
            RULES.note(kind, fn, getattr(error, "loc", None), "conflict")
        if not found:
            RULES.note(kind, fn, None, "ok")
        errors.extend(found)
    return errors


def writes(typeinfer_) -> Iterator[tuple[str, Any, Any]]:
    """Every indexed write: its target's name, its value's type, its location.

    A value's type is ``None`` while it is unknown.
    """

    setitems = (typeinfer.SetItemConstraint, typeinfer.StaticSetItemConstraint)
    for constraint in typeinfer_.constraints.constraints:
        if isinstance(constraint, setitems):
            value = typeinfer_.typevars[constraint.value.name].type
            yield constraint.target.name, value, constraint.loc


def _released_now(typeinfer_, state: _State) -> dict[Any, tuple]:
    typevars = typeinfer_.typevars
    released = {}
    for constraint, seen in state.released.items():
        if seen is None:
            continue
        func = constraint.func
        fnty = typevars[func].type if isinstance(func, str) else seen[-1]
        now = _call_types(constraint, typevars, fnty)
        released[constraint] = (seen, now)
    return released


def _raise_or_return(errors: list, raise_errors: bool) -> list | None:
    if not errors:
        return None
    if not raise_errors:
        return errors
    force_lit_args = [
        e for e in errors if isinstance(e, typeinfer.ForceLiteralArg)
    ]
    if not force_lit_args:
        raise errors[0]
    raise reduce(operator.or_, force_lit_args)


def _engine() -> tuple[tuple[type, str, Any], ...]:
    """Run typing rules at the fixed points of type inference.

    ``CallConstraint.resolve`` asks the gate rules whether a call waits;
    ``TypeInferer.propagate`` runs the strata of the module docstring.

    Scope: registered rules only. With no gate rule, a call resolves as
    before; with no rule at all, propagation is the compiler's own. A
    kernel whose types match no key consults the set but runs no rule.
    """

    propagate = typeinfer.TypeInferer.propagate
    resolve = typeinfer.CallConstraint.resolve

    def resolve_with_gates(self, typeinfer_, typevars, fnty):
        if RULES.has(GATE) and _gate(self, typeinfer_, typevars, fnty):
            return None
        return resolve(self, typeinfer_, typevars, fnty)

    def propagate_in_strata(self, raise_errors=True):
        if not RULES:
            return propagate(self, raise_errors=raise_errors)
        state = _state(self)
        while True:
            errors = propagate(self, raise_errors=False) or []
            token = self.get_state_token()
            conflicts = _run(SETTLE, self, tuple(state.waiting))
            if conflicts:
                errors.extend(conflicts)
                break
            if self.get_state_token() != token:
                continue
            if state.waiting:
                for constraint, fn in state.waiting.items():
                    RULES.note(GATE, fn, constraint.loc, "released")
                state.released.update(dict.fromkeys(state.waiting))
                state.waiting.clear()
                continue
            break
        errors.extend(_run(CHECK, self, _released_now(self, state)))
        return _raise_or_return(errors, raise_errors)

    return (
        (typeinfer.TypeInferer, "propagate", propagate_in_strata),
        (typeinfer.CallConstraint, "resolve", resolve_with_gates),
    )


ENGINE = Patch(
    name="typing rules",
    ask=(
        "U-R1 to U-R3 (an extension registers typing rules that make a call "
        "wait for a later round and check the last fixed point)"
    ),
    pins=(
        Pin(typeinfer.TypeInferer, "propagate", "9d846381cc969eb8"),
        Pin(typeinfer.TypeInferer, "get_state_token", "ed391a44719f2abc"),
        Pin(typeinfer.ConstraintNetwork, "propagate", "1e606cddef0a1677"),
        Pin(typeinfer.CallConstraint, "resolve", "b41a0de1f79a9922"),
        Pin(typeinfer.IntrinsicCallConstraint, "resolve", None),
        Pin(typeinfer, "fold_arg_vars", "581289ed10e3c724"),
        Pin(typeinfer.ConstraintNetwork, "__init__", "da386c542882c6a9"),
        Pin(typeinfer.SetItemConstraint, "__init__", "2f4967dd8b94a9a3"),
        Pin(typeinfer.StaticSetItemConstraint, "__init__", "39264e13c29c4e4a"),
    ),
    build=_engine,
)


__all__ = [
    "CHECK",
    "ENGINE",
    "GATE",
    "KINDS",
    "RULES",
    "SETTLE",
    "WAIT",
    "Event",
    "RuleSet",
    "register",
    "writes",
]
