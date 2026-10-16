# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing rules: how a rule's key matches, and the payload rules in kernels."""

from __future__ import annotations

import inspect
import re

import numpy as np
import pytest
from numba_cuda_mlir import cuda, types

from cuda import coop
from cuda.coop.numba_mlir._pipeline import _payload_rules, testing
from cuda.coop.numba_mlir._pipeline._errors import TypedPipelineError
from cuda.coop.numba_mlir._pipeline._typing_rules import RULES, Event

from ._support import BLOCK, CHIP, f32_in_out, requires_typed_pipeline


def _compile(kernel):
    return testing.compile_kernel(kernel, f32_in_out(), block=BLOCK, chip=CHIP)


def _line(fn, text):
    lines, start = inspect.getsourcelines(fn)
    return start + next(i for i, line in enumerate(lines) if text in line)


@requires_typed_pipeline
def test_the_registered_rules_are_coops_typing_spec():
    assert [row[:3] for row in RULES.describe()] == [
        ("gate", "ThreadDataType", "read_after_producers"),
        ("check", "ThreadDataType", "released_reads_agree"),
        ("check", "ThreadDataType", "writes_agree"),
    ]


def test_a_key_matches_instances_tuples_and_receivers():
    from cuda.coop.numba_mlir._pipeline import _bridges as nb
    from cuda.coop.numba_mlir._pipeline._typing_rules import GATE, RuleSet

    class Base(nb.types.Opaque):
        pass

    class Derived(Base):
        pass

    class Method:
        key = "test.method"

    def rule(*args):
        return None

    rules = RuleSet()
    rules.register(Base, GATE, rule)
    derived = Derived("derived")
    bound = nb.types.BoundFunction(Method, derived)
    nested = nb.types.Tuple((nb.types.int64, nb.types.UniTuple(derived, 1)))
    for ty in (derived, nested, bound):
        assert rules.matching(GATE, (ty,)) == (rule,)
    assert rules.matching(GATE, (nb.types.int64, nb.types.float32[::1])) == ()
    assert rules.lookups == 4
    with pytest.raises(ValueError, match="unknown typing rule kind"):
        rules.register(Base, "widen", rule)
    with pytest.raises(TypeError, match="a Numba type class"):
        rules.register(int, GATE, rule)


def _review(source, output):
    payload = coop.ThreadData(items_per_thread=1)
    payload[0] = np.float64(1.25)
    observed = payload[0]
    coop.load(coop.this_block(), source, payload)
    if cuda.threadIdx.x == 0:
        output[0] = observed + payload[0]


@requires_typed_pipeline
def test_a_read_before_the_load_waits_for_it():
    compiled = _compile(_review)
    assert set(re.findall(r"\bobserved = .*?:: (\S+)", compiled.typed_ir)) == {
        "float32"
    }
    source = np.full(32, 1.0, np.float32)
    output = np.zeros(1, np.float32)
    cuda.jit(_review)[1, 32](source, output)
    assert output[0] == 2.25


@requires_typed_pipeline
def test_the_trace_shows_the_read_wait_for_the_load():
    line = _line(_review, "observed = payload[0]")
    with RULES.tracing() as events:
        _compile(_review)
    assert [e for e in events if e.kind == "gate"] == [
        Event("gate", "read_after_producers", line, "wait"),
        Event("gate", "read_after_producers", line, "open"),
    ]


@requires_typed_pipeline
def test_without_the_gate_the_read_keeps_the_write_dtype():
    with (
        RULES.without(_payload_rules.read_after_producers),
        pytest.raises(
            TypedPipelineError, match=r"typed as float64, but the payload's"
        ),
    ):
        _compile(_review)


def _mixed_writes(source, output):
    payload = coop.ThreadData(2)
    payload[0] = np.int32(2)
    payload[1] = np.float64(1.25)
    if cuda.threadIdx.x == 0:
        output[0] = payload[0] + payload[1]


@requires_typed_pipeline
def test_writes_that_disagree_are_refused_in_typing():
    with (
        RULES.tracing() as events,
        pytest.raises(
            TypedPipelineError,
            match=r"'payload': its writes disagree on the dtype "
            r"\(float64, int32\)",
        ),
    ):
        _compile(_mixed_writes)
    line = _line(_mixed_writes, "np.int32(2)")
    assert Event("check", "writes_agree", line, "conflict") in events


@requires_typed_pipeline
def test_without_the_check_disagreeing_writes_convert_silently():
    output = np.zeros(1, np.float32)
    with RULES.without(_payload_rules.writes_agree):
        cuda.jit(_mixed_writes)[1, 32](np.zeros(32, np.float32), output)
    assert output[0] == 3.0


@cuda.jit(device=True, inline="never")
def _pick(source, value):
    return source


@requires_typed_pipeline
def test_a_producer_that_depends_on_a_released_read_is_refused():
    def kernel(source, output):
        payload = coop.ThreadData(1)
        payload[0] = np.float64(1.25)
        observed = payload[0]
        coop.load(coop.this_block(), _pick(source, observed), payload)
        output[0] = observed

    with pytest.raises(
        TypedPipelineError,
        match=(
            r"a read at line \d+ typed against coop\.ThreadData\(float64, "
            r"items=1, write\), but a producer that depends on it refined "
            r"the payload to coop\.ThreadData\(float32, items=1, producer\)"
        ),
    ):
        _compile(kernel)


@requires_typed_pipeline
def test_a_plain_kernel_runs_no_rule():
    from cuda.coop.numba_mlir._pipeline._typing_rules import RULES

    @cuda.jit
    def plain(source, output):
        values = cuda.local.array(4, types.float32)
        for i in range(4):
            values[i] = source[i] * 2
        total = 0.0
        for v in values:
            total += v
        output[cuda.threadIdx.x] = total

    lookups, calls = RULES.lookups, RULES.calls
    output = np.zeros(32, np.float32)
    plain[1, 32](np.arange(4, dtype=np.float32), output)
    np.testing.assert_array_equal(output, 12)
    assert RULES.lookups > lookups
    assert RULES.calls == calls
