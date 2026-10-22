# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check Load result ownership and argument choices before compilation."""

from copy import copy
from importlib import import_module

import numpy as np
import pytest

pytest.importorskip("cutlass")

from cuda import coop
from cuda.coop import cutlass as cutlass_coop

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


@pytest.fixture(autouse=True)
def cutlass_trace(monkeypatch):
    for module in ("_dispatch", "load_store"):
        monkeypatch.setattr(
            import_module(f"cuda.coop._core.api.{module}"),
            "_backend_module_name",
            lambda: "cuda.coop.cutlass",
        )


@pytest.mark.parametrize(
    "api", (coop, cutlass_coop), ids=("common", "qualified")
)
@pytest.mark.parametrize("items_per_thread", (1, 4))
def test_load_result_ownership(monkeypatch, api, items_per_thread):
    """Fresh results remain independent; explicit output keeps its identity.

    The provider spy isolates facade behavior from memory access and code
    generation, which are covered by the runtime movement tests.
    """

    movement = import_module("cuda.coop.cutlass._group_load_store")
    lowering = import_module("cuda.coop.cutlass._lowering._load_store")
    calls = []

    def provider_load(**kwargs):
        output = kwargs["output"]
        calls.append(output)
        output.dtype = np.int32
        for item, value in enumerate(kwargs["source"]):
            output[item] = np.int32(value)

    monkeypatch.setattr(
        movement,
        "_resolve_group",
        lambda group, algorithm, temp_storage, operation: (
            group,
            None,
            algorithm,
        ),
    )
    monkeypatch.setattr(lowering, "provider_load", provider_load)
    source = list(range(items_per_thread))
    group = api.this_block()
    original = api.load(group, source, items_per_thread=items_per_thread)
    assert isinstance(original, cutlass_coop.ThreadData)
    assert original.items_per_thread == items_per_thread
    assert original.dtype is np.int32
    alias = original
    updated = [value + 10 for value in source]
    original = api.load(group, updated, items_per_thread=items_per_thread)
    assert original is not alias
    assert list(alias) == source
    assert list(original) == updated

    output = api.ThreadData(items_per_thread, alignment=64)
    output_alias = output
    assert api.load(group, source, output) is None
    assert output is output_alias
    assert output.dtype is np.int32
    assert output.alignment == 64
    assert list(output_alias) == source
    assert calls == [alias, original, output]

    if api is coop:
        for payload in (original, copy(original)):
            with pytest.raises(TypeError, match="dtypes"):
                payload[0] = True


@pytest.mark.parametrize(
    "api", (coop, cutlass_coop), ids=("common", "qualified")
)
def test_load_requires_one_extent_source(api):
    group = api.this_block()
    with pytest.raises(TypeError, match="output or items_per_thread"):
        api.load(group, object())
    with pytest.raises(TypeError, match="output or items_per_thread"):
        api.load(group, object(), api.ThreadData(1), items_per_thread=1)
    for count, error in (
        (True, TypeError),
        (1.5, TypeError),
        (0, ValueError),
        (-1, ValueError),
    ):
        with pytest.raises(error, match="items_per_thread"):
            api.load(group, object(), items_per_thread=count)
