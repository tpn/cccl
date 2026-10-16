# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Collect every Numba-CUDA-MLIR interface the typed pipeline uses.

The typed pipeline runs on Numba-CUDA-MLIR ``>=0.5.0,<0.6`` without changes
to the compiler. Some of what it needs is public (``numba_cuda_mlir.extending``
and ``numba_cuda_mlir.models``); the rest is reached through the vendored
``numba_cuda_mlir.numba_cuda`` package or through behaviour that is not
documented. Every such use is made here, and only here, so that the list of
private dependencies is the import list of this module, and each one names
the upstream change that would replace it.

=====================================  =======================================
Bridge                                 Replaced by
=====================================  =======================================
``register_rewrite("after-inference")``  a public post-typing hook
``attach_inline_wrapper``              ``register_inline_alias``, or letting
                                       a template refine its call's arguments
``force_literal_arg``                  a fixed ``bind_fold_arguments``
``rebind_targetoptions``               a public ``launch_bounds`` setter
``LTOIR`` (class name)                 link kinds detected by ``isinstance``
``types.undefined``, ``unliteral``     public re-exports
templates, models, IR utilities        public re-exports
MLIR dialects, lowering utilities      public re-exports
``external_abi_function_type``         a public external-call builder
``cast_value``                         a public value-conversion helper for
                                       lowering
patch: argument-sized                  U-P-1: ``cuda.local.array`` requests a
``cuda.local.array``                   literal for a non-literal shape
patch: tuple refinement                U-H1-4: refine hooks for
                                       ``static_getitem`` and ``build_tuple``
patch: phi refinement                  U-H1-3: refine every phi input; a
                                       constructor's refine hook does not
                                       raise
patch: typing rules                    U-R1 to U-R3: a typing-rule bucket in
(``_typing_rules.ENGINE``)             the registry; a gate before a call
                                       resolves; strata at the fixed point
=====================================  =======================================

Importing this module fails with ``TypedPipelineUnavailableError`` when any
of these interfaces is missing, before anything is registered, so that an
unsupported compiler fails closed instead of compiling kernels incorrectly.

Rows marked "patch" change Numba-CUDA-MLIR's own behaviour. They live in
``_ncm_patches``, are pinned to Numba-CUDA-MLIR 0.5.4 and to the source of
what they replace, and are applied only when the typed pipeline activates;
a mismatch fails closed in the same way.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


class TypedPipelineUnavailableError(ImportError):
    """The installed compiler lacks an interface the typed pipeline uses."""


try:
    from numba_cuda_mlir import cuda, extending
    from numba_cuda_mlir._mlir import ir as mlir_ir
    from numba_cuda_mlir._mlir.dialects import arith, func, gpu, llvm, memref
    from numba_cuda_mlir._mlir.extras import types as mlir_types
    from numba_cuda_mlir.extending import (
        lowering_registry,
        require_launch_config,
        typing_registry,
    )
    from numba_cuda_mlir.lowering_utilities import (
        get_or_insert_function,
        memref_data_pointer,
        user_signature_to_external_abi_signature,
    )
    from numba_cuda_mlir.models import (
        ArrayModel,
        PrimitiveModel,
        register_model,
    )
    from numba_cuda_mlir.numba_cuda import types
    from numba_cuda_mlir.numba_cuda.core import errors, ir
    from numba_cuda_mlir.numba_cuda.core.ir_utils import build_definitions
    from numba_cuda_mlir.numba_cuda.core.rewrites import (
        Rewrite,
        register_rewrite,
    )
    from numba_cuda_mlir.numba_cuda.cudadrv import linkable_code
    from numba_cuda_mlir.numba_cuda.typeconv import Conversion
    from numba_cuda_mlir.numba_cuda.types import undefined, unliteral
    from numba_cuda_mlir.numba_cuda.typing.templates import (
        AbstractTemplate,
        AttributeTemplate,
        bound_function,
        signature,
    )
    from numba_cuda_mlir.tools import resolve_gpu_target
except ImportError as exc:
    raise TypedPipelineUnavailableError(
        "The cuda.coop typed pipeline needs Numba-CUDA-MLIR >=0.5.0,<0.6; "
        f"the installed compiler is missing {exc.name or exc}."
    ) from exc


def register_after_inference_rewrite(rewrite: type) -> type:
    """Run ``rewrite`` on every function after typing, before lowering.

    The registry is process-wide and also runs on nested device compiles, so
    the rewrite must decide for itself which functions it applies to.
    """

    return register_rewrite("after-inference")(rewrite)


def attach_inline_wrapper(function: Callable, wrapper: Callable) -> None:
    """Make Numba inline ``wrapper`` wherever a kernel calls ``function``.

    Numba's inliner looks for ``targetoptions`` and ``py_func`` on the called
    object itself. Attaching them changes the function object at run time,
    not its source, and has no effect on host calls.
    """

    function.targetoptions = {"inline": "always"}
    function.py_func = wrapper


def force_literal_arg(
    indices: set[int],
    fold_arguments: Callable[[tuple, Mapping], tuple],
) -> Exception:
    """Return the exception that restarts typing with literal arguments.

    ``ForceLiteralArg.bind_fold_arguments`` imports a module that does not
    exist on 0.5.x, so the folding function is passed to the constructor.
    """

    return errors.ForceLiteralArg(indices, fold_arguments=fold_arguments)


def rebind_targetoptions(state: Any, **updates: Any) -> None:
    """Replace the function's target options with an updated copy.

    The dictionary is shared with later overloads compiled through the same
    dispatcher, so it is never changed in place.
    """

    state.metadata["targetoptions"] = {
        **state.metadata.get("targetoptions", {}),
        **updates,
    }


class LTOIR(linkable_code.LTOIR):
    """An LTO-IR link item whose bytes are produced on first use.

    Numba-CUDA-MLIR recognises LTO-IR items by their class name, so this
    subclass must be called ``LTOIR``.
    """

    def __init__(self, produce: Callable[[], bytes], name: str) -> None:
        super().__init__(None, name=name)
        self._produce = produce
        self._image: bytes | None = None

    @property
    def data(self) -> bytes:
        if self._image is None:
            self._image = self._produce()
        return self._image


def refresh_registries() -> None:
    """Make newly registered typing and lowering visible to the compiler."""

    extending.refresh_registries()


def external_abi_function_type(builder: Any, sig: Any) -> Any:
    """Return the MLIR function type of an external call with Numba's ABI.

    The lowering builder has no public way to declare an external function
    taking a return pointer followed by C-ABI operands.
    """

    abi_sig = user_signature_to_external_abi_signature(sig)
    return builder._external_abi_function_type(abi_sig, numba_abi=True)


def cast_value(builder: Any, value: Any, source: Any, target: Any) -> Any:
    """Convert an MLIR ``value`` of Numba type ``source`` to ``target``.

    This is the conversion Numba applies to a call argument, such as a
    narrowing or a sign extension.
    """

    return builder.lower_cast(source, target, value)


__all__ = [
    "LTOIR",
    "AbstractTemplate",
    "ArrayModel",
    "AttributeTemplate",
    "Conversion",
    "PrimitiveModel",
    "Rewrite",
    "TypedPipelineUnavailableError",
    "arith",
    "attach_inline_wrapper",
    "bound_function",
    "build_definitions",
    "cast_value",
    "cuda",
    "errors",
    "extending",
    "external_abi_function_type",
    "force_literal_arg",
    "func",
    "get_or_insert_function",
    "gpu",
    "ir",
    "llvm",
    "lowering_registry",
    "memref",
    "memref_data_pointer",
    "mlir_ir",
    "mlir_types",
    "rebind_targetoptions",
    "refresh_registries",
    "register_after_inference_rewrite",
    "register_model",
    "require_launch_config",
    "resolve_gpu_target",
    "signature",
    "types",
    "typing_registry",
    "undefined",
    "unliteral",
]
