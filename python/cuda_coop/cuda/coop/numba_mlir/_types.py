# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generate C++ wrappers that Numba kernels can call as device functions.

A provider is the implementation of a cooperative operation for a particular
configuration, such as a block load with a fixed dtype and tile size. An
``Algorithm`` describes that configuration and emits a C++ wrapper around the
CUB operation. Parameter descriptors are ``Parameter`` objects such as
``Value``, ``Pointer``, and ``Array``. They describe how arguments accepted by
Numba map to the wrapper's C++ signature, including pointers, arrays, scalar
controls, and values embedded at compile time. For example, ``Array`` describes
an array parameter's type and extent; ``ThreadData`` constructs the per-thread
payload passed to that parameter. Group planning separately uses
``ThreadGroup`` and ``ThreadHierarchy`` to describe which threads cooperate.
``TypeWrapper`` supplies matching C++ storage declarations for compiler types
without a builtin C++ spelling.

Construction specializes an algorithm and returns its source description.
The typed pipeline combines provider wrappers into a shared translation unit
and coalesces equivalent providers, reducing repeated NVRTC compilation. It
queries scratch layouts with the provider image and links the resulting LTO
IR directly from memory.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from io import StringIO
from numbers import Integral
from typing import cast

import numba_cuda_mlir.numba_cuda.types as numba_types
from numba_cuda_mlir import cuda
from numba_cuda_mlir.descriptor import mlir_target
from numba_cuda_mlir.numba_cuda.typing.templates import signature

from cuda.coop._core import SynchronizationScope

from ._compiler import _nvrtc as nvrtc
from ._compiler._operations import StorageABI
from ._semantic import _numba_semantic_token

NUMBA_TYPES_TO_CPP = {
    numba_types.boolean: "bool",
    numba_types.int8: "::cuda::std::int8_t",
    numba_types.int16: "::cuda::std::int16_t",
    numba_types.int32: "::cuda::std::int32_t",
    numba_types.int64: "::cuda::std::int64_t",
    numba_types.uint8: "::cuda::std::uint8_t",
    numba_types.uint16: "::cuda::std::uint16_t",
    numba_types.uint32: "::cuda::std::uint32_t",
    numba_types.uint64: "::cuda::std::uint64_t",
    numba_types.float16: "__half",
    numba_types.float32: "float",
    numba_types.float64: "double",
}

_BlockThreads = int | tuple[int, ...] | list[int]

_SUPPORTED_LOGICAL_WARP_THREADS = frozenset({1, 2, 4, 8, 16, 32})


_DEVICE_LTOIR_CACHE: dict[
    tuple[object, tuple[int, int], str, tuple[tuple[str, object], ...]], bytes
] = {}
_KERNEL_COMPILE_TARGET: ContextVar[nvrtc.CompileTarget | None] = ContextVar(
    "cuda_coop_numba_mlir_kernel_compile_target", default=None
)


@contextmanager
def kernel_compile_target(
    target: nvrtc.CompileTarget | None,
) -> Iterator[None]:
    """Generate device code for the kernel being compiled in this context.

    The typed pipeline enters this once per compiler attempt with the
    kernel's resolved target. Group helpers, providers, and Python
    callbacks created inside then follow the kernel's ``chip`` and
    ``lineinfo`` options instead of the current device. ``None`` selects the
    current device. Exiting restores the previous target, so a nested kernel
    compilation does not leak its target to its caller.
    """

    token = _KERNEL_COMPILE_TARGET.set(target)
    try:
        yield
    finally:
        _KERNEL_COMPILE_TARGET.reset(token)


def numba_type_to_cpp(numba_type):
    """Return the builtin C++ spelling or the opaque ``storage_t`` name."""
    cpp_type = NUMBA_TYPES_TO_CPP.get(numba_type)
    if cpp_type is not None:
        return cpp_type
    return "storage_t"


def _validate_logical_warp_threads(logical_warp_threads, *, power_of_two=True):
    """Check the logical warp width before compiling or allocating scratch.

    Warp factories and provider code generation use this to require a Python
    integer from the supported widths. WarpReduce also accepts widths that
    are not powers of two; its provider explicitly opts into that CUB range.
    ``logical_warp_threads`` counts participating lanes per logical warp.
    Return it unchanged, or raise ``ValueError`` for an invalid width or type.
    """

    supported_widths = (
        _SUPPORTED_LOGICAL_WARP_THREADS if power_of_two else range(1, 33)
    )
    if (
        isinstance(logical_warp_threads, bool)
        or not isinstance(logical_warp_threads, int)
        or logical_warp_threads not in supported_widths
    ):
        supported = ", ".join(str(value) for value in sorted(supported_widths))
        raise ValueError(
            "warp-scoped providers require a logical width in "
            f"{{{supported}}}; got {logical_warp_threads!r}"
        )
    return logical_warp_threads


def _symbol_component(value):
    component = re.sub(r"\W+", "_", str(value)).strip("_")
    return component or "anon"


def _hash_symbol_value(hasher, value, depth=0):
    del depth
    hasher.update(
        repr(_numba_semantic_token(value)).encode(
            "utf-8", errors="backslashreplace"
        )
    )


def _callable_symbol_component(fn: Callable) -> str:
    """Combine a readable callable name with its semantic digest.

    Use the underlying Python function for a dispatcher. Module and qualified
    name aid inspection; the digest distinguishes functions with different
    code or captured values even when their names match.
    """

    py_func = getattr(fn, "py_func", fn)
    module = getattr(py_func, "__module__", "unknown")
    qualname = getattr(
        py_func,
        "__qualname__",
        getattr(py_func, "__name__", type(py_func).__name__),
    )
    hasher = hashlib.sha1()
    _hash_symbol_value(hasher, py_func)
    digest = hasher.hexdigest()[:20]
    return _symbol_component(f"{module}_{qualname}_{digest}")


def _python_operator_symbol_name(
    binary_op: Callable,
    ret_dtype: numba_types.Type,
    arg_dtypes: Sequence[numba_types.Type],
    *,
    state_dtype: numba_types.Type | None = None,
) -> str:
    """Name a compiled Python operator from its callable and signature.

    Include the callable's semantic identity and the result and argument
    dtypes. A stateful operator also includes its state dtype, so callbacks
    with different state-pointer signatures cannot share a symbol. Runtime
    state contents and diagnostic labels do not affect this name.
    """

    callable_component = _callable_symbol_component(binary_op)
    if state_dtype is not None:
        callable_component = (
            f"stateful_{_symbol_component(state_dtype)}_{callable_component}"
        )
    signature_component = f"{_symbol_component(ret_dtype)}__" + "_".join(
        _symbol_component(dtype) for dtype in arg_dtypes
    )
    return f"cuda_coop_numba_mlir_F{callable_component}_{signature_component}"


def _normalize_compute_capability(compute_capability) -> tuple[int, int]:
    """Validate a target as a positive major and single-digit minor pair.

    Reject booleans, malformed pairs, and invalid components before a target
    is used in callback cache keys or encoded as a numeric architecture
    identifier. Return plain integers for consistent identity.
    """

    if (
        not isinstance(compute_capability, (tuple, list))
        or len(compute_capability) != 2
    ):
        raise RuntimeError(
            "cuda.coop.numba_mlir requires a "
            "two-component CUDA compute capability"
        )
    major, minor = compute_capability
    if (
        isinstance(major, bool)
        or isinstance(minor, bool)
        or not isinstance(major, Integral)
        or not isinstance(minor, Integral)
        or major < 1
        or minor < 0
        or int(minor) > 9
    ):
        raise RuntimeError(
            "cuda.coop.numba_mlir received an invalid CUDA compute capability "
            f"{compute_capability!r}"
        )
    return int(major), int(minor)


def _current_compile_target() -> nvrtc.CompileTarget:
    """Return the target for generated device code.

    During a kernel compilation this is the kernel's own target, set by
    ``kernel_compile_target``. Outside one, such as when a provider is
    specialized directly, fall back to the current device without line
    information.
    """

    target = _KERNEL_COMPILE_TARGET.get()
    if target is not None:
        return target
    # The CUDA module reexports this accessor but omits it from its stub.
    device = (
        cuda.get_current_device()  # pyright: ignore[reportAttributeAccessIssue]
    )
    return nvrtc.CompileTarget(
        cc=_compute_capability_number(
            _normalize_compute_capability(device.compute_capability)
        )
    )


def _current_compute_capability() -> tuple[int, int]:
    """Return the exact target used for callback device compilation."""

    return _normalize_compute_capability(
        divmod(_current_compile_target().cc, 10)
    )


def _compute_capability_number(compute_capability: tuple[int, int]) -> int:
    major, minor = compute_capability
    return major * 10 + minor


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _lto_ir_digest(lto_ir):
    if isinstance(lto_ir, str):
        data = lto_ir.encode("utf-8")
    else:
        data = bytes(lto_ir)
    return hashlib.sha1(data).hexdigest()


def _struct_size_alignment(
    member_types: Iterable[numba_types.Type],
) -> tuple[int, int]:
    """Lay out ordered members and return their byte size and alignment.

    Each field starts at its required alignment. Tail padding lets an array of
    these structs keep that alignment for every element.
    """
    offset = 0
    max_align = 1
    for member_type in member_types:
        member_size, member_align = _size_alignment_from_numba_type(member_type)
        offset = _align_up(offset, member_align)
        offset += member_size
        max_align = max(max_align, member_align)
    return _align_up(offset, max_align), max_align


def _registered_struct_member_types(
    numba_type: numba_types.Type,
) -> tuple[numba_types.Type, ...] | None:
    """Find a struct layout whose CUDA and MLIR member types agree.

    The storage-layout fallback must describe the same value on both sides of
    the compiler boundary. Require a CUDA ``StructModel`` and an MLIR
    ``StructModel`` with identical ordered member types before using their
    members to compute size and alignment. This compares member descriptions,
    not independently measured byte layouts.

    MLIR model lookup can require an active context and location. Supply only
    those that are missing, leaving an existing caller context in place.

    Parameters
    ----------
    numba_type : numba_types.Type
        Compiler type to look up in the CUDA/default and MLIR data managers.

    Returns
    -------
    tuple of numba_types.Type or None
        Ordered members when both models agree, including an empty tuple for
        matching empty models. ``None`` means either lookup has no supported
        struct model or the member sequences differ.
    """

    from numba_cuda_mlir import models as mlir_models

    # These classes are provided by MLIR's compiled extension without stubs.
    from numba_cuda_mlir._mlir.ir import (
        Context,  # pyright: ignore[reportAttributeAccessIssue]
        Location,  # pyright: ignore[reportAttributeAccessIssue]
    )
    from numba_cuda_mlir.numba_cuda.datamodel.models import (
        StructModel as CudaStructModel,
    )
    from numba_cuda_mlir.numba_cuda.datamodel.registry import default_manager
    from numba_cuda_mlir.numba_cuda.models import cuda_data_manager

    def get_member_types(manager, model_type):
        try:
            model = manager.lookup(numba_type)
        except (KeyError, NotImplementedError, TypeError, ValueError):
            return None
        if not isinstance(model, model_type):
            return None
        return tuple(
            model.get_type(index) for index in range(model.field_count)
        )

    cuda_manager = cuda_data_manager.chain(default_manager)
    cuda_members = get_member_types(cuda_manager, CudaStructModel)
    if cuda_members is None:
        return None

    try:
        current_context = Context.current
    except ValueError:
        current_context = None

    if current_context is None:
        with Context(), Location.unknown():
            mlir_members = get_member_types(
                mlir_models.mlir_data_manager,
                mlir_models.StructModel,
            )
    else:
        try:
            current_location = Location.current
        except ValueError:
            current_location = None
        if current_location is None:
            with Location.unknown():
                mlir_members = get_member_types(
                    mlir_models.mlir_data_manager,
                    mlir_models.StructModel,
                )
        else:
            mlir_members = get_member_types(
                mlir_models.mlir_data_manager,
                mlir_models.StructModel,
            )

    if mlir_members != cuda_members:
        return None
    return cuda_members


def _size_alignment_from_numba_type(
    numba_type: numba_types.Type,
) -> tuple[int, int]:
    """Compute a supported value layout when LLVM ABI inspection is unavailable.

    ``TypeWrapper`` needs byte size and alignment to generate the opaque C++
    ``storage_t`` used for non-builtin types. Recursively lay out scalar values,
    uniform tuples, native aggregates, and matching registered CUDA/MLIR struct
    models. Struct fields receive alignment padding and the final size is
    rounded to the largest member alignment. Bitfield aggregates use their
    compiler-provided storage type instead of laying out individual bitfields.

    This is a restricted fallback, not a general data-model size estimator.
    Unknown types and registered structs with no inspectable members fail rather
    than receiving a guessed CUB storage layout.

    Parameters
    ----------
    numba_type : numba_types.Type
        Type whose by-value representation must be shared with generated C++.

    Returns
    -------
    tuple of int
        ``(size, alignment)`` in bytes.

    Raises
    ------
    TypeError
        The type or one of its members has no supported layout description.
    """

    from numba_cuda_mlir.type_defs.aggregate_types import AggregateType

    if isinstance(
        numba_type, (numba_types.Boolean, numba_types.BooleanLiteral)
    ):
        return 1, 1
    if isinstance(
        numba_type, (numba_types.Integer, numba_types.IntegerLiteral)
    ):
        size = max(1, numba_type.bitwidth // 8)
        return size, size
    if isinstance(numba_type, numba_types.Float):
        size = max(1, numba_type.bitwidth // 8)
        return size, size
    if isinstance(numba_type, numba_types.Complex):
        elem_size = max(1, numba_type.underlying_float.bitwidth // 8)
        return 2 * elem_size, elem_size
    if isinstance(numba_type, numba_types.UniTuple):
        elem_size, elem_align = _size_alignment_from_numba_type(
            numba_type.dtype
        )
        return elem_size * numba_type.count, elem_align
    if isinstance(numba_type, AggregateType):
        if numba_type.is_bitfield_struct:
            return _size_alignment_from_numba_type(
                numba_type.get_bitfield_storage_type()
            )
        return _struct_size_alignment(
            field_type
            for field_name, field_type, _bit_width in numba_type.fields
            if field_name is not None
        )

    struct_member_types = _registered_struct_member_types(numba_type)
    if struct_member_types:
        return _struct_size_alignment(struct_member_types)

    dtype_name = getattr(numba_type, "name", str(numba_type))
    raise TypeError(
        "cuda.coop.numba_mlir cannot safely materialize CUB storage for dtype "
        f"{dtype_name}: exact ABI size and alignment are unavailable; use a "
        "compiler-native dtype, Numba-CUDA-MLIR AggregateType, or matching "
        "registered CUDA and MLIR StructModels with inspectable member types"
    )


def _compile_device_ltoir(
    fn: Callable,
    *,
    sig,
    abi_info: dict[str, object],
    compute_capability: tuple[int, int],
    semantic_identity=None,
) -> bytes:
    """Compile and cache a callable for one device signature and target.

    Operator specialization calls this once its type dependencies are
    concrete. The resulting callback image will be linked with the C++
    provider, so compilation must use the same signature and device
    target as the declaration emitted into that provider.

    Cache identity includes callable semantics, compute capability, concrete
    signature, and ABI options. ``semantic_identity`` can describe an adapter
    in terms of the original callback and its pointer/value choices.

    Compile the underlying Python function rather than an existing dispatcher.
    The compiler can change dispatcher options during compilation, so this
    lets one dispatcher take part in several cooperative specializations.
    Return LTO-IR bytes and cache only successful results.

    Parameters
    ----------
    fn : callable
        Device-compilable Python function or dispatcher, possibly an
        ABI adapter generated for an aggregate callback.
    sig : numba signature
        Concrete return and argument types after pointer/value ABI
        adaptation.
    abi_info : dict of str to object
        Compiler ABI options, including the device symbol in
        ``abi_name``. These options participate in cache identity.
    compute_capability : tuple of int
        Exact target as ``(major, minor)``. The generated image is
        later linked only with a provider targeting the same
        capability.
    semantic_identity : object or None, optional
        Stable description of the original callback and any ABI
        adaptation. ``None`` uses the underlying Python function
        itself.
    """

    compute_capability = _normalize_compute_capability(compute_capability)
    # numba-cuda-mlir's compile() mutates dispatcher target options when passed
    # an existing dispatcher. Compile the underlying Python function so one
    # dispatcher can safely participate in multiple cooperative specializations.
    py_func = getattr(fn, "py_func", fn)
    identity = py_func if semantic_identity is None else semantic_identity
    cache_key = (
        _numba_semantic_token(identity),
        compute_capability,
        repr(sig),
        tuple(
            sorted(
                (name, _numba_semantic_token(value))
                for name, value in abi_info.items()
            )
        ),
    )
    cached = _DEVICE_LTOIR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Numba-CUDA-MLIR exports the compiler function without declaring its stub.
    ltoir, _ = cuda.compile(  # pyright: ignore[reportAttributeAccessIssue]
        py_func,
        sig=sig,
        output="ltoir",
        abi_info=abi_info,
        forceinline=True,
        cc=compute_capability,
    )
    ltoir_blob = (
        ltoir.encode("utf-8") if isinstance(ltoir, str) else bytes(ltoir)
    )
    _DEVICE_LTOIR_CACHE[cache_key] = ltoir_blob
    return ltoir_blob


def _adapt_python_operator_abi(
    py_func: Callable,
    *,
    return_by_pointer: bool,
    arguments_by_pointer: tuple[bool, ...],
) -> Callable:
    """Adapt value-level callback code to pointer-based aggregate operands.

    ``DependentPythonOperator.specialize`` uses this adapter before
    compiling the callback. The generated C++ wrapper passes aggregate
    storage by address, while Python callback authors still write
    ordinary value arguments and a returned value.

    Primitive arguments and returns stay by value. An aggregate argument is
    read through its pointer; an aggregate result is written through a final
    output pointer. Return the original function when no adaptation is needed.

    The wrapper calls an inline device version of the original function and
    supports one or two input values. It changes the native calling convention
    without changing the value-level callback signature seen by its author.

    Parameters
    ----------
    py_func : callable
        Original value-level callback, already unwrapped from an
        outer dispatcher.
    return_by_pointer : bool
        Whether the native caller supplies a final output pointer in
        place of a returned aggregate value.
    arguments_by_pointer : tuple of bool
        One flag per input, in callback argument order. True
        dereferences that input pointer before calling the original
        function.
    """

    if not return_by_pointer and not any(arguments_by_pointer):
        return py_func
    if len(arguments_by_pointer) not in {1, 2}:
        raise TypeError(
            "cuda.coop.numba_mlir aggregate Python operators must accept one "
            "or two value arguments"
        )

    device_op = cuda.jit(device=True, inline="always")(py_func)
    first_by_pointer = arguments_by_pointer[0]
    if len(arguments_by_pointer) == 1:
        if return_by_pointer:

            def adapted(first, result):
                first_value = first[0] if first_by_pointer else first
                result[0] = device_op(first_value)

        else:

            def adapted(first):
                first_value = first[0] if first_by_pointer else first
                return device_op(first_value)

        return adapted

    second_by_pointer = arguments_by_pointer[1]
    if return_by_pointer:

        def adapted(first, second, result):
            first_value = first[0] if first_by_pointer else first
            second_value = second[0] if second_by_pointer else second
            result[0] = device_op(first_value, second_value)

    else:

        def adapted(first, second):
            first_value = first[0] if first_by_pointer else first
            second_value = second[0] if second_by_pointer else second
            return device_op(first_value, second_value)

    return adapted


def _ltoir_to_ptx(ltoir: bytes, *, name: str, cc: int) -> str:
    """Link one LTO-IR image to PTX for explicit artifact inspection.

    Compile tests use this to inspect generated instructions and storage.
    Provider creation obtains layouts directly from NVRTC and does not call
    this helper. The link uses ``cc`` as its target and ``name`` to identify
    the input image; it does not change the provider's retained artifacts.
    """

    # cuda-core chooses cu12/cu13 exports dynamically, outside its stubs.
    from cuda.core import (
        Linker,  # pyright: ignore[reportAttributeAccessIssue]
        LinkerOptions,  # pyright: ignore[reportAttributeAccessIssue]
        ObjectCode,  # pyright: ignore[reportAttributeAccessIssue]
    )

    ltoir_obj = ObjectCode.from_ltoir(ltoir, name=name)
    linker_options = LinkerOptions(
        arch=f"sm_{cc}",
        link_time_optimization=True,
        ptx=True,
    )
    linked_ptx = Linker(ltoir_obj, options=linker_options).link("ptx")
    return linked_ptx.code.decode("utf-8")


class TypeWrapper:
    """Supply an aligned C++ storage declaration for a compiler value type."""

    def __init__(self, numba_type: numba_types.Type) -> None:
        """Build the C++ storage declaration for a compiler value type.

        Builtin types already have C++ spellings and need no declaration. Other
        types use an aligned byte-array ``storage_t`` with the compiler's ABI
        size and alignment; no field accessors or value conversion are
        generated. First query the target context's LLVM layout. If that
        inspection fails, use the restricted native/registered-model fallback
        rather than guessing.

        Parameters
        ----------
        numba_type : numba_types.Type
            Type that will appear in a provider template or parameter.

        Raises
        ------
        TypeError
            LLVM layout inspection failed and the fallback cannot establish the
            type's size and alignment.

        Notes
        -----
        The instance stores the declaration in ``code`` and initializes
        ``lto_irs`` to an empty list. Constructing this wrapper does not compile
        any source.
        """

        self.lto_irs = []

        if numba_type in NUMBA_TYPES_TO_CPP:
            self.code = ""
            return

        # TypeWrapper needs LLVM ABI layout info (size/alignment) to materialize
        # the storage_t wrapper for user-defined dtypes.
        try:
            abi_context = mlir_target.target_context
            value_type = abi_context.get_value_type(numba_type)
            size = value_type.get_abi_size(abi_context.target_data)
            alignment = value_type.get_abi_alignment(abi_context.target_data)
        except Exception:  # noqa: BLE001 - unsupported LLVM layouts use the validated fallback.
            size, alignment = _size_alignment_from_numba_type(numba_type)
        buf = StringIO()
        w = buf.write
        w(f"struct __align__({alignment}) storage_t\n")
        w("{\n")
        w(f"    char data[{size}];\n")
        w("};\n")

        self.code = buf.getvalue()


def numba_type_to_wrapper(numba_type: numba_types.Type) -> TypeWrapper:
    return TypeWrapper(numba_type)


_CPP_PARAMETER_RESERVED_NAMES = frozenset(
    [
        "alignas",
        "alignof",
        "and",
        "and_eq",
        "asm",
        "atomic_cancel",
        "atomic_commit",
        "atomic_noexcept",
        "auto",
        "bitand",
        "bitor",
        "bool",
        "break",
        "case",
        "catch",
        "char",
        "char8_t",
        "char16_t",
        "char32_t",
        "class",
        "compl",
        "concept",
        "const",
        "consteval",
        "constexpr",
        "constinit",
        "const_cast",
        "continue",
        "co_await",
        "co_return",
        "co_yield",
        "decltype",
        "default",
        "delete",
        "do",
        "double",
        "dynamic_cast",
        "else",
        "enum",
        "explicit",
        "export",
        "extern",
        "false",
        "float",
        "for",
        "friend",
        "goto",
        "if",
        "inline",
        "int",
        "long",
        "mutable",
        "namespace",
        "new",
        "noexcept",
        "not",
        "not_eq",
        "nullptr",
        "operator",
        "or",
        "or_eq",
        "private",
        "protected",
        "public",
        "reflexpr",
        "register",
        "reinterpret_cast",
        "requires",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "static_assert",
        "static_cast",
        "struct",
        "switch",
        "synchronized",
        "template",
        "this",
        "thread_local",
        "throw",
        "true",
        "try",
        "typedef",
        "typeid",
        "typename",
        "union",
        "unsigned",
        "using",
        "virtual",
        "void",
        "volatile",
        "wchar_t",
        "while",
        "xor",
        "xor_eq",
        "threadIdx",
        "blockDim",
        "storage_t",
    ]
)


def _cpp_parameter_names(parameters, *, reserved=()):
    """Choose readable identifiers without shadowing generated C++ locals."""

    used = set(_CPP_PARAMETER_RESERVED_NAMES).union(reserved)
    names = []
    for pid, parameter in enumerate(parameters):
        raw_name = getattr(parameter, "parameter_name", None)
        name = (
            re.sub(r"[^a-zA-Z0-9_]", "_", raw_name).rstrip("_")
            if isinstance(raw_name, str)
            else ""
        )
        if not name or not name[0].isalpha() or "__" in name:
            name = f"param_{pid}"
        stem = name
        suffix = 0
        while True:
            # Source and ABI wrappers derive these locals from the formal name.
            identifiers = {
                name,
                f"checked_{name}",
                f"transformed_{name}",
                f"{name}_state",
                f"cast_{name}",
                f"ref_{name}",
            }
            if not used.intersection(identifiers):
                break
            suffix += 1
            name = f"{stem}_{suffix}"
        used.update(identifiers)
        names.append(name)
    return names


class Parameter:
    """Describe one argument at the Python-to-C++ call boundary.

    Subclasses define the compiler dtype and C++ representation. An output is
    supplied by the wrapper and becomes the call result, so it consumes no
    Python input argument. Concrete descriptors return themselves from
    ``specialize``; dependent descriptors resolve template arguments first.
    """

    # The formal parameter name is separate from an operator's linker symbol.
    parameter_name: str | None = None

    def __init__(self, is_output=False):
        self.is_output = is_output

    def __repr__(self) -> str:
        return f"Parameter(out={self.is_output})"

    def specialize(self, template_arguments):
        return self

    def dtype(self):
        raise NotImplementedError("parameter must be specialized before typing")


class Value(Parameter):
    """Pass a scalar by value using the descriptor's compiler dtype."""

    def __init__(self, value_type, is_output=False):
        self.value_type = value_type
        super().__init__(is_output)

    def __repr__(self) -> str:
        return f"Value(dtype={self.value_type}, out={self.is_output})"

    def dtype(self):
        return self.value_type

    @property
    def provider_dtype(self):
        """Return the scalar dtype received by the underlying provider."""

        return self.value_type

    def cpp_decl(self, name):
        return numba_type_to_cpp(self.value_type) + " " + name

    def mangled_name(self):
        return f"{self.value_type}"


class ExactValue(Value):
    """A scalar whose compiler dtype must exactly match the provider ABI."""

    def __repr__(self) -> str:
        return f"ExactValue(dtype={self.value_type}, out={self.is_output})"

    def mangled_name(self):
        return f"Exact{self.value_type}"


class BoundedInteger(Value):
    """A signed 64-bit integer checked before provider-type narrowing."""

    def __init__(self, provider_dtype, *, minimum, maximum):
        if not isinstance(provider_dtype, numba_types.Integer) or isinstance(
            provider_dtype, numba_types.IntegerLiteral
        ):
            raise TypeError(
                "provider_dtype must be a non-literal integer dtype"
            )
        for name, bound in (("minimum", minimum), ("maximum", maximum)):
            if not isinstance(bound, Integral) or isinstance(bound, bool):
                raise TypeError(f"{name} must be an integer")
        minimum = int(minimum)
        maximum = int(maximum)
        if minimum > maximum:
            raise ValueError("minimum must not exceed maximum")

        if provider_dtype.signed:
            provider_minimum = -(1 << (provider_dtype.bitwidth - 1))
            provider_maximum = (1 << (provider_dtype.bitwidth - 1)) - 1
        else:
            provider_minimum = 0
            provider_maximum = (1 << provider_dtype.bitwidth) - 1
        if not provider_minimum <= minimum <= maximum <= provider_maximum:
            raise ValueError(
                "bounded integer limits must fit the provider integer dtype"
            )
        if not -(1 << 63) <= minimum <= maximum <= (1 << 63) - 1:
            raise ValueError(
                "bounded integer limits must fit the signed 64-bit ABI"
            )

        self._provider_dtype = provider_dtype
        self.minimum = minimum
        self.maximum = maximum
        super().__init__(numba_types.int64)

    def __repr__(self) -> str:
        return (
            "BoundedInteger("
            f"provider_dtype={self.provider_dtype}, minimum={self.minimum}, "
            f"maximum={self.maximum})"
        )

    @property
    def provider_dtype(self):
        return self._provider_dtype

    def mangled_name(self):
        bounds = internal_mangle_cpp(f"{self.minimum}_{self.maximum}")
        return f"Bounded{self.value_type}To{self.provider_dtype}_{bounds}"


class PointerOffset(Value):
    """Adjust an earlier pointer argument by an element count.

    ``pointer_arg_index`` selects the earlier provider argument. A static
    value is embedded in C++ source; otherwise the caller supplies the offset
    at runtime. The offset itself is not an argument to the CUB method.
    """

    def __init__(self, value_type, pointer_arg_index=0, static_value=None):
        if static_value is not None and (
            not isinstance(static_value, Integral)
            or isinstance(static_value, bool)
        ):
            raise TypeError("static pointer offset must be an integer")
        self.pointer_arg_index = pointer_arg_index
        self.static_value = None if static_value is None else int(static_value)
        super().__init__(value_type)

    def __repr__(self) -> str:
        return (
            f"PointerOffset(dtype={self.value_type}, "
            f"pointer_arg_index={self.pointer_arg_index}, "
            f"static_value={self.static_value})"
        )

    def mangled_name(self):
        static = (
            "Runtime"
            if self.static_value is None
            else f"Static{internal_mangle_cpp(str(self.static_value))}"
        )
        return f"Offset{self.value_type}{static}"


class Pointer(Parameter):
    """Pass contiguous one-dimensional array storage as a typed C++ pointer.

    Typing requires a C-contiguous array because the wrapper passes only the
    address; it cannot retain an arbitrary array stride.
    """

    def __init__(self, value_dtype, is_output=False):
        self.value_dtype = value_dtype
        super().__init__(is_output)

    def __repr__(self) -> str:
        return f"Pointer(dtype={self.value_dtype}, out={self.is_output})"

    def cpp_decl(self, name):
        return numba_type_to_cpp(self.value_dtype) + "* " + name

    def dtype(self):
        # Provider wrappers receive a raw pointer and therefore cannot preserve
        # an arbitrary array stride. Require contiguous one-dimensional memory
        # so a sliced device view cannot silently produce adjacent-element
        # results.
        return numba_types.Array(self.value_dtype, 1, "C")

    def mangled_name(self):
        return f"P{self.value_dtype}"


class DependentPointer(Parameter):
    """Resolve a template dtype before constructing a pointer descriptor."""

    def __init__(self, value_dtype, is_output=False):
        self.value_dtype = value_dtype
        super().__init__(is_output)

    def __repr__(self) -> str:
        return f"DependentPointer(dep={self.value_dtype}, out={self.is_output})"

    def specialize(self, template_arguments):
        return Pointer(
            self.value_dtype.resolve(template_arguments), self.is_output
        )


class PointerReference(Pointer):
    """Pass an array pointer, then dereference it for the CUB method call."""

    def __init__(self, value_dtype, is_output=False):
        self.deref_on_call = True
        super().__init__(value_dtype, is_output)

    def __repr__(self) -> str:
        return (
            f"PointerReference(dtype={self.value_dtype}, out={self.is_output})"
        )


class DependentPointerReference(DependentPointer):
    """Resolve the dtype of a pointer that the CUB call will dereference."""

    def __init__(self, value_dtype, is_output=False):
        self.deref_on_call = True
        super().__init__(value_dtype, is_output)

    def __repr__(self) -> str:
        return (
            f"DependentPointerReference(dep="
            f"{self.value_dtype}, out={self.is_output})"
        )

    def specialize(self, template_arguments):
        return PointerReference(
            self.value_dtype.resolve(template_arguments),
            self.is_output,
        )


class Reference(Parameter):
    """Represent a C++ reference using a scalar at the external-call boundary.

    The ABI wrapper creates a local scalar and passes it by reference to the
    typed wrapper. An output reference becomes the external call's result.
    """

    def __init__(self, value_dtype, is_output=False):
        self.value_dtype = value_dtype
        super().__init__(is_output)

    def __repr__(self) -> str:
        return f"Reference(dtype={self.value_dtype}, out={self.is_output})"

    def cpp_decl(self, name):
        return numba_type_to_cpp(self.value_dtype) + "& " + name

    def dtype(self):
        return self.value_dtype

    def mangled_name(self):
        return f"R{self.value_dtype}"


class DependentReference(Parameter):
    """Resolve a template dtype before constructing a reference descriptor."""

    def __init__(self, value_dtype, is_output=False):
        self.value_dtype = value_dtype
        super().__init__(is_output)

    def __repr__(self) -> str:
        return (
            f"DependentReference(dep={self.value_dtype}, out={self.is_output})"
        )

    def specialize(self, template_arguments):
        return Reference(
            self.value_dtype.resolve(template_arguments), self.is_output
        )


class Array(Pointer):
    """Pass a fixed-size per-thread array to a CUB array-reference parameter.

    The external ABI receives a pointer. The typed wrapper casts that pointer
    to a C++ array reference with ``size`` elements.
    """

    def __init__(self, value_dtype, size, is_output=False):
        self.size = size
        super().__init__(value_dtype, is_output)

    def __repr__(self) -> str:
        return (
            f"Array(dtype={self.value_dtype}, size="
            f"{self.size}, out={self.is_output})"
        )

    def cpp_decl(self, name):
        return f"{numba_type_to_cpp(self.value_dtype)} (&{name})[{self.size}]"

    def dtype(self):
        return numba_types.Array(self.value_dtype, 1, "C")

    def mangled_name(self):
        return f"A{self.size}_{self.value_dtype}"


class TransformedArray(Array):
    """Array input converted elementwise before the native CUB call."""

    def __init__(
        self,
        source_dtype,
        target_dtype,
        size,
        cpp_expression,
    ):
        if "{value}" not in cpp_expression:
            raise ValueError("array input transform must reference {value}")
        self.target_dtype = target_dtype
        self.cpp_expression = cpp_expression
        super().__init__(source_dtype, size, is_output=False)

    def __repr__(self) -> str:
        return (
            "TransformedArray("
            f"source_dtype={self.value_dtype}, "
            f"target_dtype={self.target_dtype}, "
            f"size={self.size}, cpp_expression={self.cpp_expression!r})"
        )

    def mangled_name(self):
        expression_digest = hashlib.sha1(
            self.cpp_expression.encode("utf-8")
        ).hexdigest()[:12]
        return (
            f"TA{self.size}_{self.value_dtype}_to_{self.target_dtype}_"
            f"{expression_digest}"
        )


class SubstitutionFailure(Exception):
    """Signal that a missing template value excludes one overload."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class Dependency:
    """Look up a required template argument during specialization."""

    def __init__(self, dep):
        self.dep = dep

    def resolve(self, template_arguments):
        if self.dep not in template_arguments:
            raise SubstitutionFailure(
                f"Template argument {self.dep} not provided"
            )
        if template_arguments[self.dep] is None:
            raise SubstitutionFailure(f"Template argument {self.dep} is None")
        return template_arguments[self.dep]


class Constant:
    """Keep one value unchanged while dependent parameters are specialized."""

    def __init__(self, val):
        self.val = val

    def resolve(self, _):
        return self.val


class StatelessOperator(Parameter):
    """Hold a compiled Python callback for use as a C++ function object.

    ``name`` identifies its C device symbol and ``ltoir`` supplies the link
    image. The concrete C++ return and argument types determine the forward
    declaration and generated lambda. ``storage_t`` uses a pointer ABI for
    aggregate inputs and a final output pointer for an aggregate result.

    ``compute_capability`` records the callback's exact target so provider
    source generation can reject a mismatched link image. The operator is
    embedded in generated source and is not a kernel runtime argument.
    """

    def __init__(
        self,
        name,
        ret_cpp_type,
        arg_cpp_types,
        ltoir,
        *,
        compute_capability,
    ):
        super().__init__()
        self.name = name
        self.ret_cpp_type = ret_cpp_type
        self.arg_cpp_types = tuple(arg_cpp_types)
        self.ltoir = bytes(ltoir)
        self.compute_capability = _normalize_compute_capability(
            compute_capability
        )

    def __repr__(self) -> str:
        return f"StatelessOperator(name={self.name!r})"

    def mangled_name(self):
        return self.name

    def forward_decl(self):
        """Declare the compiled callback with its concrete C device ABI.

        Map aggregate storage to opaque pointers and append an output pointer
        when the aggregate return cannot be passed by value.
        """

        return_type = (
            "void" if self.ret_cpp_type == "storage_t" else self.ret_cpp_type
        )
        arg_decls = [
            "const void*" if arg == "storage_t" else arg
            for arg in self.arg_cpp_types
        ]
        if self.ret_cpp_type == "storage_t":
            arg_decls.append("void*")
        return (
            f'extern "C" __device__ {return_type} '
            f"{self.name}({', '.join(arg_decls)});"
        )

    def wrap_decl(self, name):
        """Render a C++ lambda that calls the callback through its device ABI.

        CUB receives values through const references. The lambda passes
        aggregate addresses, supplies temporary storage for an aggregate
        result, and returns the resulting value to CUB. Scalar arguments and
        results pass by value.
        """

        param_decls = []
        param_refs = []
        for index, arg_type in enumerate(self.arg_cpp_types):
            arg_name = f"wp_{index}"
            param_decls.append(f"const {arg_type}& {arg_name}")
            param_refs.append(
                f"&{arg_name}" if arg_type == "storage_t" else arg_name
            )

        param_decls_csv = ", ".join(param_decls)
        param_refs_csv = ", ".join(param_refs)
        buf = StringIO()
        w = buf.write
        w(f"auto {name} = []({param_decls_csv}) {{\n")
        if self.ret_cpp_type == "storage_t":
            w("    storage_t result;\n")
            call_args = ", ".join((*param_refs, "&result"))
            w(f"    {self.name}({call_args});\n")
            w("    return result;\n")
        else:
            w(f"    return {self.name}({param_refs_csv});\n")
        w("};\n")
        return buf.getvalue()


class StatefulOperator(Parameter):
    """Bind compiled callback code to a runtime state-array parameter.

    The callback is already compiled to LTO for one compute capability.
    Its C++ lambda captures the caller's state pointer and forwards it as
    the first argument to that device function. CUB receives the lambda;
    the provider's runtime ABI receives the state array, not a callable.

    Parameters
    ----------
    name : str
        Compiled device symbol used by declarations and callback calls.
    state_dtype : numba type
        Element type of the runtime state array, independent of payload type.
    ret_cpp_type : str
        C++ spelling of the callback result type.
    arg_cpp_types : sequence of str
        C++ spellings of callback arguments after the state pointer.
    ltoir : bytes-like
        Compiled callback image included when linking the provider.
    compute_capability : tuple of int
        Exact target of that image. Provider source generation checks that
        the callback and provider targets agree.

    Notes
    -----
    Scan planning requires a one-item state array. The Numba parameter type
    checks its dtype and contiguous one-dimensional layout; it cannot encode
    the one-item extent. Runtime state values are never specialization inputs.
    """

    def __init__(
        self,
        name,
        state_dtype,
        ret_cpp_type,
        arg_cpp_types,
        ltoir,
        *,
        compute_capability,
    ):
        super().__init__()
        self.name = name
        self.state_dtype = state_dtype
        self.ret_cpp_type = ret_cpp_type
        self.arg_cpp_types = tuple(arg_cpp_types)
        self.ltoir = bytes(ltoir)
        self.compute_capability = _normalize_compute_capability(
            compute_capability
        )

    def __repr__(self) -> str:
        return (
            f"StatefulOperator(name={self.name!r}"
            f", state_dtype={self.state_dtype})"
        )

    def mangled_name(self):
        return self.name

    def forward_decl(self):
        """Declare the compiled callback with its state pointer first.

        The provider treats state as an untyped byte pointer. Callback code
        uses the corresponding typed pointer. This declaration preserves the
        external ABI without exposing a state object to C++.
        """

        return_type = (
            "void" if self.ret_cpp_type == "storage_t" else self.ret_cpp_type
        )
        arg_decls = ["char *state"]
        arg_decls.extend(
            "const void*" if arg == "storage_t" else arg
            for arg in self.arg_cpp_types
        )
        if self.ret_cpp_type == "storage_t":
            arg_decls.append("void*")
        return (
            f'extern "C" __device__ {return_type} '
            f"{self.name}({', '.join(arg_decls)});"
        )

    def cpp_decl(self, name):
        """Declare the provider runtime state pointer."""

        return f"char *{name}_state"

    def dtype(self):
        """Require a contiguous one-dimensional array of the state dtype.

        Planning separately verifies the one-item extent. Numba's array type
        does not carry that extent into overload matching.
        """

        return numba_types.Array(self.state_dtype, 1, "C")

    def wrap_decl(self, name):
        """Emit the C++ functor that closes over the runtime state pointer.

        CUB calls this lambda with the scan aggregate. The lambda prepends the
        captured pointer when calling the compiled device symbol, so updates
        reach the caller's state array without passing state through CUB's
        callback signature.
        """

        param_decls = []
        param_refs = []
        for index, arg_type in enumerate(self.arg_cpp_types):
            arg_name = f"wp_{index}"
            param_decls.append(f"const {arg_type}& {arg_name}")
            param_refs.append(
                f"&{arg_name}" if arg_type == "storage_t" else arg_name
            )

        state_name = f"{name}_state"
        buf = StringIO()
        w = buf.write
        w(f"auto {name} = [{state_name}]({', '.join(param_decls)}) {{\n")
        if self.ret_cpp_type == "storage_t":
            w("    storage_t result;\n")
            call_args = ", ".join((state_name, *param_refs, "&result"))
            w(f"    {self.name}({call_args});\n")
            w("    return result;\n")
        else:
            call_args = ", ".join((state_name, *param_refs))
            w(f"    return {self.name}({call_args});\n")
        w("};\n")
        return buf.getvalue()


class DependentPythonOperator:
    """Describe a callback whose concrete types are not known yet.

    Return type, argument types, and callable are backend resolvables. During
    ``specialize``, template arguments resolve them into a concrete signature
    and a compiled ``StatelessOperator``. This separates shared-core operator
    selection from target-specific device compilation.
    """

    def __init__(self, ret_dtype, arg_dtypes, op):
        self.ret_dtype = ret_dtype
        self.arg_dtypes = tuple(arg_dtypes)
        self.op = op

    def specialize(self, template_arguments):
        """Resolve callback types and compile the matching device ABI.

        Algorithm specialization calls this while resolving its operator
        parameters, before the provider wrapper is compiled. The callback's
        compiled definition and C++ declaration must therefore be produced
        together.

        Resolve the callable and dtype dependencies from
        ``template_arguments``. Build pointer transforms for aggregate values,
        derive the stable symbol, and compile for the enclosing kernel's
        compute capability. Aggregate returns use void plus a final output
        pointer; scalar returns remain by value.

        Return a ``StatelessOperator`` carrying declarations, LTO IR, and
        target identity for later provider linkage. A non-callable resolution
        is an error.

        Parameters
        ----------
        template_arguments : dict of str to object
            Concrete algorithm bindings used to resolve the callback and
            its return/argument dtype dependencies, such as ``T``.
        """

        op = self.op.resolve(template_arguments)
        if not callable(op):
            raise TypeError("Python operator must be a stateless callable")

        ret_dtype = self.ret_dtype.resolve(template_arguments)
        ret_cpp_type = numba_type_to_cpp(ret_dtype)
        ret_numba_type = (
            numba_types.CPointer(ret_dtype)
            if ret_cpp_type == "storage_t"
            else ret_dtype
        )
        arg_dtypes = tuple(
            arg.resolve(template_arguments) for arg in self.arg_dtypes
        )
        arg_cpp_types = tuple(numba_type_to_cpp(dtype) for dtype in arg_dtypes)
        arg_numba_types = tuple(
            numba_types.CPointer(dtype) if cpp_type == "storage_t" else dtype
            for dtype, cpp_type in zip(arg_dtypes, arg_cpp_types)
        )

        operator_py_func = getattr(op, "py_func", op)
        return_by_pointer = ret_cpp_type == "storage_t"
        arguments_by_pointer = tuple(
            cpp_type == "storage_t" for cpp_type in arg_cpp_types
        )
        compile_op = _adapt_python_operator_abi(
            operator_py_func,
            return_by_pointer=return_by_pointer,
            arguments_by_pointer=arguments_by_pointer,
        )
        compile_identity = (
            "numba-cuda-mlir-stateless-python-operator-abi-v1",
            operator_py_func,
            return_by_pointer,
            arguments_by_pointer,
        )
        mangled_name = _python_operator_symbol_name(op, ret_dtype, arg_dtypes)
        compute_capability = _current_compute_capability()
        if return_by_pointer:
            operator_signature = signature(
                numba_types.void,
                *arg_numba_types,
                ret_numba_type,
            )
        else:
            operator_signature = signature(ret_numba_type, *arg_numba_types)
        ltoir = _compile_device_ltoir(
            compile_op,
            sig=operator_signature,
            abi_info={"abi_name": mangled_name},
            compute_capability=compute_capability,
            semantic_identity=compile_identity,
        )
        return StatelessOperator(
            mangled_name,
            ret_cpp_type,
            arg_cpp_types,
            ltoir,
            compute_capability=compute_capability,
        )


class DependentStatefulOperator:
    """Resolve a stateful callback's types before compiling its device code.

    State, result, argument, and callable dependencies are resolved against
    the provider's template arguments. Specialization produces a
    ``StatefulOperator`` with compiled LTO and a concrete state-pointer ABI.
    No state allocation or state values are stored in this descriptor.
    """

    def __init__(self, state_dtype, ret_dtype, arg_dtypes, op, *, name=None):
        self.state_dtype = state_dtype
        self.ret_dtype = ret_dtype
        self.arg_dtypes = tuple(arg_dtypes)
        self.op = op
        self.name = name

    def specialize(self, template_arguments):
        """Compile a typed callback for the enclosing kernel's target.

        Algorithm specialization calls this before compiling its Scan
        wrapper. The state pointer is a device operand at each call, so only
        its type and the callback's semantics belong in the compiled
        operator identity.

        Resolve the state and payload types, reject aggregate payloads, and
        compile a signature with the typed state pointer first. For a functor
        class, compile its unbound ``__call__`` method. The first parameter
        receives the pointer instead of a Python instance.

        Build the symbol and compilation identity from callable semantics and
        types, including the state dtype. Return a ``StatefulOperator`` with
        the LTO image and exact compute capability. Source generation checks
        that the provider target agrees. Wrapper compilation happens later.

        Parameters
        ----------
        template_arguments : dict of str to object
            Concrete algorithm bindings for the callback, state dtype,
            and payload argument/result types. Runtime state values are
            not part of these bindings.
        """

        state_dtype = self.state_dtype.resolve(template_arguments)
        ret_dtype = self.ret_dtype.resolve(template_arguments)
        arg_dtypes = tuple(
            dtype.resolve(template_arguments) for dtype in self.arg_dtypes
        )
        op = self.op.resolve(template_arguments)
        if not callable(op):
            raise TypeError("Stateful Python operator must be callable")

        ret_cpp_type = numba_type_to_cpp(ret_dtype)
        arg_cpp_types = tuple(numba_type_to_cpp(dtype) for dtype in arg_dtypes)
        if ret_cpp_type == "storage_t" or "storage_t" in arg_cpp_types:
            raise TypeError(
                "cuda.coop.numba_mlir stateful Python operators support "
                "numeric payload dtypes only"
            )

        operator_py_func = op.__call__ if isinstance(op, type) else op
        operator_py_func = getattr(
            operator_py_func, "py_func", operator_py_func
        )
        compute_capability = _current_compute_capability()
        mangled_name = _python_operator_symbol_name(
            operator_py_func,
            ret_dtype,
            arg_dtypes,
            state_dtype=state_dtype,
        )
        operator_signature = signature(
            ret_dtype,
            numba_types.CPointer(state_dtype),
            *arg_dtypes,
        )
        compile_identity = (
            "numba-cuda-mlir-stateful-python-operator-abi-v1",
            operator_py_func,
            _numba_semantic_token(state_dtype),
        )
        ltoir = _compile_device_ltoir(
            operator_py_func,
            sig=operator_signature,
            abi_info={"abi_name": mangled_name},
            compute_capability=compute_capability,
            semantic_identity=compile_identity,
        )
        return StatefulOperator(
            mangled_name,
            state_dtype,
            ret_cpp_type,
            arg_cpp_types,
            ltoir,
            compute_capability=compute_capability,
        )


class CxxFunction(Parameter):
    """Embed a C++ expression in the provider call without a runtime argument.

    The expression can name a functor or encode a compile-time scalar value.
    Its text contributes to the generated symbol so distinct expressions can
    select distinct wrappers.
    """

    def __init__(self, cpp, func_dtype):
        super().__init__()
        self.cpp = cpp
        self.func_dtype = func_dtype

    def __repr__(self) -> str:
        return f"CxxFunction(cpp={self.cpp})"

    def mangled_name(self):
        return f"cuda_coop_numba_mlir_F{internal_mangle_cpp(self.cpp)}"

    def dtype(self):
        return self.func_dtype


class DependentCxxOperator:
    """Hold a C++ functor template with one unresolved dtype placeholder.

    ``dep`` identifies a template argument and ``cpp`` names a functor using
    that bracketed type, such as ``<T>``. Specialization produces a
    value-initialized ``CxxFunction`` without compiling a Python callback.
    """

    def __init__(self, dep: Dependency, cpp: str):
        self.dep = dep
        self.cpp = cpp

    def specialize(self, template_arguments):
        """Resolve the bracketed dtype placeholder and build the functor.

        Require exactly one occurrence so malformed source cannot silently
        leave an unresolved type or replace an unrelated token. Append value
        construction after resolving the dtype to its C++ spelling.
        """

        dtype = self.dep.resolve(template_arguments)
        dtype_cpp = numba_type_to_cpp(dtype)
        source = f"<{self.dep.dep}>"
        target = f"<{dtype_cpp}>"
        match_count = self.cpp.count(source)
        if match_count != 1:
            raise ValueError(
                f"Expected exactly one {source!r} placeholder in C++ operator "
                f"{self.cpp!r}; found {match_count}."
            )
        cpp = self.cpp.replace(source, target, 1)
        return CxxFunction(cpp=f"{cpp}{{}}", func_dtype=dtype)


class DependentArray(Parameter):
    """Resolve both the dtype and length of a fixed-size array parameter."""

    def __init__(self, value_dtype, size, is_output=False):
        self.value_dtype = value_dtype
        self.size = size
        super().__init__(is_output)

    def __repr__(self) -> str:
        return f"DependentArray(dep={self.value_dtype}, out={self.is_output})"

    def specialize(self, template_arguments):
        return Array(
            self.value_dtype.resolve(template_arguments),
            self.size.resolve(template_arguments),
            self.is_output,
        )


class TemplateParameter:
    """Name a required argument in the CUB class template parameter list."""

    def __init__(self, name):
        self.name = name

    def __repr__(self) -> str:
        return f"{self.name}"


def internal_mangle_cpp(cpp_name: str) -> str:
    """Turn a C++ spelling into a readable fragment of a generated name.

    Replace every character other than an ASCII letter or digit with an
    underscore. Callers incorporate the result into private wrapper, type,
    and parameter names. Distinct spellings can produce the same fragment,
    and a leading digit remains a digit: this helper neither guarantees a
    valid standalone C identifier nor performs C++ ABI name mangling.
    ``_bind_private_symbol_namespace`` separately binds the provider identity.

    Parameters
    ----------
    cpp_name : str
        C++ spelling or static parameter text used in a generated name.

    Returns
    -------
    str
        Text of the same length containing only ASCII letters, digits, and
        underscores.
    """

    return re.sub(r"[^a-zA-Z0-9]", "_", cpp_name)


def mangle_symbol(name, template_parameters):
    return "_".join(
        [name]
        + [
            template_parameter.mangled_name()
            for template_parameter in template_parameters
        ]
    )


class Algorithm:
    """Generate and compile wrappers for one concrete CUB specialization.

    Construction resolves ``template_arguments`` and keeps only applicable
    parameter overloads. Storage and synchronization settings describe the
    wrapper contract. Compilation later binds a device target, qualifies
    private symbols, and records the compiled scratch size and alignment.
    """

    def __init__(
        self,
        struct_name,
        method_name,
        c_name,
        includes,
        template_parameters,
        parameters,
        *,
        template_arguments: Mapping[str, object],
        storage_abi,
        execution_scope,
        synchronization_scope,
        type_definitions=None,
        fake_return=False,
        output_by_reference=False,
        compile_context=None,
    ):
        """Resolve template arguments and initialize provider compile state.

        ``parameters`` lists the candidate method signatures. A missing
        dependent value excludes that signature. Every declared class-template
        parameter requires a value. ``storage_abi`` determines whether scratch
        is the first runtime argument. ``synchronization_scope`` is ``NONE``
        or equal to ``execution_scope``. The typed pipeline owns barriers
        and scratch allocation. Construction does not run NVRTC.
        """
        self.struct_name = struct_name
        self.method_name = method_name
        self.c_name = (
            c_name
            if c_name.startswith("cuda_coop_numba_mlir_")
            else f"cuda_coop_numba_mlir_{c_name}"
        )
        self.includes = includes
        self.template_parameters = template_parameters
        self.parameters = parameters
        self.storage_abi = StorageABI(storage_abi)
        self.execution_scope = SynchronizationScope(execution_scope)
        self.synchronization_scope = SynchronizationScope(synchronization_scope)
        if self.synchronization_scope not in {
            SynchronizationScope.NONE,
            self.execution_scope,
        }:
            raise ValueError(
                "synchronization_scope must be NONE or match execution_scope"
            )
        self.type_definitions = type_definitions
        self.fake_return = fake_return
        self.output_by_reference = output_by_reference
        self._compile_context = compile_context
        self.logical_warp_threads: int | None = None
        self.block_threads: _BlockThreads | None = None
        self._private_symbol_digest = None
        self._private_symbol_key = None
        self._provider_compile_identity: nvrtc.CompilerIdentity | None = None
        self._specialize(template_arguments)

    def __repr__(self) -> str:
        return (
            f"{self.struct_name}::"
            f"{self.method_name}{self.template_parameters}: {self.parameters}"
        )

    def _symbol_base_name(self):
        namespace = (
            ""
            if self._private_symbol_digest is None
            else f"_S{self._private_symbol_digest}"
        )
        return f"{self.c_name}{namespace}_{self.method_name}"

    def _current_provider_compile_identity(self) -> nvrtc.CompilerIdentity:
        """Describe compilation for the current target and this provider.

        ``_bind_provider_compile_identity`` calls this when its caller has
        not supplied an identity. Query the target even when this object has
        cached artifacts so switching targets cannot silently reuse them.
        Return the NVRTC identity for relocatable LTO output, including the
        resolved header/toolkit context and the target from
        ``_current_compile_target``.
        """

        return nvrtc.compiler_identity(
            context=self._resolved_compile_context(),
            rdc=True,
            code="lto",
            **_current_compile_target().nvrtc_kwargs(),
        )

    def _bind_provider_compile_identity(
        self, compile_identity: nvrtc.CompilerIdentity | None = None
    ) -> nvrtc.CompilerIdentity:
        """Bind compilation inputs before naming or reusing provider code.

        Symbol qualification and source generation call
        this so a specialization cannot reuse code for a different target or
        compiler setup. ``compile_identity`` is the record returned by
        ``nvrtc.compiler_identity``; ``None`` queries the current target and
        this provider's resolved context.

        Store and return the first identity. Later calls return the same
        identity or raise ``RuntimeError`` if the supplied or observed inputs
        differ. This check does not compile code or validate thread counts.

        Binding belongs to this ``Algorithm`` instance. Separate kernel
        compilations create fresh specializations; reusing this instance on
        another device is valid only with the same compute capability and
        compiler inputs.
        """
        observed = (
            self._current_provider_compile_identity()
            if compile_identity is None
            else compile_identity
        )
        existing = self._provider_compile_identity
        # For example, an Algorithm bound while targeting compute_90 cannot
        # later supply its artifacts to a compute_80 compilation. That compile
        # needs a fresh Algorithm. A different device with the same target and
        # compiler inputs is compatible; device ordinal is not part of identity.
        if existing is not None and existing != observed:
            raise RuntimeError(
                "Provider artifacts were already qualified for a different "
                "compute capability, output kind, or compiler-option set."
            )
        self._provider_compile_identity = observed
        return observed

    def _bind_private_symbol_namespace(
        self,
        *,
        logical_warp_threads: int | None = None,
        block_threads: _BlockThreads | None = None,
        compile_identity: nvrtc.CompilerIdentity | None = None,
    ) -> nvrtc.CompilerIdentity:
        """Bind this provider to a deterministic private symbol namespace.

        Source emission and typed call lowering must agree on exported names,
        including when several specializations share one translation unit. Hash
        the provider's coalescing key, which includes its ABI, thread
        configuration, compiler context, and compilation target. Equivalent
        providers therefore share names; providers with different interfaces or
        targets remain distinct.

        The first call stores the compile identity, key, and symbol suffix on
        this object. Later calls validate the current inputs and reuse that
        suffix without hashing again. Specialization, bundling, and source
        emission each call this because provider descriptors are mutable.
        Once names are bound, changes require a fresh specialization.
        When no identity is supplied, query the current CUDA device even if a
        previous call has already bound the provider.

        Parameters
        ----------
        logical_warp_threads : int, optional
            Logical warp width override used in the coalescing key. ``None``
            uses ``self.logical_warp_threads``.
        block_threads : int, tuple of int, or list of int, optional
            Enclosing block configuration override. ``None`` uses
            ``self.block_threads``.
        compile_identity : nvrtc.CompilerIdentity, optional
            Target and options returned by ``nvrtc.compiler_identity``. ``None``
            resolves them from this provider's compiler context and current
            device.

        Returns
        -------
        nvrtc.CompilerIdentity
            Bound compilation identity for subsequent source and artifact
            creation.

        Raises
        ------
        RuntimeError
            This object was already bound to a different compilation identity or
            coalescing key.
        """

        compile_identity = self._bind_provider_compile_identity(
            compile_identity
        )
        key = algo_coalesce_key(
            self,
            logical_warp_threads=logical_warp_threads,
            block_threads=block_threads,
        )
        if self._private_symbol_key is not None:
            # The target/options check above has already passed. Here we also
            # protect the meaning of names that callers may already reference:
            # changing this provider's warp width from 16 to 32, for example,
            # needs a fresh specialization with its own symbol namespace.
            if self._private_symbol_key != key:
                raise RuntimeError(
                    "Provider symbols were already qualified for a different "
                    "provider definition or thread configuration. "
                    "Create a new specialization for the changed inputs."
                )
            # Reuse the digest, but validate the mutable provider on every call.
            return compile_identity

        self._private_symbol_key = key
        self._private_symbol_digest = hashlib.sha1(
            repr(key).encode("utf-8")
        ).hexdigest()[:20]
        return compile_identity

    def _resolved_compile_context(self):
        """Resolve header and toolkit identity once for this provider."""
        if self._compile_context is None:
            self._compile_context = nvrtc.resolve_compile_context()
        return self._compile_context

    def mangled_name(self, parameters):
        return mangle_symbol(self._symbol_base_name(), parameters)

    def _specialize(self, template_arguments: Mapping[str, object]) -> None:
        """Resolve template dependencies during provider construction.

        A substitution failure removes its entire overload while other
        overloads remain available. Copy resolved descriptors before retaining
        parameter names so shared descriptors remain unchanged. Bind the C++
        struct, symbol prefix, and logical warp width before code generation.
        """

        # Every template parameter requires an argument.
        template_list = []
        for template_parameter in self.template_parameters:
            if template_parameter.name not in template_arguments:
                raise ValueError(
                    f"Template argument {template_parameter.name} not provided"
                )
            template_argument = template_arguments[template_parameter.name]
            if isinstance(template_argument, int):
                template_list.append(str(template_argument))
            elif isinstance(template_argument, str):
                template_list.append(template_argument)
            else:
                template_list.append(numba_type_to_cpp(template_argument))
        template_list = ", ".join(template_list)

        # '::cuda::std::int32_t, 32' -> __cuda__std__int32_t__32
        mangle = internal_mangle_cpp(template_list)

        specialized_parameters = []
        for method in self.parameters:
            specialized_signature = []
            try:
                for parameter in method:
                    specialized_parameter = copy(
                        parameter.specialize(template_arguments)
                    )
                    specialized_parameter.parameter_name = getattr(
                        parameter, "parameter_name", None
                    )
                    specialized_signature.append(specialized_parameter)
                specialized_parameters.append(specialized_signature)
            except SubstitutionFailure:
                pass  # Substitution failure is not an error

        if self.template_parameters:
            self.struct_name = f"{self.struct_name}<{template_list}>"
        self.c_name += mangle
        self.template_parameters = []
        self.parameters = specialized_parameters
        self.logical_warp_threads = cast(
            int | None,
            template_arguments.get(
                "LOGICAL_WARP_THREADS",
                template_arguments.get("VIRTUAL_WARP_THREADS"),
            ),
        )

    @staticmethod
    def _ignore_codegen_param(param):
        return isinstance(param, (CxxFunction, StatelessOperator)) or (
            isinstance(param, PointerOffset) and param.static_value is not None
        )

    def _emit_abi_wrapper(
        self,
        w: Callable[[str], object],
        method: Sequence[Parameter],
        exported_name: str,
        internal_name: str,
        temp_storage_type_name: str | None = "temp_storage_t",
        temp_storage_param_pid: int | None = 0,
    ) -> None:
        """Write the external-call ABI shim for one typed C++ wrapper.

        Direct MLIR lowering calls a symbol ending in ``__abi`` with a leading
        return slot and an integer status result. Convert untyped pointer
        arguments back to their C++ pointee types, materialize scalar references
        as local values, and route the single output through the return slot.
        Emit status zero after the call; the shim does not allocate scratch or
        add synchronization.

        C++ functors and static pointer offsets are already embedded in the
        internal wrapper, so neither appears in this runtime ABI. Stateful
        Python operators pass their state pointer as an input. Output parameters
        are removed from the input list and represented by ``__ret``.

        Parameters
        ----------
        w : callable
            Text writer receiving generated C++ fragments.
        method : sequence of Parameter
            Specialized parameters in internal-wrapper order, with at most one
            output.
        exported_name : str
            Symbol prefix; the emitted entry point appends ``__abi``.
        internal_name : str
            Typed C++ wrapper called by the shim.
        temp_storage_type_name : str or None, optional
            C++ scratch type used when casting the designated byte pointer.
        temp_storage_param_pid : int or None, optional
            Index of an explicit scratch pointer in ``method``. ``None``
            disables that special cast for storage-free wrappers.

        Returns
        -------
        None
            Generated source is appended through ``w``.

        Raises
        ------
        ValueError
            More than one runtime output parameter is present.
        """

        output_param = None
        user_params = []
        for pid, param in enumerate(method):
            if self._ignore_codegen_param(param):
                continue
            if param.is_output:
                if output_param is not None:
                    raise ValueError("Multiple output parameters not supported")
                output_param = (pid, param)
            else:
                user_params.append((pid, param))

        abi_param_decls = ["void *__ret"]
        body_lines = []
        call_args_by_pid = {}

        parameter_names = _cpp_parameter_names(
            [param for _, param in user_params],
            reserved={"__ret", "out", internal_name, temp_storage_type_name},
        )
        for (pid, param), name in zip(user_params, parameter_names):
            if isinstance(
                param,
                (Pointer, Array, PointerReference, StatefulOperator),
            ):
                abi_param_decls.append(f"void *{name}")
                cast_name = f"cast_{name}"
                if (
                    pid == temp_storage_param_pid
                    and isinstance(param, Pointer)
                    and param.value_dtype == numba_types.uint8
                ):
                    # Non-alloc wrappers receive raw temporary storage
                    # explicitly.
                    body_lines.append(
                        "    "
                        f"{temp_storage_type_name} *{cast_name} = "
                        f"reinterpret_cast<{temp_storage_type_name} *>({name});"
                    )
                elif isinstance(param, StatefulOperator):
                    body_lines.append(
                        f"    char *{cast_name} = "
                        f"reinterpret_cast<char *>({name});"
                    )
                else:
                    pointee_type = numba_type_to_cpp(param.value_dtype)
                    body_lines.append(
                        f"    {pointee_type} *{cast_name} = reinterpret_cast<"
                        f"{pointee_type} *>({name});"
                    )
                call_args_by_pid[pid] = cast_name
            elif isinstance(param, Reference):
                value_type = numba_type_to_cpp(param.value_dtype)
                abi_param_decls.append(f"{value_type} {name}")
                local_name = f"ref_{name}"
                body_lines.append(f"    {value_type} {local_name} = {name};")
                call_args_by_pid[pid] = local_name
            else:
                value_type = numba_type_to_cpp(param.dtype())
                abi_param_decls.append(f"{value_type} {name}")
                call_args_by_pid[pid] = name

        output_var = None
        output_cpp_type = None
        if output_param is not None:
            output_cpp_type = numba_type_to_cpp(output_param[1].dtype())
            output_var = "out"
            body_lines.append(f"    {output_cpp_type} {output_var};")
            call_args_by_pid[output_param[0]] = output_var

        call_args = [
            call_args_by_pid[pid]
            for pid, param in enumerate(method)
            if not self._ignore_codegen_param(param)
        ]
        args_csv = ", ".join(call_args)
        if args_csv:
            body_lines.append(f"    {internal_name}({args_csv});")
        else:
            body_lines.append(f"    {internal_name}();")

        if output_var is not None:
            body_lines.append(
                f"    *reinterpret_cast<{output_cpp_type}"
                f" *>(__ret) = {output_var};"
            )
        body_lines.append("    return 0;")

        abi_params_csv = ", ".join(abi_param_decls)
        w(
            f'extern "C" __device__ int {exported_name}'
            f"__abi({abi_params_csv}) {{\n"
        )
        for line in body_lines:
            w(f"{line}\n")
        w("}\n\n")

    def _collect_support_ltoirs_and_udf_declarations(self, *, provider_cc):
        """Collect link images and declarations required by the provider.

        Include type-definition images and both stateless and stateful Python
        callbacks. Require each callback's recorded target to match
        ``provider_cc`` before adding its declaration. Repeated device symbols
        must have the same declaration; conflicting signatures are rejected.

        Return deduplicated LTO images and declarations in first-seen order.
        The provider source and linker consume these together, so generated
        calls and their compiled definitions describe the same ABI.
        """

        lto_irs = []
        udf_declarations = OrderedDict()

        if self.type_definitions:
            for type_definition in self.type_definitions:
                lto_irs.extend(type_definition.lto_irs)

        for method in self.parameters:
            for param in method:
                if not isinstance(param, (StatelessOperator, StatefulOperator)):
                    continue
                callback_cc = _compute_capability_number(
                    param.compute_capability
                )
                if callback_cc != provider_cc:
                    major, minor = param.compute_capability
                    raise RuntimeError(
                        "Python operator LTO IR target "
                        "does not match its provider: "
                        f"callback {major}.{minor}, provider "
                        f"{provider_cc // 10}."
                        f"{provider_cc % 10}"
                    )
                declaration = param.forward_decl()
                previous = udf_declarations.setdefault(param.name, declaration)
                if previous != declaration:
                    raise RuntimeError(
                        "Python operators produced conflicting device symbols"
                    )
                lto_irs.append(param.ltoir)

        return _dedupe_ltoirs(lto_irs), udf_declarations

    def _source_code(
        self,
        logical_warp_threads: int | None = None,
        block_threads: _BlockThreads | None = None,
        *,
        compile_identity: nvrtc.CompilerIdentity | None = None,
    ) -> tuple[str, list[bytes], tuple[str, ...], OrderedDict[str, str]]:
        """Generate C++ wrappers and compile-time storage metadata.

        Emit a typed wrapper and an ``__abi`` shim for each concrete method.
        The typed wrapper adapts array pointers to CUB array references and
        applies input transforms. It folds offsets into earlier pointer
        arguments and traps on out-of-range ``BoundedInteger`` values before
        narrowing them. C++ functors and static offsets are embedded in source
        instead of passed as runtime arguments. Python callbacks use lambdas:
        stateless lambdas have no capture, while stateful lambdas capture a
        runtime state pointer and forward it to the compiled callback.

        For ``LEADING_POINTER`` storage, the wrapper takes explicit scratch.
        The typed pipeline owns allocation and reuse synchronization.
        Storage-free providers emit no scratch metadata.

        This method qualifies private symbols without running NVRTC. It
        returns scratch type names so artifact creation can query their C++
        size and alignment during provider compilation. Each name describes
        one ``TempStorage`` instance for storage analysis to lay out.

        Parameters
        ----------
        logical_warp_threads : int, optional
            Logical warp width for provider identity. ``None`` uses
            ``self.logical_warp_threads``.
        block_threads : int, tuple of int, or list of int, optional
            Exact enclosing block size or dimensions. ``None`` uses
            ``self.block_threads``.
        compile_identity : nvrtc.CompilerIdentity, optional
            Bound target/options identity. ``None`` resolves the current
            device's identity through ``_bind_private_symbol_namespace``.

        Returns
        -------
        src : str
            Complete CUDA C++ translation unit for this specialization.
        support_lto_irs : list of bytes
            Deduplicated type-definition and Python-callback link images.
        temp_storage_types : tuple of str
            C++ scratch type names to query, or an empty tuple without scratch.
        udf_declarations : collections.OrderedDict
            Device declarations for Python operators, keyed by symbol name.
            Source coalescing uses these to construct a shared preamble.

        Raises
        ------
        ValueError
            A pointer offset lacks an earlier pointer target, the method has
            multiple outputs.
        RuntimeError
            The provider has incompatible bound compiler inputs, a callback
            targets a different compute capability, or two callbacks declare
            the same device symbol with different signatures.
        """

        compile_identity = self._bind_private_symbol_namespace(
            logical_warp_threads=logical_warp_threads,
            block_threads=block_threads,
            compile_identity=compile_identity,
        )
        support_lto_irs, udf_declarations = (
            self._collect_support_ltoirs_and_udf_declarations(
                provider_cc=compile_identity.cc
            )
        )

        algorithm_name = self.struct_name
        includes = self.includes or []
        type_definitions = self.type_definitions or []
        method_name = self.method_name

        alias_suffix = internal_mangle_cpp(self._symbol_base_name())
        algorithm_type_name = f"algorithm_t_{alias_suffix}"
        temp_storage_type_name = None
        temp_storage_types = ()
        if self.storage_abi is StorageABI.LEADING_POINTER:
            temp_storage_type_name = f"temp_storage_t_{alias_suffix}"
            temp_storage_types = (temp_storage_type_name,)

        buf = StringIO()
        w = buf.write

        w("#include <cuda/std/cstdint>\n")
        for include in includes:
            w(f"#include <{include}>\n")
        for type_definition in type_definitions:
            w(f"{type_definition.code}\n")
        w("\n")
        for decl in udf_declarations.values():
            w(f"{decl}\n")
        w("\n")

        w(f"using {algorithm_type_name} = cub::{algorithm_name};\n")
        if temp_storage_type_name is not None:
            w(
                f"using {temp_storage_type_name} = typename "
                f"{algorithm_type_name}::TempStorage;\n"
            )

        src = buf.getvalue()

        for method in self.parameters:
            param_decls = []
            func_decls = []
            param_args = []
            out_param = None

            param_arg_positions_by_pid = {}
            provider_parameters = (
                method[1:]
                if self.storage_abi is StorageABI.LEADING_POINTER
                else method
            )
            parameter_names = _cpp_parameter_names(
                provider_parameters,
                reserved={
                    *udf_declarations,
                    "temp_storage",
                    "temp_storages",
                    algorithm_type_name,
                    temp_storage_type_name,
                },
            )
            for pid, (param, name) in enumerate(
                zip(provider_parameters, parameter_names)
            ):
                if isinstance(param, StatelessOperator):
                    func_decls.extend(
                        param.wrap_decl(name).rstrip().splitlines()
                    )
                    param_args.append(name)
                elif isinstance(param, StatefulOperator):
                    func_decls.extend(
                        param.wrap_decl(name).rstrip().splitlines()
                    )
                    param_args.append(name)
                    param_decls.append(param.cpp_decl(name))
                elif isinstance(param, CxxFunction):
                    param_args.append(param.cpp)
                else:
                    if isinstance(param, PointerOffset):
                        offset_expression = name
                        if param.static_value is None:
                            param_decls.append(param.cpp_decl(name))
                        else:
                            offset_expression = str(param.static_value)
                        pointer_arg_pos = param_arg_positions_by_pid.get(
                            param.pointer_arg_index
                        )
                        if pointer_arg_pos is None:
                            raise ValueError(
                                "PointerOffset must reference an earlier "
                                "pointer parameter."
                            )
                        param_args[pointer_arg_pos] = (
                            f"({param_args[pointer_arg_pos]} + "
                            f"{offset_expression})"
                        )
                        continue
                    if isinstance(param, BoundedInteger):
                        checked_name = f"checked_{name}"
                        param_decls.append(param.cpp_decl(name))
                        func_decls.append(
                            f"if ({name} < {param.minimum} || "
                            f"{name} > {param.maximum}) {{"
                        )
                        func_decls.append('  asm volatile("trap;" : : :);')
                        func_decls.append("}")
                        provider_cpp = numba_type_to_cpp(param.provider_dtype)
                        func_decls.append(
                            f"{provider_cpp} {checked_name} = "
                            f"static_cast<{provider_cpp}>({name});"
                        )
                        param_arg = checked_name
                    elif isinstance(param, TransformedArray):
                        source_type = numba_type_to_cpp(param.value_dtype)
                        target_type = numba_type_to_cpp(param.target_dtype)
                        transformed_name = f"transformed_{name}"
                        param_decls.append(f"{source_type} *{name}")
                        func_decls.append(
                            f"{target_type} {transformed_name}[{param.size}];"
                        )
                        for index in range(param.size):
                            expression = param.cpp_expression.format(
                                value=f"{name}[{index}]"
                            )
                            func_decls.append(
                                f"{transformed_name}[{index}] = {expression};"
                            )
                        param_arg = transformed_name
                    elif isinstance(param, Array):
                        value_type = numba_type_to_cpp(param.value_dtype)
                        param_decls.append(f"{value_type} *{name}")
                        param_arg = (
                            f"*reinterpret_cast<{value_type}"
                            f" (*)[{param.size}]>({name})"
                        )
                    else:
                        param_decls.append(param.cpp_decl(name))
                        if getattr(param, "deref_on_call", False):
                            param_arg = f"*{name}"
                        else:
                            param_arg = name
                    if not self.fake_return and param.is_output:
                        if out_param is not None:
                            raise ValueError(
                                "Multiple output parameters not supported"
                            )
                        out_param = name
                        if self.output_by_reference:
                            param_arg_positions_by_pid[pid] = len(param_args)
                            param_args.append(param_arg)
                    else:
                        param_arg_positions_by_pid[pid] = len(param_args)
                        param_args.append(param_arg)

            buf = StringIO()
            w = buf.write

            mangled_name = self.mangled_name(method)
            param_decls_csv = ", ".join(param_decls)
            param_args_csv = ", ".join(param_args)

            w(f'extern "C" __device__ void {mangled_name}(')
            if self.storage_abi is StorageABI.LEADING_POINTER:
                assert temp_storage_type_name is not None
                w(f"{temp_storage_type_name} *temp_storage")
                if param_decls_csv:
                    w(", ")
            w(f"{param_decls_csv}) {{\n")
            for decl in func_decls:
                w(f"    {decl}\n")
            if out_param and not self.output_by_reference:
                w(f"    {out_param} = ")
            else:
                w("    ")
            if self.storage_abi is StorageABI.LEADING_POINTER:
                w(f"{algorithm_type_name}(*temp_storage).")
            else:
                w(f"{algorithm_type_name}().")
            w(f"{method_name}({param_args_csv});\n")
            w("}\n\n")
            self._emit_abi_wrapper(
                w,
                method,
                exported_name=mangled_name,
                internal_name=mangled_name,
                temp_storage_type_name=(
                    "temp_storage_t"
                    if temp_storage_type_name is None
                    else temp_storage_type_name
                ),
                temp_storage_param_pid=(
                    0
                    if self.storage_abi is StorageABI.LEADING_POINTER
                    else None
                ),
            )

            src += buf.getvalue()

        return (
            src,
            support_lto_irs,
            temp_storage_types,
            udf_declarations,
        )


def _dedupe_ltoirs(lto_irs):
    """Keep the first occurrence of each supporting link image.

    Provider compilation and bundling collect images from type definitions
    and parameter descriptors. ``lto_irs`` is their ordered iterable of byte
    images or text; compare content digests and return a list of the original
    objects, preserving first-use order.
    """

    seen = set()
    deduped = []
    for lto_ir in lto_irs:
        key = _lto_ir_digest(lto_ir)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(lto_ir)
    return deduped


def _param_coalesce_key(param):
    """Describe one parameter's effect on provider source and calling ABI.

    Preserve dtype, extent, transforms, bounds, and static values that alter
    generated code. Python callbacks also contribute their symbol, signature,
    exact target, and LTO digest; stateful callbacks include the state dtype.
    Runtime state contents are absent, allowing one provider to serve arrays
    with different values. Unknown descriptor types fall back to their class
    name and representation.
    """

    if isinstance(param, TransformedArray):
        return (
            "TransformedArray",
            str(param.value_dtype),
            str(param.target_dtype),
            param.size,
            param.cpp_expression,
        )
    if isinstance(param, Array):
        return ("Array", str(param.value_dtype), param.size, param.is_output)
    if isinstance(param, Pointer):
        return (
            "Pointer",
            str(param.value_dtype),
            param.is_output,
            getattr(param, "deref_on_call", False),
        )
    if isinstance(param, Reference):
        return ("Reference", str(param.value_dtype), param.is_output)
    if isinstance(param, PointerOffset):
        return (
            "PointerOffset",
            str(param.value_type),
            param.pointer_arg_index,
            param.static_value,
            param.is_output,
        )
    if isinstance(param, BoundedInteger):
        return (
            "BoundedInteger",
            str(param.value_type),
            str(param.provider_dtype),
            param.minimum,
            param.maximum,
            param.is_output,
        )
    if isinstance(param, ExactValue):
        return ("ExactValue", str(param.value_type), param.is_output)
    if isinstance(param, Value):
        return ("Value", str(param.value_type), param.is_output)
    if isinstance(param, CxxFunction):
        return ("CxxFunction", param.cpp, str(param.func_dtype))
    if isinstance(param, StatelessOperator):
        return (
            "StatelessOperator",
            param.name,
            param.ret_cpp_type,
            param.arg_cpp_types,
            param.compute_capability,
            _lto_ir_digest(param.ltoir),
        )
    if isinstance(param, StatefulOperator):
        return (
            "StatefulOperator",
            param.name,
            str(param.state_dtype),
            param.ret_cpp_type,
            param.arg_cpp_types,
            param.compute_capability,
            _lto_ir_digest(param.ltoir),
        )
    return (type(param).__name__, repr(param))


def algo_coalesce_key(
    algo: Algorithm,
    *,
    logical_warp_threads: int | None = None,
    block_threads: _BlockThreads | None = None,
) -> tuple[object, ...]:
    """Describe a provider for source coalescing and symbol qualification.

    Include the C++ operation, ordered ABI descriptors, type-definition source
    and link-image digests, topology, storage/synchronization contracts, and
    already resolved compiler identity. Human-readable parameter names are
    excluded: renaming a wrapper argument does not create a distinct provider.
    The bundler uses this key to choose one source representative for equivalent
    algorithms; private symbol qualification hashes the same description.

    This function only reads the supplied state. It does not resolve a compiler
    context, bind a device target, or validate thread dimensions, so callers
    must establish those inputs before comparing compilation-ready providers.

    Parameters
    ----------
    algo : Algorithm
        Provider whose current source and ABI state is described.
    logical_warp_threads : int, optional
        Logical warp width override, falling back to
        ``algo.logical_warp_threads``.
    block_threads : int, tuple of int, or list of int, optional
        Block configuration override, falling back to ``algo.block_threads``.

    Returns
    -------
    tuple
        Equality key for the provider's current compilation-relevant state.
    """

    type_defs = []
    for type_definition in getattr(algo, "type_definitions", None) or []:
        code = getattr(type_definition, "code", None) or ""
        lto_irs = getattr(type_definition, "lto_irs", []) or []
        lto_ir_hashes = tuple(_lto_ir_digest(lto_ir) for lto_ir in lto_irs)
        type_defs.append((code, lto_ir_hashes))

    params_key = tuple(
        tuple(_param_coalesce_key(param) for param in method)
        for method in getattr(algo, "parameters", [])
    )

    resolved_logical_warp_threads = logical_warp_threads
    if resolved_logical_warp_threads is None:
        resolved_logical_warp_threads = getattr(
            algo, "logical_warp_threads", None
        )
    resolved_block_threads = block_threads
    if resolved_block_threads is None:
        resolved_block_threads = getattr(algo, "block_threads", None)

    return (
        getattr(algo, "struct_name", None),
        getattr(algo, "method_name", None),
        getattr(algo, "c_name", None),
        tuple(getattr(algo, "includes", None) or []),
        tuple(type_defs),
        params_key,
        resolved_logical_warp_threads,
        resolved_block_threads,
        getattr(algo, "fake_return", None),
        getattr(algo, "output_by_reference", None),
        getattr(algo, "storage_abi", None),
        getattr(algo, "execution_scope", None),
        getattr(algo, "synchronization_scope", None),
        getattr(algo, "_compile_context", None),
        getattr(algo, "_provider_compile_identity", None),
    )


def _strip_source_preamble(src, algo, udf_decls):
    """Remove shared includes and declarations before combining wrappers.

    Only the first exact occurrence of each generated fragment is removed.
    The bundle emits the shared preamble once for all representatives.
    """
    body = src
    body = body.replace("#include <cuda/std/cstdint>\n", "", 1)
    for include in algo.includes or []:
        body = body.replace(f"#include <{include}>\n", "", 1)
    for type_definition in algo.type_definitions or []:
        code = getattr(type_definition, "code", "")
        if code:
            body = body.replace(f"{code}\n", "", 1)
    for decl in udf_decls.values():
        body = body.replace(f"{decl}\n", "", 1)
    return body.lstrip()


def prepare_specialization(
    specialization: Algorithm,
    *,
    logical_warp_threads: int | None = None,
    block_threads: _BlockThreads | None = None,
) -> Algorithm:
    """Qualify a provider's private symbols and retain its exact topology.

    Return the specialization for the typed pipeline to compile and link with
    the kernel's other providers.
    """

    if logical_warp_threads is not None:
        specialization.logical_warp_threads = logical_warp_threads
    if block_threads is not None:
        specialization.block_threads = block_threads
    specialization._bind_private_symbol_namespace(
        logical_warp_threads=logical_warp_threads,
        block_threads=block_threads,
    )
    return specialization


__all__ = [
    "NUMBA_TYPES_TO_CPP",
    "Algorithm",
    "Array",
    "BoundedInteger",
    "Constant",
    "CxxFunction",
    "Dependency",
    "DependentArray",
    "DependentCxxOperator",
    "DependentPointer",
    "DependentPointerReference",
    "DependentPythonOperator",
    "DependentReference",
    "DependentStatefulOperator",
    "ExactValue",
    "Parameter",
    "Pointer",
    "PointerOffset",
    "PointerReference",
    "Reference",
    "StatefulOperator",
    "StatelessOperator",
    "SubstitutionFailure",
    "TemplateParameter",
    "TransformedArray",
    "TypeWrapper",
    "Value",
    "_hash_symbol_value",
    "_ltoir_to_ptx",
    "_validate_logical_warp_threads",
    "algo_coalesce_key",
    "internal_mangle_cpp",
    "kernel_compile_target",
    "mangle_symbol",
    "numba_type_to_cpp",
    "numba_type_to_wrapper",
    "prepare_specialization",
]
