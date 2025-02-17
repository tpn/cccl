# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum
from typing import Callable, Literal, Type, Union

import numba

from cuda.cooperative.experimental._common import make_binary_tempfile
from cuda.cooperative.experimental._types import (
    Algorithm,
    Dependency,
    DependentArray,
    DependentFunction,
    DependentOperator,
    DependentReference,
    DependentValue,
    Invocable,
    Pointer,
    TemplateParameter,
)

CUB_BLOCK_SCAN_ALGOS = {
    "raking": "::cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING",
    "raking_memoize": "::cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING_MEMOIZE",
    "warp_scans": "::cub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS",
}


class ScanOpType(Enum):
    Sum = "Sum"
    Known = "Known"
    Callable = "Callable"


class ScanOp:
    """
    Represents an associative binary operator for a prefix scan operation.
    """

    OPS = {
        "+": "::std::plus<T>",
        "-": "::std::minus<T>",
        "*": "::std::multiplies<T>",
        "min": "::std::min<T>",
        "max": "::std::max<T>",
        "&": "::std::bit_and<T>",
        "|": "::std::bit_or<T>",
        "^": "::std::bit_xor<T>",
    }

    def __init__(
        self, op: Union[Literal["+", "*", "min", "max", "&", "|", "^"], Callable]
    ):
        self.op = op
        self.op_type = None
        self.op_cpp = None
        if isinstance(op, str):
            if op not in self.OPS:
                raise ValueError(f"Unsupported scan operator: {op}")
            if op == "+":
                self.op_type = ScanOpType.Sum
            else:
                self.op_type = ScanOpType.Known
            self.op_cpp = self.OPS[op]
        else:
            # Verify that the callable is a valid binary operator.
            if not callable(op):
                raise ValueError("scan_op must be a callable object")
            if not callable(op(0, 0)):
                raise ValueError("scan_op must be a binary operator")
            self.op_type = ScanOpType.Callable

    def __repr__(self):
        return f"ScanOp({self.op})"


def _scan(
    dtype: Type[numba.types.Number],
    threads_in_block: int,
    items_per_thread: int = 1,
    mode: Literal["inclusive", "exclusive"] = "exclusive",
    initial_value: Type[numba.types.Number] = None,
    scan_op: Union[Literal["+", "*", "min", "max", "&", "|", "^"], Callable] = "+",
    block_prefix_callback_op: Callable = None,
    algorithm: Literal["raking", "raking_memoize", "warp_scans"] = "raking",
) -> Callable:
    if algorithm not in CUB_BLOCK_SCAN_ALGOS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    if items_per_thread < 1:
        raise ValueError("items_per_thread must be greater than or equal to 1")

    if mode == "inclusive":
        mode_str = "Inclusive"
    elif mode == "exclusive":
        mode_str = "Exclusive"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # This will raise an error if scan_op is invalid.
    scan_op = ScanOp(scan_op)
    if scan_op.op_type == ScanOpType.Sum:
        # Make sure we specialize the correct CUB API for exclusive sum.
        cpp_function_name = f"{mode_str}Sum"
    else:
        cpp_function_name = f"{mode_str}Scan"

    specialization_kwds = {
        "T": dtype,
        "BLOCK_DIM_X": threads_in_block,
        "ALGORITHM": CUB_BLOCK_SCAN_ALGOS[algorithm],
    }

    template_parameters = [
        TemplateParameter("T"),
        TemplateParameter("BLOCK_DIM_X"),
        TemplateParameter("ALGORITHM"),
    ]

    fake_return = False

    if scan_op.op_type == ScanOpType.Sum:
        if items_per_thread == 1:
            parameters = [
                # Signature:
                # void BlockScan<T, BLOCK_DIM_X, ALGORITHM>(
                #     temp_storage
                # ).<Inclusive|Exclusive>Sum(
                #     T, # input
                #     T& # output
                # )
                [
                    # temp_storage
                    Pointer(numba.uint8),
                    # T input
                    DependentValue(Dependency("T")),
                    # T& output
                    DependentReference(Dependency("T"), is_output=True),
                ],
                # Signature:
                # void BlockScan<T, BLOCK_DIM_X, ALGORITHM>(
                #     temp_storage
                # ).<Inclusive|Exclusive>Sum(
                #     T,                     # input
                #     T&,                    # output
                #     BlockPrefixCallbackOp& # block_prefix_callback_op
                # )
                [
                    # temp_storage
                    Pointer(numba.uint8),
                    # T input
                    DependentValue(Dependency("T")),
                    # T& output
                    DependentReference(Dependency("T"), is_output=True),
                    # BlockPrefixCallbackOp& block_prefix_callback_op
                    DependentOperator(
                        Dependency("T"),
                        [Dependency("T")],
                        Dependency("BlockPrefixCallbackOp"),
                    ),
                ],
            ]

            fake_return = True

        else:
            assert items_per_thread > 1, items_per_thread

            parameters = [
                # Signature:
                # void BlockScan<T, BLOCK_DIM_X, ITEMS_PER_THREAD, ALGORITHM>(
                #     temp_storage
                # ).<Inclusive|Exclusive>Sum(
                #     T (&)[ITEMS_PER_THREAD], # input
                #     T (&)[ITEMS_PER_THREAD]  # output
                # )
                [
                    # temp_storage
                    Pointer(numba.uint8),
                    # T (&)[ITEMS_PER_THREAD] input
                    DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                    # T (&)[ITEMS_PER_THREAD] output
                    DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                ],
                # Signature:
                # void BlockScan<T, BLOCK_DIM_X, ITEMS_PER_THREAD, ALGORITHM>(
                #     temp_storage
                # ).<Inclusive|Exclusive>Sum(
                #     T (&)[ITEMS_PER_THREAD], # input
                #     T (&)[ITEMS_PER_THREAD], # output
                #     BlockPrefixCallbackOp&   # block_prefix_callback_op
                # )
                [
                    # temp_storage
                    Pointer(numba.uint8),
                    # T (&)[ITEMS_PER_THREAD] input
                    DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                    # T (&)[ITEMS_PER_THREAD] output
                    DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                    # BlockPrefixCallbackOp& block_prefix_callback_op
                    DependentOperator(
                        Dependency("T"),
                        [Dependency("T")],
                        Dependency("BlockPrefixCallbackOp"),
                    ),
                ],
            ]

            specialization_kwds["ITEMS_PER_THREAD"] = items_per_thread

            template_parameters.append(TemplateParameter("ITEMS_PER_THREAD"))

    elif scan_op.op_type == ScanOpType.Known:
        if items_per_thread == 1:
            parameters = [
                # Signature:
                # void BlockScan<T, BLOCK_DIM_X, ALGORITHM>(
                #     temp_storage
                # ).<Inclusive|Exclusive>Scan(
                #     T,     # input
                #     T&,    # output
                #     T,     # initial_value
                #     ScanOp # scan_op
                # )
                [
                    # temp_storage
                    Pointer(numba.uint8),
                    # T input
                    DependentValue(Dependency("T")),
                    # T& output
                    DependentReference(Dependency("T"), is_output=True),
                    # T initial_value
                    DependentValue(Dependency("T")),
                    # ScanOp scan_op
                    DependentFunction(Dependency("ScanOp"), op=scan_op.op_cpp),
                ],
                # Signature:
                # void BlockScan<T, BLOCK_DIM_X, ALGORITHM>(
                #     temp_storage
                # ).<Inclusive|Exclusive>Scan(
                #     T,                     # input
                #     T&,                    # output
                #     T,                     # initial_value
                #     ScanOp,                # scan_op
                #     BlockPrefixCallbackOp& # block_prefix_callback_op
                # )
                [
                    # temp_storage
                    Pointer(numba.uint8),
                    # T input
                    DependentValue(Dependency("T")),
                    # T& output
                    DependentReference(Dependency("T"), is_output=True),
                    # T initial_value
                    DependentValue(Dependency("T")),
                    # ScanOp scan_op
                    DependentFunction(Dependency("ScanOp"), op=scan_op.op_cpp),
                    # BlockPrefixCallbackOp& block_prefix_callback_op
                    DependentOperator(
                        Dependency("T"),
                        [Dependency("T")],
                        Dependency("BlockPrefixCallbackOp"),
                    ),
                ],
            ]

            fake_return = True

        else:
            parameters = [
                # Signature:
                # void BlockScan<T, BLOCK_DIM_X, ITEMS_PER_THREAD, ALGORITHM>(
                #     temp_storage
                # ).<Inclusive|Exclusive><Sum|Scan>(
                #     T (&)[ITEMS_PER_THREAD], # input
                #     T (&)[ITEMS_PER_THREAD], # output
                #     T,                       # initial_value
                #     ScanOp                   # scan_op
                # )
                [
                    # temp_storage
                    Pointer(numba.uint8),
                    # T (&)[ITEMS_PER_THREAD] input
                    DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                    # T (&)[ITEMS_PER_THREAD] output
                    DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                    # T initial_value
                    DependentValue(Dependency("T")),
                    # ScanOp scan_op
                    DependentFunction(Dependency("ScanOp")),
                ],
                # Signature:
                # void BlockScan<T, BLOCK_DIM_X, ITEMS_PER_THREAD, ALGORITHM>(
                #     temp_storage
                # ).<Inclusive|Exclusive><Sum|Scan>(
                #     T (&)[ITEMS_PER_THREAD], # input
                #     T (&)[ITEMS_PER_THREAD], # output
                #     T,                       # initial_value
                #     ScanOp,                  # scan_op
                #     BlockPrefixCallbackOp&   # block_prefix_callback_op
                # )
                [
                    # temp_storage
                    Pointer(numba.uint8),
                    # T (&)[ITEMS_PER_THREAD] input
                    DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                    # T (&)[ITEMS_PER_THREAD] output
                    DependentArray(Dependency("T"), Dependency("ITEMS_PER_THREAD")),
                    # T initial_value
                    DependentValue(Dependency("T")),
                    # ScanOp scan_op
                    DependentFunction(Dependency("ScanOp")),
                    # BlockPrefixCallbackOp& block_prefix_callback_op
                    DependentOperator(
                        Dependency("T"),
                        [Dependency("T")],
                        Dependency("BlockPrefixCallbackOp"),
                    ),
                ],
            ]

            specialization_kwds["ITEMS_PER_THREAD"] = items_per_thread

            template_parameters.append(TemplateParameter("ITEMS_PER_THREAD"))

    template = Algorithm(
        "BlockScan",
        cpp_function_name,
        "block_scan",
        ["cub/block/block_scan.cuh"],
        template_parameters,
        parameters,
        fake_return=fake_return,
    )

    if block_prefix_callback_op is not None:
        specialization_kwds["BlockPrefixCallbackOp"] = block_prefix_callback_op

    specialization = template.specialize(specialization_kwds)
    return Invocable(
        temp_files=[
            make_binary_tempfile(ltoir, ".ltoir")
            for ltoir in specialization.get_lto_ir()
        ],
        temp_storage_bytes=specialization.get_temp_storage_bytes(),
        algorithm=specialization,
    )


def exclusive_sum(
    dtype: Type[numba.types.Number],
    threads_in_block: int,
    items_per_thread: int = 1,
    prefix_op: Callable = None,
    algorithm: Literal["raking", "raking_memoize", "warp_scans"] = "raking",
) -> Callable:
    """
    Computes an exclusive block-wide prefix sum.
    """
    return _scan(
        dtype=dtype,
        threads_in_block=threads_in_block,
        items_per_thread=items_per_thread,
        mode="exclusive",
        scan_op="+",
        block_prefix_callback_op=prefix_op,
        algorithm=algorithm,
    )


def inclusive_sum(
    dtype: Type[numba.types.Number],
    threads_in_block: int,
    items_per_thread: int = 1,
    prefix_op: Callable = None,
    algorithm: Literal["raking", "raking_memoize", "warp_scans"] = "raking",
) -> Callable:
    """
    Computes an inclusive block-wide prefix sum.
    """
    return _scan(
        dtype=dtype,
        threads_in_block=threads_in_block,
        items_per_thread=items_per_thread,
        mode="inclusive",
        scan_op="+",
        block_prefix_callback_op=prefix_op,
        algorithm=algorithm,
    )


def exclusive_scan(
    dtype: Type[numba.types.Number],
    threads_in_block: int,
    scan_op: Union[Literal["+", "*", "min", "max", "&", "|", "^"], Callable],
    initial_value: Type[numba.types.Number] = 1,
    items_per_thread: int = 1,
    prefix_op: Callable = None,
    algorithm: Literal["raking", "raking_memoize", "warp_scans"] = "raking",
) -> Callable:
    """
    Computes an exclusive block-wide prefix scan.
    """
    return _scan(
        dtype=dtype,
        threads_in_block=threads_in_block,
        items_per_thread=items_per_thread,
        mode="exclusive",
        initial_value=initial_value,
        scan_op=scan_op,
        block_prefix_callback_op=prefix_op,
        algorithm=algorithm,
    )


def inclusive_scan(
    dtype: Type[numba.types.Number],
    threads_in_block: int,
    scan_op: Union[Literal["+", "*", "min", "max", "&", "|", "^"], Callable],
    initial_value: Type[numba.types.Number] = 1,
    items_per_thread: int = 1,
    prefix_op: Callable = None,
    algorithm: Literal["raking", "raking_memoize", "warp_scans"] = "raking",
) -> Callable:
    """
    Computes an inclusive block-wide prefix scan.
    """
    return _scan(
        dtype=dtype,
        threads_in_block=threads_in_block,
        items_per_thread=items_per_thread,
        mode="inclusive",
        initial_value=initial_value,
        scan_op=scan_op,
        block_prefix_callback_op=prefix_op,
        algorithm=algorithm,
    )
