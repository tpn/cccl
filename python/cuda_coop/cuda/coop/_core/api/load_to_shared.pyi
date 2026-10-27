# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Protocol

from cuda.coop._typing import IntegerValue

from .thread_group import BlockGroup

class PendingSharedCopyLike(Protocol):
    def wait(self) -> Any: ...

def load_to_shared_async(
    group: BlockGroup,
    source: object,
    shared_buffer: object,
    /,
    *,
    num_items: int,
    offset: IntegerValue = 0,
) -> PendingSharedCopyLike: ...

__all__ = ["PendingSharedCopyLike", "load_to_shared_async"]
