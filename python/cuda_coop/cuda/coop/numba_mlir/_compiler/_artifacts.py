# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Represent compiler versions and validate closed-set compiler options."""

from collections import namedtuple

version = namedtuple("version", ("major", "minor"))


def check_in(name, arg, values):
    """Validate a small closed-set compiler option."""

    if arg not in values:
        raise ValueError(f"{name} must be in {values} ; got {name} = {arg}")


__all__ = ["check_in", "version"]
