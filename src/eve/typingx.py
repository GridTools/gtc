# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Python version independent typings."""

# flake8: noqa

from typing import *


T = TypeVar("T")
FrozenList = Tuple[T, ...]

AnyCallable = Callable[..., Any]
AnyNoneCallable = Callable[..., None]
AnyNoArgCallable = Callable[[], Any]


def __getattr__(name: str) -> Any:
    if name == "__path__":
        # __path__ can only be defined for packages
        raise AttributeError(f"module '{name}' has no attribute '__path__'")

    try:
        import typing

        return getattr(typing, name)
    except AttributeError:
        try:
            import typing_extensions

            return getattr(typing_extensions, name)
        except AttributeError:
            raise ImportError(f"cannot import name '{name}' from 'typing' or 'typing_extensions'")
