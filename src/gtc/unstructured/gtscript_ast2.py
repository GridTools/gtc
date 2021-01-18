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

from typing import Any, Optional, Union

from eve import Node, Str
from eve.typingx import FrozenList


class Stmt(Node):
    pass


class Expr(Node):
    pass


class Name(Expr):
    name: Str


class Subscript(Expr):
    value: Name
    index: Union[Name, FrozenList[Name]]


class Call(Expr):
    func: Name
    args: FrozenList[Expr]


class Comprehension(Node):
    target: Union[Name, FrozenList[Name]]
    iter_: Expr


class GeneratorExp(Expr):
    elt: Expr
    generators: FrozenList[Comprehension]


class BinOp(Expr):
    left: Expr
    op: Str
    right: Expr


class Constant(Expr):
    value: Any


class Assign(Stmt):
    targets: FrozenList[Expr]
    value: Expr


class WithItem(Node):
    expr: Expr
    var: Optional[Name]


class With(Stmt):
    items: FrozenList[WithItem]
    body: FrozenList[Stmt]


class Function(Node):
    name: Str
    body: FrozenList[Stmt]
