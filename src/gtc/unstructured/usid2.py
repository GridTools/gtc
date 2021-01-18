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

from eve import Bool, Int, Node, Str
from eve.typingx import FrozenList
from gtc import common
from gtc.unstructured.gtir2 import ReduceOperator


class Connectivity(Node):
    name: Str
    max_neighbors: Int
    has_skip_values: Bool


class Field(Node):
    name: Str
    # dtype is missing because it is not used in the code generation


class SparseField(Field):
    connectivity: Str


class Temporary(Node):
    name: Str
    location_type: Str
    dtype: Str


class Composite(Node):
    name: Str
    items: FrozenList[Str]


class Expr(Node):
    pass


class Literal(Expr):
    value: Str


class FieldAccess(Expr):
    name: Str
    location: Str


class BinaryOp(Expr):
    op: common.BinaryOperator
    left: Expr
    right: Expr


# TODO(till): discuss it with Hannes (primary, secondary)
class NeighborReduce(Expr):
    op: ReduceOperator
    dtype: Str
    connectivity: Str
    primary: Str
    secondary: Str
    body: Expr


class Assign(Node):
    left: FieldAccess
    right: Expr


class Kernel(Node):
    location_type: Str
    primary: Composite
    secondaries: FrozenList[Composite]
    body: FrozenList[Assign]


class Computation(Node):
    name: Str
    connectivities: FrozenList[Connectivity]
    args: FrozenList[Field]
    temporaries: FrozenList[Temporary]
    kernels: FrozenList[Kernel]
