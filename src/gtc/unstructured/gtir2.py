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

import enum
from typing import Any, Union

from eve import Bool, Int, Node, Str, StrEnum
from eve.typingx import FrozenList
from gtc import common


class Connectivity(Node):
    name: Str
    primary: common.LocationType
    secondary: common.LocationType
    max_neighbors: Int
    has_skip_values: Bool


class Field(Node):
    name: Str
    location_type: common.LocationType
    dtype: common.DataType


class SparseField(Node):
    name: Str
    connectivity: Str
    dtype: common.DataType


class Expr(Node):
    pass


class FieldAccess(Expr):
    name: Str
    location: Str


class SparseFieldAccess(Expr):
    name: Str
    primary: Str
    secondary: Str


class BinaryOp(Expr):
    op: common.BinaryOperator
    left: Expr
    right: Expr


class Literal(Expr):
    value: Any
    dtype: common.DataType


@enum.unique
class ReduceOperator(StrEnum):
    """Reduction operator identifier."""

    SUM = "sum"
    PRODUCT = "product"
    MIN = "min"
    MAX = "max"


class SecondaryLocation(Node):
    name: Str
    connectivity: Str
    primary: Str


class NeighborReduce(Expr):
    op: ReduceOperator
    dtype: common.DataType
    location: SecondaryLocation
    body: Expr


class Assign(Node):
    left: FieldAccess
    right: Expr


class PrimaryLocation(Node):
    name: Str
    location_type: common.LocationType


class Stencil(Node):
    loop_order: common.LoopOrder
    location: PrimaryLocation
    body: FrozenList[Assign]


class Computation(Node):
    name: Str
    connectivities: FrozenList[Connectivity]
    args: FrozenList[Union[Field, SparseField]]
    temporaries: FrozenList[Field]
    stencils: FrozenList[Stencil]
