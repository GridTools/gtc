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
from typing import List, Optional, Tuple, Union  # noqa: F401

from pydantic import Field, root_validator, validator  # noqa: F401

from eve.core import Node, NodeVisitor, SourceLocation, StrEnum  # noqa: F401

from . import common


@enum.unique
class LocationType(enum.IntEnum):
    Node = 0
    Edge = 1
    Face = 2
    NoLocation = 3


class Expr(Node):
    location_type: LocationType
    pass


class Stmt(Node):
    location_type: LocationType
    pass


class FieldAccessExpr(Expr):
    name: str
    offset: Tuple[bool, int]
    # TODO to add a validator we need to lookup a symbol table for the field's location type


class LiteralExpr(Expr):
    value: str
    data_type: common.DataType


class AssignmentExpr(Expr):
    left: Expr
    right: Expr

    @root_validator(pre=True)
    def check_location_type(cls, values):
        if not (values["left"].location_type == values["right"].location_type):
            raise ValueError("Location type mismatch")

        if "location_type" not in values:
            values["location_type"] = values["left"].location_type
        else:
            if not (values["left"] == values["location_type"]):
                raise ValueError("Location type mismatch")

        return values


class ReduceOverNeighbourExpr(Expr):
    operation: common.BinaryOperator
    right: Expr
    init: Expr
    right_location_type: LocationType  # TODO Doesn't make sense?

    @root_validator(pre=True)
    def check_location_type(cls, values):
        # TODO location type of init? Do we need a `NoLocation` location type?
        if "right_location_type" not in values:
            values["right_location_type"] = values["right"].location_type
        else:
            if not (values["right_location_type"] == values["right"].location_type):
                raise ValueError("Location type mismatch")
        return values


class BlockStmt(Stmt):
    statements: List[Stmt]

    @root_validator(pre=True)
    def check_location_type(cls, values):
        statements = values.get("statements")
        if len(statements) == 0:
            raise ValueError("BlockStmt is empty")

        if not all(s == statements[0] for s in statements):
            raise ValueError("Location type mismatch")

        if "location_type" not in values:
            values["location_type"] = statements[0].location_type
        else:
            if not (statements[0].location_type == values["location_type"]):
                raise ValueError("Location type mismatch")
        return values


class ExprStmt(Stmt):
    expr: AssignmentExpr

    @root_validator(pre=True)
    def check_location_type(cls, values):
        if "location_type" not in values:
            values["location_type"] = values["expr"].location_type
        else:
            if not (values["expr"].location_type == values["location_type"]):
                raise ValueError("Location type mismatch")
        return values


class UnstructuredField(Node):
    name: str
    location_type: LocationType
    data_type: common.DataType


class HorizontalLoop(Node):
    location_type: LocationType
    ast: BlockStmt

    @root_validator(pre=True)
    def check_location_type(cls, values):
        if "location_type" not in values:
            values["location_type"] = values["ast"].location_type
        else:
            if not (values["ast"].location_type == values["location_type"]):
                raise ValueError("Location type mismatch")
        return values


class ForK(Node):
    horizontal_loops: List[HorizontalLoop]
    loop_order: common.LoopOrder


class Stencil(Node):
    name: str
    k_loops: List[ForK]


class Computation(Node):
    params: List[UnstructuredField]
    stencils: List[Stencil]
