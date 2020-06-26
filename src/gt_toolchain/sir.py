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
from typing import List, Optional, Union

from pydantic import root_validator, validator  # noqa: F401

from eve.core import Node

from . import common


# = statements.proto =


@enum.unique
class LocationType(enum.IntEnum):
    LocationTypeUnknown = 0
    Vertex = 1
    Cell = 2
    Edge = 3


class Expr(Node):
    location_type: Optional[LocationType]


class Stmt(Node):
    location_type: Optional[LocationType]


class AST(Node):
    root: Stmt


class UnstructuredDimension(Node):
    dense_location_type: LocationType
    sparse_part: Optional[List[LocationType]]


class FieldDimensions(Node):
    horizontal_dimension: Union[UnstructuredDimension]  # TODO CartesianDimension
    # TODO mask_k: int


class Field(Node):
    name: str
    is_temporary: bool
    field_dimensions: FieldDimensions


# TODO class Direction
# TODO class Offset
# TODO class StencilFunctionArg


class Interval(Node):
    # TODO
    pass


class BuiltinType(Node):
    type_id: common.DataType


# TODO class Dimension


class Type(Node):
    data_type: Union[str, BuiltinType]
    is_const: bool
    is_volatile: bool


class VerticalRegion(Node):
    loop_order: common.LoopOrder
    ast: AST
    interval: Interval
    # i_range: Interval
    # j_range: Interval


# TODO class StencilCall
# TODO class Extents
# TODO class Accesses

# = statements.proto = AST


class BlockStmt(Stmt):
    statements: List[Stmt]


class ExprStmt(Stmt):
    expr: Expr


# TODO class ReturnStmt(Stmt_


class VarDeclStmt(Stmt):
    data_type: Type
    name: str
    dimension: int
    op: str
    init_list: List[Expr]


class VerticalRegionDeclStmt(Stmt):
    vertical_region: VerticalRegion


# TODO class StencilCallDeclStmt(Stmt)
# TODO class BoundaryConditionDeclStmt(Stmt)
# TODO class IfStmt(Stmt)


class UnaryOperator(Expr):
    op: str
    operand: Expr
    right: Expr


class BinaryOperator(Expr):
    left: Expr
    op: str
    right: Expr


class AssignmentExpr(Expr):
    left: Expr
    op: str
    right: Expr


class TernaryOperator(Expr):
    cond: Expr
    left: Expr
    right: Expr


# TODO class FunCallExpr
# TODO class StencilFunCallExpr
# TODO class StencilFunArgExpr


class VarAccessExpr(Expr):
    name: str
    # index: Expr # TODO
    # is_external: bool # TODO


class ZeroOffset(Node):
    pass


class UnstructuredOffset(Node):
    has_offset: bool


class FieldAccessExpr(Expr):
    name: str
    vertical_offset: int
    horizontal_offset: Union[UnstructuredOffset, ZeroOffset]  # TODO CartesianOffset
    # TODO argument_map
    # TODO argument_offset
    # TODO negate_offset
    # TODO AccessExprData and ID probably unused in SIR


class LiteralAccessExpr(Expr):
    value: str
    data_type: BuiltinType
    # TODO AccessExprData and ID probably unused in SIR


class ReductionOverNeighborExpr(Expr):
    op: str
    rhs: Expr
    init: Expr
    # TODO weights
    chain: List[LocationType]


# = sir.proto =


class Stencil(Node):
    name: str
    params: List[Field]  # TODO 'fields' would shadow base class 'fields'
    ast: AST


# TODO GlobalVariableMap
# TODO GlobalVariableValue
# TODO StencilFunction


class SIR(Node):
    stencils: List[Stencil]
    filename: str
    # TODO stencil_functions
    # TODO global_variables
