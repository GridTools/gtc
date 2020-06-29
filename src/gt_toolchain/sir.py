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


class SourceLocation(Node):
    line: int
    column: int


# helper (not part of dawn-SIR)
class ExprCommon(Node):
    location_type: Optional[LocationType]
    loc: Optional[SourceLocation]


class Expr(ExprCommon):
    pass


class Stmt(ExprCommon):
    pass


class AST(Node):
    root: Stmt


class CartesianDimension(Node):
    mask_cart_i: int
    mask_cart_j: int


class UnstructuredDimension(Node):
    dense_location_type: LocationType
    sparse_part: Optional[List[LocationType]]


class FieldDimensions(Node):
    horizontal_dimension: Union[CartesianDimension, UnstructuredDimension]
    mask_k: int = 1  # 1 == active


class Field(Node):
    name: str
    loc: Optional[SourceLocation]
    is_temporary: bool
    field_dimensions: FieldDimensions


# TODO class Direction
# TODO class Offset
# TODO class StencilFunctionArg


@enum.unique
class SpecialLevel(enum.IntEnum):
    Start = 0
    End = 1


# TODO maybe remove defaults
class Interval(Node):
    lower_level: Union[SpecialLevel, int] = SpecialLevel.Start
    upper_level: Union[SpecialLevel, int] = SpecialLevel.End
    lower_offset: int = 0
    upper_offset: int = 0


class BuiltinType(Node):
    type_id: common.DataType


@enum.unique
class Direction(enum.IntEnum):
    I_dir = 0
    J_dir = 1
    K_dir = 2
    Invalid = 3


class Dimension(Node):
    direction: Direction


class Type(Node):
    data_type: Union[str, BuiltinType]
    is_const: bool
    is_volatile: bool


# TODO maybe remove defaults
class VerticalRegion(Node):
    loop_order: common.LoopOrder
    loc: Optional[SourceLocation]
    ast: AST
    interval: Interval
    i_range: Interval = Interval()
    j_range: Interval = Interval()


# TODO class StencilCall


class Extent(Node):
    minus: int
    plus: int


class CartesianExtent(Node):
    i_extent: Extent
    j_extent: Extent


class UnstructedExtent(Node):
    has_extent: bool


class ZeroExtent(Node):
    pass


class Extents(Node):
    horizontal_extent: Union[CartesianExtent, UnstructedExtent, ZeroExtent]
    vertical_extent: Extent


# TODO class Accesses (TODO is this IIR only?)


class BlockStmt(Stmt):
    statements: List[Stmt]


# TODO Loop stuff


class ExprStmt(Stmt):
    expr: Expr


class ReturnStmt(Stmt):
    expr: Expr


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


class IfStmt(Stmt):
    cond_part: ExprStmt
    then_part: Stmt
    else_part: Optional[Stmt]


class UnaryOperator(Expr):
    op: str
    operand: Expr


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
    index: Optional[Expr]
    is_external: bool = False


class CartesianOffset(Node):
    i_offset: int
    j_offset: int


class UnstructuredOffset(Node):
    has_offset: bool


class ZeroOffset(Node):
    pass


class FieldAccessExpr(Expr):
    name: str
    vertical_offset: int
    horizontal_offset: Union[CartesianOffset, UnstructuredOffset, ZeroOffset]
    # TODO argument_map
    # TODO argument_offset
    # TODO negate_offset


class LiteralAccessExpr(Expr):
    value: str
    data_type: BuiltinType


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


@enum.unique
class GridType(enum.IntEnum):
    GridTypeUnknown = 0
    Unstructured = 1
    Cartesian = 2


class SIR(Node):
    grid_type: GridType
    stencils: List[Stencil]
    filename: str
    # TODO stencil_functions
    # TODO global_variables
