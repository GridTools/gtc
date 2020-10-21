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

from typing import List, Optional, Tuple, Union

from devtools import debug  # noqa: F401
from pydantic import root_validator, validator

import eve
from eve import Node, Str
from gtc import common


class Expr(Node):
    location_type: common.LocationType
    pass


class Stmt(Node):
    location_type: common.LocationType
    pass


class Literal(Expr):
    value: Union[Str, common.BuiltInLiteral]
    vtype: common.DataType


class NeighborChain(Node):
    elements: Tuple[common.LocationType, ...]

    class Config(eve.concepts.FrozenModelConfig):
        pass

    # TODO see https://github.com/eth-cscs/eve_toolchain/issues/40
    def __hash__(self):
        return hash(self.elements)

    def __eq__(self, other):
        return self.elements == other.elements

    @validator("elements")
    def not_empty(cls, elements):
        if len(elements) < 1:
            raise ValueError("NeighborChain must contain at least one locations")
        return elements

    def __str__(self):
        return "_".join([common.LocationType(loc).name for loc in self.elements])


class LocalVar(Node):
    name: Str
    location_type: common.LocationType


class ScalarLocalVar(LocalVar):
    vtype: common.DataType
    init: Optional[Expr]  # TODO: use in gtir to nir lowering for reduction var


class TensorLocalVar(LocalVar):
    vtype: common.DataType
    shape: List[int]
    init: Optional[List[Expr]]


class LocalFieldVar(TensorLocalVar):
    def __init__(self, *args, max_size, **kwargs):
        assert "shape" not in kwargs
        return super().__init__(*args, shape=[max_size], **kwargs)

    domain: common.LocationType  # the type of locations the LocalField is defined on

    @validator("shape", pre=True, always=True)
    def ensure_one_dimensional(cls, shape):
        if len(shape) != 1:
            raise ValueError("Invalid shape for LocalFieldVar.")
        return shape

    @property
    def max_size(self):
        return self.shape[0]


class BlockStmt(Stmt):
    declarations: List[LocalVar]
    statements: List[Stmt]

    class Config:
        validate_assignment = True

    @root_validator(pre=True)
    def check_location_type(cls, values):
        all_locations = [s.location_type for s in values["statements"]] + [
            d.location_type for d in values["declarations"]
        ]

        if len(all_locations) == 0:
            return values  # nothing to validate

        if any(location != all_locations[0] for location in all_locations):
            raise ValueError(
                "Location type mismatch: not all statements and declarations have the same location type"
            )

        if "location_type" not in values:
            values["location_type"] = all_locations[0]
        elif all_locations[0] != values["location_type"]:
            raise ValueError("Location type mismatch")
        return values


class NeighborLoop(Stmt):
    name: Str
    neighbors: NeighborChain
    body: BlockStmt

    @root_validator(pre=True)
    def check_location_type(cls, values):
        if values["neighbors"].elements[-1] != values["body"].location_type:
            raise ValueError("Location type mismatch")
        return values


class Access(Expr):
    name: Str


class FieldAccess(Access):
    primary: NeighborChain
    secondary: Optional[NeighborChain]

    @property
    def extent(self):
        return len(self.primary.elements) > 1


class VarAccess(Access):
    pass


# class IndexAccess(Access): # TODO(tehrengruber): use for TensorLocalVar
#    indices: List[int]


class LocationLocalIdAccess(Access):
    pass


class NeighborLoopLocationAccess(LocationLocalIdAccess):
    pass


class LocalFieldAccess(Access):
    location: LocationLocalIdAccess


class AssignStmt(Stmt):
    left: Access  # there are no local variables in gtir, only fields
    right: Expr

    @root_validator(pre=True)
    def check_location_type(cls, values):
        if values["left"].location_type != values["right"].location_type:
            raise ValueError("Location type mismatch")

        if "location_type" not in values:
            values["location_type"] = values["left"].location_type
        elif values["left"].location_type != values["location_type"]:
            raise ValueError("Location type mismatch")

        return values


class BinaryOp(Expr):
    op: common.BinaryOperator
    left: Expr
    right: Expr

    @root_validator(pre=True)
    def check_location_type(cls, values):
        if values["left"].location_type != values["right"].location_type:
            raise ValueError("Location type mismatch")

        if "location_type" not in values:
            values["location_type"] = values["left"].location_type
        elif values["left"].location_type != values["location_type"]:
            raise ValueError("Location type mismatch")

        return values


class VerticalDimension(Node):
    pass


class HorizontalDimension(Node):
    primary: common.LocationType
    secondary: Optional[NeighborChain]


class Dimensions(Node):
    horizontal: Optional[HorizontalDimension]
    vertical: Optional[VerticalDimension]
    # other: TODO


class UField(Node):
    name: Str
    vtype: common.DataType
    dimensions: Dimensions


class TemporaryField(UField):
    pass


class HorizontalLoop(Node):
    stmt: BlockStmt
    location_type: common.LocationType

    @root_validator(pre=True)
    def check_location_type(cls, values):
        # Don't infer here! The location type of the loop should always come from the frontend!
        if values["stmt"].location_type != values["location_type"]:
            raise ValueError("Location type mismatch")
        return values


class VerticalLoop(Node):
    # each statement inside a `with location_type` is interpreted as a full horizontal loop (see parallel model of SIR)
    horizontal_loops: List[HorizontalLoop]
    loop_order: common.LoopOrder


class Stencil(Node):
    vertical_loops: List[VerticalLoop]


class Computation(Node):
    name: Str
    params: List[UField]
    stencils: List[Stencil]
    declarations: List[TemporaryField]
