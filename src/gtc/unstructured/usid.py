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


class Stmt(Node):
    location_type: common.LocationType


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
        return "_".join([common.LocationType(loc).name.lower() for loc in self.elements])


class IndexAccess(Expr):
    name: Str # TODO(tehrengruber): Maybe IndexDecl in NeighborLoop?


class FieldAccess(Expr):
    name: Str  # symbol ref to SidCompositeEntry
    sid: Str  # symbol ref


class StaticArrayAccess(Expr):
    name: Str
    index: IndexAccess  # TODO(tehrengruber): Union[Index, IndexAccess]


class VarDecl(Stmt):
    name: Str


class ScalarVarDecl(VarDecl):
    init: Expr
    vtype: common.DataType


class StaticArrayDecl(VarDecl):
    init: List[Expr]
    vtype: common.DataType
    length: int

class Literal(Expr):
    value: Union[common.BuiltInLiteral, Str]
    vtype: common.DataType


class VarAccess(Expr):
    name: Str  # via symbol table
    dummy: Optional[
        Str
    ]  # to distinguish from FieldAccess, see https://github.com/eth-cscs/eve_toolchain/issues/34


class AssignStmt(Stmt):
    left: Union[FieldAccess, VarAccess]
    right: Expr

    @root_validator(pre=True)
    def check_location_type(cls, values):
        if values["left"].location_type != values["right"].location_type:
            raise ValueError("Location type mismatch")

        if "location_type" not in values:
            values["location_type"] = values["left"].location_type
        elif values["left"].location_type != values["location_type"]:
            raise ValueError("Location type mismatch")
        if values["left"].location_type == values["right"].location_type:
            values["location_type"] = values["left"].location_type

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

        if values["left"].location_type == values["right"].location_type:
            values["location_type"] = values["left"].location_type

        return values


class Connectivity(Node):
    name: Str  # symbol name
    chain: NeighborChain

    @property
    def neighbor_tbl_tag(self):
        return self.name + "_neighbor_tbl_tag"

    class Config(eve.concepts.FrozenModelConfig):
        pass

    # TODO see https://github.com/eth-cscs/eve_toolchain/issues/40
    def __hash__(self):
        return hash((self.name, self.chain))

    def __eq__(self, other):
        return self.name == other.name and self.chain == other.chain


class SidCompositeEntry(Node):
    name: Str  # symbol decl (TODO ensure field exists via symbol table)

    @property
    def tag_name(self):
        return self.name + "_tag"

    class Config(eve.concepts.FrozenModelConfig):
        pass

    # TODO see https://github.com/eth-cscs/eve_toolchain/issues/40
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class SidCompositeNeighborTableEntry(Node):
    connectivity: Str
    connectivity_deref_: Optional[
        Connectivity
    ]  # TODO temporary workaround for symbol tbl reference

    @property
    def tag_name(self):
        return self.connectivity_deref_.neighbor_tbl_tag

    class Config(eve.concepts.FrozenModelConfig):
        pass

    # TODO see https://github.com/eth-cscs/eve_toolchain/issues/40
    def __hash__(self):
        return hash(self.connectivity)

    def __eq__(self, other):
        return self.connectivity == other.connectivity


class SidComposite(Node):
    name: Str  # symbol
    location: NeighborChain
    entries: List[
        Union[SidCompositeEntry, SidCompositeNeighborTableEntry]
    ]  # TODO ensure tags are unique

    @property
    def with_connectivity(self) -> bool:
        for e in self.entries:
            if isinstance(e, SidCompositeNeighborTableEntry):
                return True
        return False

    # node private symbol table to entries
    @property
    def symbol_tbl(self):
        return {e.name: e for e in self.entries if isinstance(e, SidCompositeEntry)}

    @property
    def field_name(self):
        return self.name + "_fields"

    @property
    def ptr_name(self):
        return self.name + "_ptrs"

    @property
    def origin_name(self):
        return self.name + "_origins"

    @property
    def strides_name(self):
        return self.name + "_strides"

    @validator("entries")
    def not_empty(cls, entries):
        if len(entries) < 1:
            raise ValueError("SidComposite must contain at least one entry")
        return entries


class NeighborLoop(Stmt):
    body_location_type: common.LocationType
    body: List[Stmt]
    connectivity: Str  # symbol ref to Connectivity
    outer_sid: Str  # symbol ref to SidComposite where the neighbor tables lives (and sparse fields)
    sid: Optional[
        Str
    ]  # symbol ref to SidComposite where the fields of the loop body live (None if only sparse fields are accessed)
    iter_var: Str


class Kernel(Node):
    # location_type: common.LocationType
    name: Str  # symbol decl table
    connectivities: List[Connectivity]
    sids: List[SidComposite]

    primary_connectivity: Str  # symbol ref to the above
    primary_sid: Str  # symbol ref to the above
    ast: List[Stmt]

    # private symbol table
    @property
    def symbol_tbl(self):
        return {**{s.name: s for s in self.sids}, **{c.name: c for c in self.connectivities}}


class KernelCall(Node):
    name: Str  # symbol ref


class VerticalDimension(Node):
    pass


class UField(Node):
    name: Str
    vtype: common.DataType
    dimensions: List[Union[common.LocationType, NeighborChain, VerticalDimension]]  # Set?


class Temporary(UField):
    pass


class Computation(Node):
    name: Str
    parameters: List[UField]
    temporaries: List[Temporary]
    kernels: List[Kernel]
    ctrlflow_ast: List[KernelCall]
