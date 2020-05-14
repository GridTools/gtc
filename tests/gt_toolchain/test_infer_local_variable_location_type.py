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

import pytest

from gt_toolchain.unstructured import common, sir
from gt_toolchain.unstructured.sir_passes.infer_local_variable_location_type import (
    AnalysisException,
    InferLocalVariableLocationTypeTransformation,
)

from .util import FindNodes


float_type = sir.BuiltinType(type_id=common.DataType.FLOAT32)


def make_literal(value="0", dtype=float_type):
    return sir.LiteralAccessExpr(value=value, data_type=dtype)


default_literal = make_literal()


def make_var_decl(name: str, dtype=float_type, init=default_literal):
    return sir.VarDeclStmt(
        data_type=sir.Type(data_type=dtype, is_const=False, is_volatile=False),
        name=name,
        op="=",
        dimension=1,
        init_list=[init],
    )


def make_var_acc(name):
    return sir.VarAccessExpr(name=name)


def make_assign_to_local_var(local_var_name: str, rhs):
    return sir.ExprStmt(
        expr=sir.AssignmentExpr(left=make_var_acc(local_var_name), op="=", right=rhs)
    )


def make_field_acc(name):
    return sir.FieldAccessExpr(name=name, vertical_offset=0, horizontal_offset=sir.ZeroOffset())


def make_field(name, location_type=sir.LocationType.Cell):
    return sir.Field(
        name=name,
        is_temporary=False,
        field_dimensions=sir.FieldDimensions(
            horizontal_dimension=sir.UnstructuredDimension(dense_location_type=location_type)
        ),
    )


def make_stencil(fields, statements):
    root = sir.BlockStmt(statements=statements)
    ast = sir.AST(root=root)

    vert_decl_stmt = sir.VerticalRegionDeclStmt(
        vertical_region=sir.VerticalRegion(
            ast=ast, interval=sir.Interval(), loop_order=common.LoopOrder.FORWARD
        )
    )
    ctrl_flow_ast = sir.AST(root=sir.BlockStmt(statements=[vert_decl_stmt]))

    return sir.Stencil(name="stencil", ast=ctrl_flow_ast, params=fields)


class TestInferLocalVariableLocationType:
    def test_simple_assignment(self):
        stencil = make_stencil(
            fields=[make_field("field")],
            statements=[
                make_var_decl(name="var"),
                make_assign_to_local_var("var", make_field_acc("field")),
            ],
        )

        result = InferLocalVariableLocationTypeTransformation.apply(stencil)

        vardecl = FindNodes.by_type(sir.VarDeclStmt, result)[0]
        assert vardecl.location_type == sir.LocationType.Cell

    def test_reduction(self):
        stencil = make_stencil(
            fields=[],
            statements=[
                make_var_decl(name="var"),
                make_assign_to_local_var(
                    "var",
                    sir.ReductionOverNeighborExpr(
                        op="+",
                        rhs=make_literal(),
                        init=make_literal(),
                        chain=[sir.LocationType.Edge, sir.LocationType.Cell],
                    ),
                ),
            ],
        )

        result = InferLocalVariableLocationTypeTransformation.apply(stencil)

        vardecl = FindNodes.by_type(sir.VarDeclStmt, result)[0]
        assert vardecl.location_type == sir.LocationType.Edge

    def test_chain_assignment(self):
        stencil = make_stencil(
            fields=[make_field("field")],
            statements=[
                make_var_decl(name="var"),
                make_assign_to_local_var("var", make_field_acc("field")),
                make_var_decl(name="another_var", dtype=float_type, init=make_var_acc("var")),
            ],
        )

        result = InferLocalVariableLocationTypeTransformation.apply(stencil)

        vardecls = FindNodes.by_type(sir.VarDeclStmt, result)
        assert len(vardecls) == 2
        for vardecl in vardecls:
            assert vardecl.location_type == sir.LocationType.Cell

    def test_var_type_not_deducible(self):
        stencil = make_stencil(fields=[], statements=[make_var_decl(name="var")])

        with pytest.raises(AnalysisException):
            InferLocalVariableLocationTypeTransformation.apply(stencil)

    def test_cyclic_assignment(self):
        stencil = make_stencil(
            fields=[],
            statements=[
                make_var_decl(name="var"),
                make_var_decl(name="var2"),
                make_assign_to_local_var("var", make_var_acc("var2")),
                make_assign_to_local_var("var2", make_var_acc("var")),
            ],
        )

        with pytest.raises(AnalysisException):
            InferLocalVariableLocationTypeTransformation.apply(stencil)

    def test_incompatible_location(self):
        stencil = make_stencil(
            fields=[make_field("field_edge", sir.LocationType.Edge), make_field("field_cell")],
            statements=[
                make_var_decl(name="var_edge"),
                make_var_decl(name="var_cell"),
                make_assign_to_local_var("var_edge", make_field_acc("field_edge")),
                make_assign_to_local_var("var_cell", make_field_acc("field_cell")),
                make_assign_to_local_var("var_cell", make_var_acc("var_edge")),
            ],
        )

        with pytest.raises(AnalysisException):
            InferLocalVariableLocationTypeTransformation.apply(stencil)
