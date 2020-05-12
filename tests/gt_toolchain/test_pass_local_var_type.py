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

from typing import Any

from eve import core
from gt_toolchain.unstructured import common, sir
from gt_toolchain.unstructured.sir_passes.pass_local_var_type import PassLocalVarType


# import pytest  # type: ignore


float_type = sir.BuiltinType(type_id=common.DataType.FLOAT32)


def make_literal(value="0", dtype=float_type):
    return sir.LiteralAccessExpr(value=value, data_type=dtype)


def make_var_decl(
    name: str, dtype=float_type, init=make_literal()
):  # TODO how to init with function call?
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


def make_field(name):
    return sir.Field(
        name=name,
        is_temporary=False,
        field_dimensions=sir.FieldDimensions(
            horizontal_dimension=sir.UnstructuredDimension(
                dense_location_type=sir.LocationType.Cell
            )
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


class FindNode(core.NodeVisitor):
    def __init__(self, **kwargs):
        self.result = []

    def visit(self, node: core.Node, **kwargs) -> Any:
        if isinstance(node, kwargs["search_node_type"]):
            self.result.append(node)
        self.generic_visit(node, **kwargs)

    @classmethod
    def byType(cls, search_node_type, node: core.Node, **kwargs):
        visitor = FindNode()
        visitor.visit(node, search_node_type=search_node_type)
        return visitor.result


class TestPassLocalVarType:
    def test_simple_assignment(self):
        stencil = make_stencil(
            fields=[make_field("my_field")],
            statements=[
                make_var_decl(name="local_var"),
                make_assign_to_local_var("local_var", make_field_acc("my_field")),
            ],
        )
        PassLocalVarType.apply(stencil)
        for s in FindNode.byType(sir.VarDeclStmt, stencil):
            print(s.name)

    def test_simple_assignment2(self):
        statements = [
            make_var_decl(name="local_var"),
            make_assign_to_local_var("local_var", make_field_acc("my_field")),
            # =======
            make_var_decl(name="local_var2"),
            make_assign_to_local_var(
                "local_var2",
                sir.ReductionOverNeighborExpr(
                    op="+",
                    rhs=make_field_acc("my_field"),
                    init=make_literal(),
                    chain=[sir.LocationType.Edge, sir.LocationType.Cell],
                ),
            ),
            # =======
            make_var_decl(name="local_var3", dtype=float_type, init=make_var_acc("local_var2")),
        ]

        fields = [make_field("my_field")]

        stencil = make_stencil(fields, statements)
        PassLocalVarType.apply(stencil)


if __name__ == "__main__":
    TestPassLocalVarType().test_simple_assignment()
