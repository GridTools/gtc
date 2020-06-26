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

# --- LLVM-lit test definition
#
# RUN: python -u %s | filecheck %s
# ---

from eve import codegen
from gt_toolchain import common
from gt_toolchain.unstructured import naive
from gt_toolchain.unstructured.naive_codegen import NaiveCodeGenerator


# How to write a test:
# - Call make_test(<node>, [test_name=<test_name>], <kwargs>)
#   This will produce output:
#     <node>[.test_name]
#     <generated_code>
# - Test the generated code with filecheck by using the following pattern (replace "!" with ":")\
#     CHECK! <node>[.test_name]
#     CHECK-NEXT! <pattern to test>
#     CHECK-NEXT! etc...
# - Run this file with `lit <filename>`


def make_test(node, test_name="", **kwargs):
    print(
        "{test_name}\n{generated_code}".format(
            test_name=".".join(filter(None, [node.__class__.__name__, test_name])),
            generated_code=codegen.format_source(
                "cpp", NaiveCodeGenerator().visit(node, **kwargs), style="LLVM"
            ),
        )
    )


# CHECK: UnstructuredField
# CHECK-NEXT:
# CHECK-NEXT: dawn::edge_field_t<LibTag, double> &in;
make_test(
    naive.UnstructuredField(
        name="in", location_type=naive.LocationType.Edge, data_type=common.DataType.FLOAT64
    ),
)

# CHECK: LiteralExpr
# CHECK-NEXT: (double)0.0
make_test(
    naive.LiteralExpr(
        value="0.0", data_type=common.DataType.FLOAT64, location_type=naive.LocationType.Node
    )
)

# CHECK: FieldAccessExpr
# CHECK-NEXT: field(deref(LibTag{}, iter),  k)
make_test(
    naive.FieldAccessExpr(name="field", offset=[True, 0], location_type=naive.LocationType.Edge),
    iter_var="iter",
)

# CHECK: ReduceOverNeighbourExpr.dense_access
# CHECK-NEXT: (m_sparse_dimension_idx = 0,
# CHECK-NEXT: reduceEdgeToVertex(mesh, iter, EXPR, [&](auto &lhs, auto const &redIdx) {
# CHECK-NEXT:                       lhs += field({{.*}}redIdx{{.*}} k);
# CHECK-NEXT:                       m_sparse_dimension_idx++;
# CHECK-NEXT:                       return lhs;
# CHECK-NEXT:                     }))
make_test(
    naive.ReduceOverNeighbourExpr(
        operation=common.BinaryOperator.ADD,
        right=naive.FieldAccessExpr(
            name="field", offset=[True, 0], location_type=naive.LocationType.Edge
        ),
        init=naive.Expr(location_type=naive.LocationType.Edge),
        location_type=naive.LocationType.Node,
    ),
    "dense_access",
    iter_var="iter",
)

# CHECK: ReduceOverNeighbourExpr.sparse_access
# CHECK-NEXT: (m_sparse_dimension_idx = 0,
# CHECK-NEXT:  reduceEdgeToVertex(mesh, iter, EXPR, [&](auto &lhs, auto const &redIdx) {
# CHECK-NEXT:                       lhs += field({{.*}}iter{{.*}} m_sparse_dimension_idx{{.*}},{{.*}}k);
# CHECK-NEXT:                       m_sparse_dimension_idx++;
# CHECK-NEXT:                       return lhs;
# CHECK-NEXT:                     }))
make_test(
    naive.ReduceOverNeighbourExpr(
        operation=common.BinaryOperator.ADD,
        right=naive.FieldAccessExpr(
            name="field", offset=[True, 0], location_type=naive.LocationType.Edge, is_sparse=True
        ),
        init=naive.Expr(location_type=naive.LocationType.Edge),
        location_type=naive.LocationType.Node,
    ),
    "sparse_access",
    iter_var="iter",
)

# CHECK: ExprStmt
# CHECK-NEXT:
# CHECK-NEXT: EXPR = EXPR;
make_test(
    naive.ExprStmt(
        expr=naive.AssignmentExpr(
            left=naive.Expr(location_type=naive.LocationType.Edge),
            right=naive.Expr(location_type=naive.LocationType.Edge),
        )
    )
)
