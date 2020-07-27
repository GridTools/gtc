# -*- coding: utf-8 -*-
# Eve toolchain

from devtools import debug  # noqa: F401

import eve  # noqa: F401
from gt_toolchain import common, sir
from gt_toolchain.unstructured import naive_codegen, sir_to_naive


field_acc_a = sir.FieldAccessExpr(
    name="field_a", vertical_offset=0, horizontal_offset=sir.ZeroOffset()
)
field_acc_b = sir.FieldAccessExpr(
    name="field_b", vertical_offset=0, horizontal_offset=sir.ZeroOffset()
)

assign_expr = sir.AssignmentExpr(left=field_acc_a, op="=", right=field_acc_b)
assign_expr_stmt = sir.ExprStmt(expr=assign_expr)
root = sir.BlockStmt(statements=[assign_expr_stmt])
ast = sir.AST(root=root)

vert_decl_stmt = sir.VerticalRegionDeclStmt(
    vertical_region=sir.VerticalRegion(
        ast=ast, interval=sir.Interval(), loop_order=common.LoopOrder.FORWARD
    )
)
ctrl_flow_ast = sir.AST(root=sir.BlockStmt(statements=[vert_decl_stmt]))

field_a = sir.Field(
    name="field_a",
    is_temporary=False,
    field_dimensions=sir.FieldDimensions(
        horizontal_dimension=sir.UnstructuredDimension(dense_location_type=sir.LocationType.Cell)
    ),
)
field_b = sir.Field(
    name="field_b",
    is_temporary=False,
    field_dimensions=sir.FieldDimensions(
        horizontal_dimension=sir.UnstructuredDimension(dense_location_type=sir.LocationType.Cell)
    ),
)

stencil = sir.Stencil(name="copy", ast=ctrl_flow_ast, params=[field_a, field_b])

debug(stencil)


naive = sir_to_naive.SirToNaive().visit(stencil)
cpp = naive_codegen.NaiveCodeGenerator.apply(naive)
print(cpp)
