# -*- coding: utf-8 -*-
# Eve toolchain

from devtools import debug  # noqa: F401

import eve  # noqa: F401
from gt_toolchain.unstructured import common, sir
from gt_toolchain.unstructured.sir_passes.infer_local_variable_location_type import (
    InferLocalVariableLocationType,
)


float_t = sir.BuiltinType(type_id=common.DataType.FLOAT32)

literal = sir.LiteralAccessExpr(value="0", data_type=float_t)

my_field_acc = sir.FieldAccessExpr(
    name="my_field", vertical_offset=0, horizontal_offset=sir.ZeroOffset()
)
local_var = sir.VarDeclStmt(
    data_type=sir.Type(data_type=float_t, is_const=False, is_volatile=False),
    name="local_var",
    op="=",
    dimension=1,
    init_list=[literal],
)
local_var2 = sir.VarDeclStmt(
    data_type=sir.Type(data_type=float_t, is_const=False, is_volatile=False),
    name="local_var2",
    op="=",
    dimension=1,
    init_list=[literal],
)

local_var_acc = sir.VarAccessExpr(name="local_var")
local_var2_acc = sir.VarAccessExpr(name="local_var2")

assign_expr = sir.AssignmentExpr(left=local_var_acc, op="=", right=my_field_acc)
assign_expr_stmt = sir.ExprStmt(expr=assign_expr)

# =======

red = sir.ReductionOverNeighborExpr(
    op="+", rhs=my_field_acc, init=literal, chain=[sir.LocationType.Edge, sir.LocationType.Cell],
)
assign_expr_stmt2 = sir.ExprStmt(expr=sir.AssignmentExpr(left=local_var2_acc, op="=", right=red))

# =======
local_var3 = sir.VarDeclStmt(
    data_type=sir.Type(data_type=float_t, is_const=False, is_volatile=False),
    name="local_var3",
    op="=",
    dimension=1,
    init_list=[local_var2_acc],
)
# =======


root = sir.BlockStmt(
    statements=[local_var, local_var2, assign_expr_stmt, assign_expr_stmt2, local_var3]
)
ast = sir.AST(root=root)

vert_decl_stmt = sir.VerticalRegionDeclStmt(
    vertical_region=sir.VerticalRegion(
        ast=ast, interval=sir.Interval(), loop_order=common.LoopOrder.FORWARD
    )
)
ctrl_flow_ast = sir.AST(root=sir.BlockStmt(statements=[vert_decl_stmt]))

my_field = sir.Field(
    name="my_field",
    is_temporary=False,
    field_dimensions=sir.FieldDimensions(
        horizontal_dimension=sir.UnstructuredDimension(dense_location_type=sir.LocationType.Cell)
    ),
)

stencil = sir.Stencil(name="copy", ast=ctrl_flow_ast, params=[my_field])

var_loc_type = InferLocalVariableLocationType.apply(stencil)
debug(var_loc_type)
