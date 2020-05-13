# -*- coding: utf-8 -*-
# Eve toolchain

from devtools import debug  # noqa: F401

import eve  # noqa: F401
from gt_toolchain.unstructured import common, naive_codegen, sir, sir_to_naive
from gt_toolchain.unstructured.sir_passes.infer_local_variable_location_type import (
    InferLocalVariableLocationTypeTransformation,
)


statements = []
fields = []

# fields = [
#     sir_utils.make_field("S_MXX", sir_utils.make_field_dimensions_unstructured(
#         [SIR.LocationType.Value('Edge')], 1)),
#     sir_utils.make_field("S_MYY", sir_utils.make_field_dimensions_unstructured(
#         [SIR.LocationType.Value('Edge')], 1)),
#     sir_utils.make_field("zavgS_MXX", sir_utils.make_field_dimensions_unstructured(
#         [SIR.LocationType.Value('Edge')], 1)),
#     sir_utils.make_field("zavgS_MYY", sir_utils.make_field_dimensions_unstructured(
#         [SIR.LocationType.Value('Edge')], 1)),
#     sir_utils.make_field("pp", sir_utils.make_field_dimensions_unstructured(
#         [SIR.LocationType.Value('Vertex')], 1)),
#     sir_utils.make_field("pnabla_MXX", sir_utils.make_field_dimensions_unstructured(
#         [SIR.LocationType.Value('Vertex')], 1)),
#     sir_utils.make_field("pnabla_MYY", sir_utils.make_field_dimensions_unstructured(
#         [SIR.LocationType.Value('Vertex')], 1)),
#     sir_utils.make_field("vol", sir_utils.make_field_dimensions_unstructured(
#         [SIR.LocationType.Value('Vertex')], 1)),
#     sir_utils.make_field("sign", sir_utils.make_field_dimensions_unstructured(
#         [SIR.LocationType.Value('Vertex'), SIR.LocationType.Value('Edge')], 1)),
# ]
S_MXX = sir.Field(
    name="S_MXX",
    is_temporary=False,
    field_dimensions=sir.FieldDimensions(
        horizontal_dimension=sir.UnstructuredDimension(dense_location_type=sir.LocationType.Edge)
    ),
)
fields.append(S_MXX)
S_MYY = sir.Field(
    name="S_MYY",
    is_temporary=False,
    field_dimensions=sir.FieldDimensions(
        horizontal_dimension=sir.UnstructuredDimension(dense_location_type=sir.LocationType.Edge)
    ),
)
fields.append(S_MYY)
zavgS_MXX = sir.Field(
    name="zavgS_MXX",
    is_temporary=False,
    field_dimensions=sir.FieldDimensions(
        horizontal_dimension=sir.UnstructuredDimension(dense_location_type=sir.LocationType.Edge)
    ),
)
fields.append(zavgS_MXX)
zavgS_MYY = sir.Field(
    name="zavgS_MYY",
    is_temporary=False,
    field_dimensions=sir.FieldDimensions(
        horizontal_dimension=sir.UnstructuredDimension(dense_location_type=sir.LocationType.Edge)
    ),
)
fields.append(zavgS_MYY)
pp = sir.Field(
    name="pp",
    is_temporary=False,
    field_dimensions=sir.FieldDimensions(
        horizontal_dimension=sir.UnstructuredDimension(dense_location_type=sir.LocationType.Vertex)
    ),
)
fields.append(pp)
pnabla_MXX = sir.Field(
    name="pnabla_MXX",
    is_temporary=False,
    field_dimensions=sir.FieldDimensions(
        horizontal_dimension=sir.UnstructuredDimension(dense_location_type=sir.LocationType.Vertex)
    ),
)
fields.append(pnabla_MXX)
pnabla_MYY = sir.Field(
    name="pnabla_MYY",
    is_temporary=False,
    field_dimensions=sir.FieldDimensions(
        horizontal_dimension=sir.UnstructuredDimension(dense_location_type=sir.LocationType.Vertex)
    ),
)
fields.append(pnabla_MYY)
vol = sir.Field(
    name="vol",
    is_temporary=False,
    field_dimensions=sir.FieldDimensions(
        horizontal_dimension=sir.UnstructuredDimension(dense_location_type=sir.LocationType.Vertex)
    ),
)
fields.append(vol)
sign = sir.Field(
    name="sign",
    is_temporary=False,
    field_dimensions=sir.FieldDimensions(
        horizontal_dimension=sir.UnstructuredDimension(
            dense_location_type=sir.LocationType.Vertex,
            sparse_part=[sir.LocationType.Edge],  # TODO double-check sparse_part
        )
    ),
)
fields.append(sign)

# sir_utils.make_var_decl_stmt(
#     sir_utils.make_type(SIR.BuiltinType.Float),
#     "zavg",
#     0,
#     "=",
#     sir_utils.make_binary_operator(sir_utils.make_literal_access_expr(
#         "0.5", SIR.BuiltinType.Float), "*", sir_utils.make_reduction_over_neighbor_expr(
#         "+",
#         sir_utils.make_field_access_expr("pp"),
#         sir_utils.make_literal_access_expr(
#             "0.0", SIR.BuiltinType.Float),
#         lhs_location=SIR.LocationType.Value('Edge'),
#         rhs_location=SIR.LocationType.Value('Vertex')
#         # TODO assumed iflip==0, i.e. current implementation zbc = 1
#     ))

# ),
zavg_red = sir.ReductionOverNeighborExpr(
    op="+",
    rhs=sir.FieldAccessExpr(name="pp", vertical_offset=0, horizontal_offset=sir.ZeroOffset()),
    init=sir.LiteralAccessExpr(
        value="0.0", data_type=sir.BuiltinType(type_id=common.DataType.FLOAT32),
    ),
    chain=[sir.LocationType.Edge, sir.LocationType.Vertex],
)
zavg_mul = sir.BinaryOperator(
    op="*",
    left=sir.LiteralAccessExpr(
        value="0.5", data_type=sir.BuiltinType(type_id=common.DataType.FLOAT32),
    ),
    right=zavg_red,
)
zavg_decl = sir.VarDeclStmt(
    data_type=sir.Type(
        data_type=sir.BuiltinType(type_id=common.DataType.FLOAT32),
        is_const=False,
        is_volatile=False,
    ),
    name="zavg",
    dimension=0,
    op="=",
    init_list=[zavg_mul],
)

statements.append(zavg_decl)

#             sir_utils.make_assignment_stmt(sir_utils.make_field_access_expr(
#                 "zavgS_MXX"), sir_utils.make_binary_operator(sir_utils.make_field_access_expr("S_MXX"), "*", sir_utils.make_var_access_expr("zavg"))),
assign_zavgS_MXX = sir.ExprStmt(
    expr=sir.AssignmentExpr(
        left=sir.FieldAccessExpr(
            name="zavgS_MXX", vertical_offset=0, horizontal_offset=sir.ZeroOffset()
        ),
        op="=",
        right=sir.BinaryOperator(
            left=sir.FieldAccessExpr(
                name="S_MXX", vertical_offset=0, horizontal_offset=sir.ZeroOffset()
            ),
            op="*",
            right=sir.VarAccessExpr(name="zavg"),
        ),
    )
)
statements.append(assign_zavgS_MXX)
#             sir_utils.make_assignment_stmt(sir_utils.make_field_access_expr(
#                 "zavgS_MYY"), sir_utils.make_binary_operator(sir_utils.make_field_access_expr("S_MYY"), "*", sir_utils.make_var_access_expr("zavg"))),
assign_zavgS_MYY = sir.ExprStmt(
    expr=sir.AssignmentExpr(
        left=sir.FieldAccessExpr(
            name="zavgS_MYY", vertical_offset=0, horizontal_offset=sir.ZeroOffset()
        ),
        op="=",
        right=sir.BinaryOperator(
            left=sir.FieldAccessExpr(
                name="S_MYY", vertical_offset=0, horizontal_offset=sir.ZeroOffset()
            ),
            op="*",
            right=sir.VarAccessExpr(name="zavg"),
        ),
    )
)
statements.append(assign_zavgS_MYY)
#             # ===========================
# sir_utils.make_assignment_stmt(sir_utils.make_field_access_expr(
#     "pnabla_MXX"), sir_utils.make_reduction_over_neighbor_expr(
#         "+",
#         sir_utils.make_binary_operator(sir_utils.make_field_access_expr(
#             "zavgS_MXX"), "*", sir_utils.make_field_access_expr("sign")),
#         sir_utils.make_literal_access_expr(
#             "0.0", SIR.BuiltinType.Float),
#         lhs_location=SIR.LocationType.Value('Vertex'),
#         rhs_location=SIR.LocationType.Value('Edge')
# )),
assign_pnabla_MXX = sir.ExprStmt(
    expr=sir.AssignmentExpr(
        left=sir.FieldAccessExpr(
            name="pnabla_MXX", vertical_offset=0, horizontal_offset=sir.ZeroOffset()
        ),
        op="=",
        right=sir.ReductionOverNeighborExpr(
            op="+",
            rhs=sir.BinaryOperator(
                left=sir.FieldAccessExpr(
                    name="zavgS_MXX", vertical_offset=0, horizontal_offset=sir.ZeroOffset()
                ),
                op="*",
                right=sir.FieldAccessExpr(
                    name="sign", vertical_offset=0, horizontal_offset=sir.ZeroOffset()
                ),
            ),
            init=sir.LiteralAccessExpr(
                value="0.0", data_type=sir.BuiltinType(type_id=common.DataType.FLOAT32),
            ),
            chain=[sir.LocationType.Vertex, sir.LocationType.Edge],
        ),
    )
)
statements.append(assign_pnabla_MXX)
#             sir_utils.make_assignment_stmt(sir_utils.make_field_access_expr(
#                 "pnabla_MYY"), sir_utils.make_reduction_over_neighbor_expr(
#                     "+",
#                     sir_utils.make_binary_operator(sir_utils.make_field_access_expr(
#                         "zavgS_MYY"), "*", sir_utils.make_field_access_expr("sign")),
#                     sir_utils.make_literal_access_expr(
#                         "0.0", SIR.BuiltinType.Float),
#                     lhs_location=SIR.LocationType.Value('Vertex'),
#                     rhs_location=SIR.LocationType.Value('Edge')
#             )),
assign_pnabla_MYY = sir.ExprStmt(
    expr=sir.AssignmentExpr(
        left=sir.FieldAccessExpr(
            name="pnabla_MYY", vertical_offset=0, horizontal_offset=sir.ZeroOffset()
        ),
        op="=",
        right=sir.ReductionOverNeighborExpr(
            op="+",
            rhs=sir.BinaryOperator(
                left=sir.FieldAccessExpr(
                    name="zavgS_MYY", vertical_offset=0, horizontal_offset=sir.ZeroOffset()
                ),
                op="*",
                right=sir.FieldAccessExpr(
                    name="sign", vertical_offset=0, horizontal_offset=sir.ZeroOffset()
                ),
            ),
            init=sir.LiteralAccessExpr(
                value="0.0", data_type=sir.BuiltinType(type_id=common.DataType.FLOAT32),
            ),
            chain=[sir.LocationType.Vertex, sir.LocationType.Edge],
        ),
    )
)
statements.append(assign_pnabla_MYY)
#             # ===========================
#             # TODO pole correction for pnabla_MYY
#             # ===========================
#             sir_utils.make_assignment_stmt(sir_utils.make_field_access_expr(
#                 "pnabla_MXX"),
#                 sir_utils.make_binary_operator(sir_utils.make_field_access_expr(
#                     "pnabla_MXX"), "/", sir_utils.make_field_access_expr("vol")),
#             ),
assign_pnabla_MXX_vol = sir.ExprStmt(
    expr=sir.AssignmentExpr(
        left=sir.FieldAccessExpr(
            name="pnabla_MXX", vertical_offset=0, horizontal_offset=sir.ZeroOffset()
        ),
        op="=",
        right=sir.BinaryOperator(
            left=sir.FieldAccessExpr(
                name="pnabla_MXX", vertical_offset=0, horizontal_offset=sir.ZeroOffset()
            ),
            op="/",
            right=sir.FieldAccessExpr(
                name="vol", vertical_offset=0, horizontal_offset=sir.ZeroOffset()
            ),
        ),
    )
)
statements.append(assign_pnabla_MXX_vol)
#             sir_utils.make_assignment_stmt(sir_utils.make_field_access_expr(
#                 "pnabla_MYY"),
#                 sir_utils.make_binary_operator(sir_utils.make_field_access_expr(
#                     "pnabla_MYY"), "/", sir_utils.make_field_access_expr("vol")),
#             ),
#         ]
#     )
assign_pnabla_MYY_vol = sir.ExprStmt(
    expr=sir.AssignmentExpr(
        left=sir.FieldAccessExpr(
            name="pnabla_MYY", vertical_offset=0, horizontal_offset=sir.ZeroOffset()
        ),
        op="=",
        right=sir.BinaryOperator(
            left=sir.FieldAccessExpr(
                name="pnabla_MYY", vertical_offset=0, horizontal_offset=sir.ZeroOffset()
            ),
            op="/",
            right=sir.FieldAccessExpr(
                name="vol", vertical_offset=0, horizontal_offset=sir.ZeroOffset()
            ),
        ),
    )
)
statements.append(assign_pnabla_MYY_vol)


block = sir.BlockStmt(statements=statements)
ast = sir.AST(root=sir.BlockStmt(statements=statements))
vert_decl_stmt = sir.VerticalRegionDeclStmt(
    vertical_region=sir.VerticalRegion(
        ast=ast, interval=sir.Interval(), loop_order=common.LoopOrder.FORWARD
    )
)
ctrl_flow_ast = sir.AST(root=sir.BlockStmt(statements=[vert_decl_stmt]))
stencil = sir.Stencil(name="nabla", ast=ctrl_flow_ast, params=fields)

var_loc_type_inferred = InferLocalVariableLocationTypeTransformation.apply(stencil)
naive_ir = sir_to_naive.SirToNaive().visit(var_loc_type_inferred)
print(naive_codegen.NaiveCodeGenerator.apply(naive_ir))
