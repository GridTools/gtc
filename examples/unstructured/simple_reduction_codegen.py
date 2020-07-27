# -*- coding: utf-8 -*-
# Eve toolchain

from devtools import debug  # noqa: F401

import eve  # noqa: F401
from gt_toolchain import common
from gt_toolchain.unstructured import naive
from gt_toolchain.unstructured.naive_codegen import NaiveCodeGenerator


# ### Reduce edge to node
#
# ```cpp
#   auto in_f = b.field("in_field", LocType::Edges);
#   auto out_f = b.field("out_field", LocType::Cells);
#   auto cnt = b.localvar("cnt", dawn::BuiltinTypeID::Integer);
#
#   auto stencil_instantiation = b.build(
#       "generated",
#       b.stencil(b.multistage(
#           LoopOrderKind::Parallel,
#           b.stage(LocType::Edges, b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
#                                              b.stmt(b.assignExpr(b.at(in_f), b.lit(10))))),
#           b.stage(b.doMethod(dawn::sir::Interval::Start, dawn::sir::Interval::End,
#                              b.stmt(b.assignExpr(
#                                  b.at(out_f), b.reduceOverNeighborExpr(
#                                                   Op::plus, b.at(in_f, HOffsetType::withOffset, 0),
#                                                   b.lit(0.), LocType::Cells, LocType::Edges))))))));
# ```

field_in = naive.UnstructuredField(
    name="in", location_type=naive.LocationType.Edge, data_type=common.DataType.FLOAT64
)
field_out = naive.UnstructuredField(
    name="out", location_type=naive.LocationType.Node, data_type=common.DataType.FLOAT64
)

zero = naive.LiteralExpr(
    value="0.0", data_type=common.DataType.FLOAT64, location_type=naive.LocationType.Node
)
in_acc = naive.FieldAccessExpr(name="in", offset=[True, 0], location_type=naive.LocationType.Edge)
red = naive.ReduceOverNeighbourExpr(
    operation=common.BinaryOperator.ADD,
    right=in_acc,
    init=zero,
    location_type=naive.LocationType.Node,
)

out_acc = naive.FieldAccessExpr(
    name="out", offset=[False, 0], location_type=naive.LocationType.Node
)

assign = naive.ExprStmt(expr=naive.AssignmentExpr(left=out_acc, right=red))

hori = naive.HorizontalLoop(ast=naive.BlockStmt(statements=[assign]),)
vert = naive.ForK(horizontal_loops=[hori], loop_order=common.LoopOrder.FORWARD)
sten = naive.Stencil(name="reduce_to_node", k_loops=[vert])

comp = naive.Computation(params=[field_in, field_out], stencils=[sten])

# ---------------------------

generated_code = NaiveCodeGenerator.apply(comp)
print(generated_code)

# try compile the generated code
# p = Popen(
#     [
#         "g++",
#         "-I",
#         "/home/vogtha/projects/toolchain/dawn/dawn/src",
#         "-x",
#         "c++",
#         "-c",
#         "-o",
#         "out.o",
#         "-",
#     ],
#     stdin=PIPE,
#     encoding="utf8",
# )
# p.communicate(input=generated_code)[0]
