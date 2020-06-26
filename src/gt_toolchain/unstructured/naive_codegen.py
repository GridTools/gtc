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


from types import MappingProxyType
from typing import ClassVar, Mapping

from mako import template as mako_tpl

from eve import codegen
from gt_toolchain import common

from .naive import LocationType


class NaiveCodeGenerator(codegen.TemplatedGenerator):
    LOCATION_TYPE_TO_STR: ClassVar[Mapping[LocationType, Mapping[str, str]]] = MappingProxyType(
        {
            LocationType.Node: MappingProxyType({"singular": "vertex", "plural": "vertices"}),
            LocationType.Edge: MappingProxyType({"singular": "edge", "plural": "edges"}),
            LocationType.Face: MappingProxyType({"singular": "cell", "plural": "cells"}),
        }
    )

    DATA_TYPE_TO_STR: ClassVar[Mapping[LocationType, str]] = MappingProxyType(
        {
            common.DataType.BOOLEAN: "bool",
            common.DataType.INT32: "int",
            common.DataType.UINT32: "unsigned_int",
            common.DataType.FLOAT32: "float",
            common.DataType.FLOAT64: "double",
        }
    )

    class Templates:
        Node = mako_tpl.Template("${_this_node.__class__.__name__.upper()}")  # only for testing

        UnstructuredField = mako_tpl.Template(
            """<%
    loc_type = _this_generator.LOCATION_TYPE_TO_STR[_this_node.location_type]["singular"]
    data_type = _this_generator.DATA_TYPE_TO_STR[_this_node.data_type]
    sparseloc = "sparse_" if _this_node.sparse_location_type else ""
%>
  dawn::${ sparseloc }${ loc_type }_field_t<LibTag, ${ data_type }>& ${ name };"""
        )

        FieldAccessExpr = mako_tpl.Template(
            """<%
    sparse_index = "m_sparse_dimension_idx, " if _this_node.is_sparse else ""
    field_acc_itervar = outer_iter_var if _this_node.is_sparse else iter_var
%>${ name }(deref(LibTag{}, ${ field_acc_itervar }), ${ sparse_index } k)"""
        )

        AssignmentExpr = "{left} = {right}"

        VarAccessExpr = "{name}"

        BinaryOp = "{left} {op} {right}"

        ExprStmt = "\n{expr};"

        VarDeclStmt = mako_tpl.Template(
            "\n${ _this_generator.DATA_TYPE_TO_STR[_this_node.data_type] } ${ name };"
        )

        TemporaryFieldDeclStmt = mako_tpl.Template(
            """using dawn::allocateEdgeField;
            auto ${ name } = allocate${ _this_generator.LOCATION_TYPE_TO_STR[_this_node.location_type]['singular'].capitalize() }Field<${ _this_generator.DATA_TYPE_TO_STR[_this_node.data_type] }>(mesh);"""
        )

        ForK = mako_tpl.Template(
            """<%
    if _this_node.loop_order == _this_module.common.LoopOrder.FORWARD:
        k_init = '0'
        k_cond = 'k < k_size'
        k_step = '++k'
    else:
        k_init = 'k_size -1'
        k_cond = 'k >= 0'
        k_step = '--k'
%>for (int k = ${k_init}; ${k_cond}; ${k_step}) {
    int m_sparse_dimension_idx;
    ${ "".join(horizontal_loops) }\n}"""
        )

        HorizontalLoop = mako_tpl.Template(
            """<%
    loc_type = _this_generator.LOCATION_TYPE_TO_STR[_this_node.location_type]['plural'].title()
%>for(auto const & t: get${ loc_type }(LibTag{}, mesh)) ${ ast }"""
        )

        BlockStmt = mako_tpl.Template("{${ ''.join(statements) }\n}")

        ReduceOverNeighbourExpr = mako_tpl.Template(
            """<%
    right_loc_type = _this_generator.LOCATION_TYPE_TO_STR[_this_node.right_location_type]["singular"].title()
    loc_type = _this_generator.LOCATION_TYPE_TO_STR[_this_node.location_type]["singular"].title()
%>(m_sparse_dimension_idx=0,reduce${ right_loc_type }To${ loc_type }(mesh, ${ outer_iter_var }, ${ init }, [&](auto& lhs, auto const& ${ iter_var }) {
lhs ${ operation }= ${ right };
m_sparse_dimension_idx++;
return lhs;
}))"""
        )

        LiteralExpr = mako_tpl.Template(
            "(${ _this_generator.DATA_TYPE_TO_STR[_this_node.data_type] })${ value }"
        )

        Stencil = mako_tpl.Template(
            """
void ${name}() {
using dawn::deref;

${ "\\n".join(declarations) }

${ "".join(k_loops) }
}
"""
        )

        Computation = mako_tpl.Template(
            """<%
    stencil_calls = '\\n'.join("{name}();".format(name=s.name) for s in _this_node.stencils)
    ctor_field_params = ', '.join(
        'dawn::{sparse_loc}{loc_type}_field_t<LibTag, {data_type}>& {name}'.format(
            loc_type=_this_generator.LOCATION_TYPE_TO_STR[p.location_type]['singular'],
            name=p.name,
            data_type=_this_generator.DATA_TYPE_TO_STR[p.data_type],
            sparse_loc="sparse_" if p.sparse_location_type else ""
        )
        for p in _this_node.params
    )
    ctor_field_initializers = ', '.join(
        '{name}({name})'.format(name=p.name) for p in _this_node.params
    )
%>#define DAWN_GENERATED 1
#define DAWN_BACKEND_T CXXNAIVEICO
#include <driver-includes/unstructured_interface.hpp>
namespace dawn_generated {
namespace cxxnaiveico {
template <typename LibTag>
class generated {
private:
  dawn::mesh_t<LibTag>& mesh;
  int const k_size;

${ ''.join(params) }
${ ''.join(stencils) }

public:
generated(dawn::mesh_t<LibTag>& mesh, int k_size, ${ ctor_field_params }): mesh(mesh), k_size(k_size), ${ ctor_field_initializers } {}

void run() {
    ${ stencil_calls }
}
};
}
}

"""
        )

    @classmethod
    def apply(cls, root, **kwargs) -> str:
        generated_code = super().apply(root, **kwargs)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code

    def visit_HorizontalLoop(self, node, **kwargs) -> str:
        return self.generic_visit(node, iter_var="t", **kwargs)

    def visit_ReduceOverNeighbourExpr(self, node, *, iter_var, **kwargs) -> str:
        outer_iter_var = iter_var
        return self.generic_visit(node, outer_iter_var=outer_iter_var, iter_var="redIdx", **kwargs,)
