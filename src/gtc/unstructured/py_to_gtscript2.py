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

import ast
import inspect
import textwrap
from typing import Sequence

from toolz.functoolz import compose

import eve
from gtc.unstructured import gtscript_ast2


class _Visitor(ast.NodeVisitor):
    def _visit_iterable(self, src):
        return tuple(self.visit(e) for e in src)

    def visit_Tuple(self, src: ast.Tuple):
        return tuple(self.visit(e) for e in src.elts)

    def visit_Name(self, src: ast.Name):
        return gtscript_ast2.Name(name=src.id)

    def visit_Subscript(self, src: ast.Subscript):
        return gtscript_ast2.Subscript(
            value=self.visit(src.value), index=self.visit(src.slice.value)
        )

    def visit_withitem(self, src: ast.withitem):
        return gtscript_ast2.WithItem(
            expr=self.visit(src.context_expr),
            var=self.visit(src.optional_vars) if src.optional_vars else None,
        )

    def visit_With(self, src: ast.With):
        return gtscript_ast2.With(
            items=self._visit_iterable(src.items), body=self._visit_iterable(src.body)
        )

    def visit_Call(self, src: ast.Call):
        return gtscript_ast2.Call(func=self.visit(src.func), args=self._visit_iterable(src.args))

    def visit_BinOp(self, src: ast.BinOp):
        return gtscript_ast2.BinOp(
            left=self.visit(src.left), op=self.visit(src.op), right=self.visit(src.right)
        )

    def visit_Div(self, src: ast.Div):
        return "/"

    def visit_Mult(self, src: ast.Mult):
        return "*"

    def visit_Add(self, src: ast.Div):
        return "+"

    def visit_Sub(self, src: ast.Mult):
        return "-"

    def visit_Constant(self, src: ast.Constant):
        return gtscript_ast2.Constant(value=src.value)

    def visit_GeneratorExp(self, src: ast.GeneratorExp):
        return gtscript_ast2.GeneratorExp(
            elt=self.visit(src.elt), generators=self._visit_iterable(src.generators)
        )

    def visit_comprehension(self, src: ast.comprehension):
        return gtscript_ast2.Comprehension(
            target=self.visit(src.target), iter_=self.visit(src.iter)
        )

    def visit_Assign(self, src: ast.Assign):
        return gtscript_ast2.Assign(
            targets=self._visit_iterable(src.targets), value=self.visit(src.value)
        )

    def visit_FunctionDef(self, src: ast.FunctionDef):
        return gtscript_ast2.Function(name=src.name, body=self._visit_iterable(src.body))


def _flatten(ll):
    return tuple(e for l in ll for e in l)


class _FlattenWiths(eve.NodeTranslator):
    def _process_statements(self, src: Sequence[gtscript_ast2.Stmt]):
        return (
            _flatten(
                self.visit(s, items=()) if isinstance(s, gtscript_ast2.With) else (self.visit(s),)
                for s in src
            )
            if any(isinstance(s, gtscript_ast2.With) for s in src)
            else tuple(self.visit(s) for s in src)
        )

    def visit_Function(self, src: gtscript_ast2.Function):
        return gtscript_ast2.Function(name=src.name, body=self._process_statements(src.body))

    def visit_With(self, src: gtscript_ast2.With, items: Sequence[gtscript_ast2.WithItem]):
        items = items + src.items
        return (
            _flatten(self.visit(s, items=items) for s in src.body)
            if all(isinstance(s, gtscript_ast2.With) for s in src.body)
            else (gtscript_ast2.With(items=items, body=self._process_statements(src.body)),)
        )


transform = compose(
    _FlattenWiths().visit,
    _Visitor().visit,
    lambda x: x.body[0],
    ast.parse,
    textwrap.dedent,
    inspect.getsource,
)
