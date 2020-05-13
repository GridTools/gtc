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

import eve  # noqa: F401
from eve.core import Node, NodeTranslator, NodeVisitor

from .. import sir


class PassException(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return "PassException, {0} ".format(self.message)
        else:
            return "PassException has been raised"


class InferLocalVariableLocationTypeTransformation(NodeTranslator):
    """Returns a tree were local variables have location type set or raises an PassException if deduction failed.

    Usage: InferLocalVariableLocationType.apply(node)
    """

    @classmethod
    def apply(cls, root, **kwargs) -> Node:
        inferred_location = _LocationTypeAnalysis.apply(root)
        return cls().visit(root, inferred_location=inferred_location)

    def visit_VarDeclStmt(self, node: sir.VarDeclStmt, *, inferred_location, **kwargs):
        if node.name not in inferred_location and node.location_type is None:
            raise PassException("Cannot deduce location type for {}".format(node.name))
        node.location_type = inferred_location[node.name]
        return node


class _LocationTypeAnalysis(NodeVisitor):
    """Analyse local variable usage and infers location type if possible.

    Result is a dict of variable names to LocationType.

    Usage: _LocationTypeAnalysis.apply(root: Node)

    Can deduce by assignments from:
     - Reductions (as they have a fixed location type)
     - Fields (as they have a fixed location type)
     - Variables recursively (by building a dependency tree)
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.is_root = True

        self.inferred_location = {}
        self.var_dependencies = {}  # depender -> set(dependees)
        # TODO replace naive symbol table
        self.sir_stencil_params = {}

    @classmethod
    def apply(cls, root, **kwargs) -> Node:
        return cls().visit(root, **kwargs)

    # entrypoint, do postprocessing of the result
    def visit(self, node: Node, **kwargs):
        if self.is_root:
            self.is_root = False
            super().visit(node, **kwargs)

            # propagate the inferred location type to dependencies
            original_inferred_location_keys = list(self.inferred_location.keys())
            for var_name in original_inferred_location_keys:
                self._propagate_location_type(var_name)

            result = dict(self.inferred_location)
            self.__init__()  # reset visitor state
            return result
        else:
            super().visit(node, **kwargs)

    def _propagate_location_type(self, var_name: str):
        for dependee in self.var_dependencies[var_name]:
            if (
                dependee in self.inferred_location
                and self.inferred_location[dependee] != self.inferred_location[var_name]
            ):
                raise PassException("Incompatible location type detected for {}".format(dependee))
            self.inferred_location[dependee] = self.inferred_location[var_name]
            self._propagate_location_type(dependee)

    def _set_location_type(self, cur_var_name: str, location_type: sir.LocationType):
        if cur_var_name in self.inferred_location:
            if self.inferred_location[cur_var_name] != location_type:
                raise RuntimeError("Incompatible location types deduced for {cur_var_name}")
        else:
            self.inferred_location[cur_var_name] = location_type

    def visit_Stencil(self, node: sir.Stencil, **kwargs):
        for f in node.params:
            self.sir_stencil_params[f.name] = f
        self.visit(node.ast)

    def visit_FieldAccessExpr(self, node: sir.FieldAccessExpr, **kwargs):
        if "cur_var" in kwargs:
            new_type = self.sir_stencil_params[
                node.name
            ].field_dimensions.horizontal_dimension.dense_location_type  # TODO use symbol table
            self._set_location_type(kwargs["cur_var"].name, new_type)

    def visit_VarAccessExpr(self, node: sir.VarAccessExpr, **kwargs):
        if "cur_var" in kwargs:
            # rhs of assignment/declaration
            if node.name not in self.var_dependencies:
                raise RuntimeError("{node.name} was not declared")
            self.var_dependencies[node.name].add(kwargs["cur_var"].name)

    def visit_VarDeclStmt(self, node: sir.VarDeclStmt, **kwargs):
        if node.name in self.var_dependencies:
            raise RuntimeError("Redeclaration of variable")  # TODO symbol table will take care
        else:
            self.var_dependencies[node.name] = set()

        assert len(node.init_list) == 1
        self.visit(node.init_list[0], cur_var=node)

    def visit_ReductionOverNeighborExpr(self, node: sir.ReductionOverNeighborExpr, **kwargs):
        if "cur_var" in kwargs:
            self._set_location_type(kwargs["cur_var"].name, node.chain[0])

    def visit_AssignmentExpr(self, node: sir.AssignmentExpr, **kwargs):
        if isinstance(node.left, sir.VarAccessExpr):
            if "cur_var" in kwargs:
                raise RuntimeError(
                    "Variable assignment inside rhs of variable assignment is not supported."
                )

            self.visit(node.right, cur_var=node.left)
