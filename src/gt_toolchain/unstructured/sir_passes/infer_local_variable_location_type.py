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

import copy

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


class InferLocalVariableLocationType(NodeTranslator):
    """Returns a tree were local variables have location type set or raises an PassException if deduction failed.

    Usage: InferLocalVariableLocationType.apply(node)
    """

    @classmethod
    def apply(cls, root, **kwargs) -> Node:
        inferred_location = _AnalyseLocationTypes.apply(root)
        return cls().visit(root, inferred_location=inferred_location)

    def visit_VarDeclStmt(self, node: sir.VarDeclStmt, **kwargs):
        if node.name not in kwargs["inferred_location"] and node.location_type is None:
            raise PassException("Cannot deduce location type for {}".format(node.name))
        node.location_type = kwargs["inferred_location"][node.name]
        return node


class _AnalyseLocationTypes(NodeVisitor):
    """Analyse local variable usage and infers location type if possible.

    Result is a dict of variable names to LocationType.

    Usage: _AnalyseLocationTypes.apply(root: Node)

    Can deduce by assignments from:
     - Reductions (as they have a fixed location type)
     - Fields (as they have a fixed location type)
     - Variables recursively (by building a dependency tree)
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.inferred_location = dict()
        self.cur_var = None
        self.var_dependencies = dict()  # depender -> set(dependees)
        # TODO replace naive symbol table
        self.sir_stencil_params = {}

    @classmethod
    def apply(cls, root, **kwargs) -> Node:
        root_copy = copy.deepcopy(root)

        instance = cls()
        instance.visit(root_copy)

        for var_name in list(instance.inferred_location.keys()):
            instance.propagate_location_type(var_name)

        return instance.inferred_location

    def propagate_location_type(self, var_name: str):
        for dependee in self.var_dependencies[var_name]:
            if (
                dependee in self.inferred_location
                and self.inferred_location[dependee] != self.inferred_location[var_name]
            ):
                raise PassException("Incompatible location type detected for {}".format(dependee))
            self.inferred_location[dependee] = self.inferred_location[var_name]
            self.propagate_location_type(dependee)

    def set_location_type(self, cur_var_name: str, location_type: sir.LocationType):
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
        if self.cur_var:
            new_type = self.sir_stencil_params[
                node.name
            ].field_dimensions.horizontal_dimension.dense_location_type  # TODO use symbol table
            self.set_location_type(self.cur_var.name, new_type)

    def visit_VarAccessExpr(self, node: sir.VarAccessExpr, **kwargs):
        if self.cur_var:
            # rhs of assignment/declaration
            if node.name not in self.var_dependencies:
                raise RuntimeError("{node.name} was not declared")
            self.var_dependencies[node.name].add(self.cur_var.name)

    def visit_VarDeclStmt(self, node: sir.VarDeclStmt, **kwargs):
        if node.name in self.var_dependencies:
            raise RuntimeError("Redeclaration of variable")  # TODO symbol table will take care
        else:
            self.var_dependencies[node.name] = set()

        self.cur_var = node

        assert len(node.init_list) == 1
        self.visit(node.init_list[0])

        self.cur_var = None

    def visit_ReductionOverNeighborExpr(self, node: sir.ReductionOverNeighborExpr, **kwargs):
        if self.cur_var:
            self.set_location_type(self.cur_var.name, node.chain[0])

    def visit_AssignmentExpr(self, node: sir.AssignmentExpr, **kwargs):
        if isinstance(node.left, sir.VarAccessExpr):
            if self.cur_var:
                raise RuntimeError(
                    "Variable assignment inside rhs of variable assignment is not supported."
                )
            self.cur_var = node.left

            self.visit(node.right)

            self.cur_var = None
