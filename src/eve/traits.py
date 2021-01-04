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

"""Definitions of Trait classes."""


from __future__ import annotations

import pydantic

from . import concepts, visitors
from .type_definitions import SymbolName
from .typingx import Any, Dict, Type


class _CollectSymbols(visitors.NodeVisitor):
    def __init__(self) -> None:
        self.collected: Dict[str, Any] = {}

    def visit_Node(self, node: concepts.Node) -> None:
        for name, metadata in node.__node_children__.items():
            if isinstance(metadata["definition"].type_, type) and issubclass(
                metadata["definition"].type_, SymbolName
            ):
                self.collected[getattr(node, name)] = node

            if isinstance(metadata["definition"].type_, type) and not issubclass(
                metadata["definition"].type_, SymbolTableTrait
            ):
                self.visit(getattr(node, name))

    @classmethod
    def apply(cls, node: concepts.TreeNode) -> Dict[str, Any]:
        instance = cls()
        instance.generic_visit(node)
        return instance.collected


class SymbolTableTrait(concepts.Model):
    symtable_: Dict[str, Any] = pydantic.Field(default_factory=dict)

    @staticmethod
    def _collect_symbols(root_node: concepts.TreeNode) -> Dict[str, Any]:
        return _CollectSymbols.apply(root_node)

    @pydantic.root_validator(skip_on_failure=True)
    def _collect_symbols_validator(  # type: ignore  # validators are classmethods
        cls: Type[SymbolTableTrait], values: Dict[str, Any]
    ) -> Dict[str, Any]:
        values["symtable_"] = cls._collect_symbols(values)
        return values

    def collect_symbols(self) -> None:
        self.symtable_ = self._collect_symbols(self)
