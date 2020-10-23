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


import pydantic

from . import concepts, iterators
from ._typing import Any, Dict, Type
from .type_definitions import SymbolName


class SymbolTableTrait(concepts.Model):
    symtable_attr_: Dict[str, Any] = pydantic.Field(default_factory=dict)

    @staticmethod
    def _collect_symbols(root_node: concepts.TreeNode) -> Dict[str, Any]:
        collected = {}
        for node in iterators.traverse_tree(root_node):
            if isinstance(node, concepts.BaseNode):
                for name, metadata in node.__node_children__.items():
                    if isinstance(metadata["definition"].type_, type) and issubclass(
                        metadata["definition"].type_, SymbolName
                    ):
                        collected[getattr(node, name)] = node

        return collected

    @pydantic.root_validator(skip_on_failure=True)
    def _collect_symbols_validator(  # type: ignore  # validators are classmethods
        cls: Type["SymbolTableTrait"], values: Dict[str, Any]
    ) -> Dict[str, Any]:
        values["symtable_attr_"] = cls._collect_symbols(values)
        return values

    def collect_symbols(self) -> None:
        self.symtable_attr_ = self._collect_symbols(self)
