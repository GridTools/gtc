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

from . import concepts, iterators
from .type_definitions import SymbolName
from .typingx import Any, Dict


class SymbolTableTrait(concepts.Model):
    """Trait implementing automatic symbol table creation for nodes.

    Nodes inheriting this trait will collect all the
    :class:`eve.type_definitions.SymbolName` instances defined in the
    descendant nodes and store them in the ``__node_symtable__`` private
    attribute.

    Attributes:
        __node_symtable__: Dict[str, concepts.BaseNode]
        mapping from symbol name to the symbol node
            where it was defined.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.collect_symbols()

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

    def collect_symbols(self) -> None:
        assert isinstance(self, concepts.BaseNode)
        self.__node_impl__.symtable = self._collect_symbols(self)
