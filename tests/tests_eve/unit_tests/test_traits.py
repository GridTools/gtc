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


from __future__ import annotations

import pytest

import eve
from eve.typingx import List

from .. import definitions


@pytest.fixture
def symtable_node_and_expected_symbols():
    node = definitions.make_node_with_symbol_table()
    symbols = {
        node.node_with_name.name: node.node_with_name,
        node.node_with_default_name.name: node.node_with_default_name,
        node.compound_with_name.node_with_name.name: node.compound_with_name.node_with_name,
    }
    symbols.update({n.name: n for n in node.list_with_name})

    yield node, symbols


class _NodeWithSymbolName(eve.Node):
    name: eve.SymbolName = eve.SymbolName("symbol_name")


class _NodeWithSymbolTable(eve.Node, eve.SymbolTableTrait):
    symbols: List[_NodeWithSymbolName]


class _NodeWithSymbolTableAndSymbol(eve.Node, eve.SymbolTableTrait):
    name: eve.SymbolName = eve.SymbolName("symbol_name")
    symbols: List[_NodeWithSymbolName]


class _NodeWithChildSymbolTable(eve.Node, eve.SymbolTableTrait):
    symbols: List[_NodeWithSymbolName]
    tbl: List[_NodeWithSymbolTableAndSymbol]


@pytest.fixture
def node_with_duplicated_names_maker():
    def _maker():
        return _NodeWithSymbolTable(symbols=[_NodeWithSymbolName(), _NodeWithSymbolName()])

    yield _maker


class TestSymbolTable:
    def test_symbol_table_creation(self, symtable_node_and_expected_symbols):
        node, expected_symbols = symtable_node_and_expected_symbols
        collected_symtable = node.symtable_
        assert isinstance(node.symtable_, dict)
        assert all(isinstance(key, str) for key in collected_symtable)

    def test_symbol_table_collection(self, symtable_node_and_expected_symbols):
        node, expected_symbols = symtable_node_and_expected_symbols
        collected_symtable = node.symtable_
        assert collected_symtable == expected_symbols
        assert all(
            collected_symtable[symbol_name] is symbol_node
            for symbol_name, symbol_node in expected_symbols.items()
        )

    def test_symble_table_collection_nested(self):
        inner_symbol_table_node = _NodeWithSymbolTableAndSymbol(
            name="outer_symbol_for_inner_symbol_tbl",
            symbols=[_NodeWithSymbolName(name="inner_symbol")],
        )
        outer_symbol_table_node = _NodeWithChildSymbolTable(
            tbl=[inner_symbol_table_node], symbols=[_NodeWithSymbolName(name="outer_symbol")]
        )

        expected_outer_symbols = {
            outer_symbol_table_node.symbols[0].name: outer_symbol_table_node.symbols[0],
            inner_symbol_table_node.name: inner_symbol_table_node,
        }
        collected_outer_symbols = outer_symbol_table_node.symtable_
        assert collected_outer_symbols == expected_outer_symbols

        expected_inner_symbols = {
            inner_symbol_table_node.symbols[0].name: inner_symbol_table_node.symbols[0]
        }
        collected_inner_symbols = inner_symbol_table_node.symtable_
        assert collected_inner_symbols == expected_inner_symbols
