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

from .. import common_definitions


@pytest.fixture
def symtable_node_with_expected_symbols(request):
    node = common_definitions.make_node_with_symbol_table()
    symbols = {
        node.node_with_name.name: node.node_with_name,
        node.node_with_default_name.name: node.node_with_default_name,
        node.compound_with_name.node_with_name.name: node.compound_with_name.node_with_name,
    }
    symbols.update({n.name: n for n in node.list_with_name})

    yield node, symbols


class TestSymbolTable:
    def test_symbol_table_creation(self, symtable_node_with_expected_symbols):
        node, expected_symbols = symtable_node_with_expected_symbols
        collected_symtable = node.__node_impl__.symtable
        assert isinstance(node.__node_impl__.symtable, dict)
        assert all(isinstance(key, str) for key in collected_symtable)

    def test_symbol_table_collection(self, symtable_node_with_expected_symbols):
        node, expected_symbols = symtable_node_with_expected_symbols
        collected_symtable = node.__node_impl__.symtable
        assert collected_symtable == expected_symbols
        assert all(
            collected_symtable[symbol_name] is symbol_node
            for symbol_name, symbol_node in expected_symbols.items()
        )

        # import devtools
        # devtools.debug(node)
        # print(f"COLLECTED: {collected_symtable}")
        # print(f"EXPECTED: {expected_symbols}")
