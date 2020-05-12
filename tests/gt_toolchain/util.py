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

from typing import Any, Callable, Type

from eve import core


class FindNodes(core.NodeVisitor):
    def __init__(self, **kwargs):
        self.result = []

    def visit(self, node: core.Node, **kwargs) -> Any:
        if kwargs["predicate"](node):
            self.result.append(node)
        self.generic_visit(node, **kwargs)
        return self.result

    @classmethod
    def by_predicate(cls, predicate: Callable[[core.Node], bool], node: core.Node, **kwargs):
        return cls().visit(node, predicate=predicate)

    @classmethod
    def by_type(cls, node_type: Type[core.Node], node: core.Node, **kwargs):
        def type_predicate(node: core.Node):
            return isinstance(node, node_type)

        return cls.by_predicate(type_predicate, node)
