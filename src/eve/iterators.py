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
# version. See the LICENSE.txt file at the top-l directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Iterator utils."""


import collections.abc
from typing import Iterator

from . import concepts
from ._typing import Any, Generator, List, Optional
from .type_definitions import StrEnum


class TraversalOrder(StrEnum):
    DFS_PREORDER = "dfs_preorder"
    DFS_POSTORDER = "dfs_postorder"
    BFS = "bfs"


def traverse_dfs_preorder(
    node: concepts.TreeNode, *, with_keys: bool = False, __key__: Optional[Any] = None
) -> Generator[concepts.TreeIterationItem, None, None]:
    if with_keys:
        yield __key__, node
        for key, child in concepts.generic_iter_children(node, with_keys=True):
            yield from traverse_dfs_preorder(child, with_keys=True, __key__=key)
    else:
        yield node
        for child in concepts.generic_iter_children(node, with_keys=False):
            yield from traverse_dfs_preorder(child, with_keys=False)


def traverse_dfs_postorder(
    node: concepts.TreeNode, *, with_keys: bool = False, __key__: Optional[Any] = None
) -> Generator[concepts.TreeIterationItem, None, None]:
    if with_keys:
        for key, child in concepts.generic_iter_children(node, with_keys=True):
            yield from traverse_dfs_postorder(child, with_keys=True, __key__=key)
        yield __key__, node
    else:
        for child in concepts.generic_iter_children(node, with_keys=False):
            yield from traverse_dfs_postorder(child, with_keys=False)
        yield node


def traverse_bfs(
    node: concepts.TreeNode,
    *,
    with_keys: bool = False,
    __key__: Optional[Any] = None,
    __queue__: Optional[List] = None,
) -> Generator[concepts.TreeIterationItem, None, None]:
    __queue__ = __queue__ or []
    if with_keys:
        yield __key__, node
        __queue__.extend(concepts.generic_iter_children(node, with_keys=True))
        if __queue__:
            key, child = __queue__.pop(0)
            yield from traverse_bfs(child, with_keys=True, __key__=key, __queue__=__queue__)
    else:
        yield node
        __queue__.extend(concepts.generic_iter_children(node, with_keys=False))
        if __queue__:
            child = __queue__.pop(0)
            yield from traverse_bfs(child, with_keys=False, __queue__=__queue__)


def traverse_tree(
    node: concepts.TreeNode,
    traversal_order: TraversalOrder = TraversalOrder.DFS_PREORDER,
    *,
    with_keys: bool = False,
) -> Iterator[concepts.TreeIterationItem]:
    assert isinstance(traversal_order, TraversalOrder)
    iterator = globals()[f"traverse_{traversal_order.value}"](node=node, with_keys=with_keys)
    assert isinstance(iterator, collections.abc.Iterator)

    return iterator
