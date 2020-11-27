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


from . import concepts, utils
from .type_definitions import Enum
from .typingx import Any, Generator, List, Optional, Sequence, Type, Union


try:
    import cytoolz as toolz
except ModuleNotFoundError:
    import toolz  # noqa


class TraversalOrder(Enum):
    PRE_ORDER = "pre"
    POST_ORDER = "post"
    LEVELS_ORDER = "levels"


@utils.as_xiter
def traverse_pre(
    node: concepts.TreeNode, *, with_keys: bool = False, __key__: Optional[Any] = None
) -> Generator[concepts.TreeIterationItem, None, None]:
    """Create a pre-order tree traversal iterator (Depth-First Search).

    Args:
        with_keys: Return tuples of (key, object) values where keys are
            the reference to the object node in the parent.
            Defaults to `False`.

    """

    if with_keys:
        yield __key__, node
        for key, child in concepts.generic_iter_children(node, with_keys=True):
            yield from traverse_pre(child, with_keys=True, __key__=key)
    else:
        yield node
        for child in concepts.generic_iter_children(node, with_keys=False):
            yield from traverse_pre(child, with_keys=False)


@utils.as_xiter
def traverse_post(
    node: concepts.TreeNode, *, with_keys: bool = False, __key__: Optional[Any] = None
) -> Generator[concepts.TreeIterationItem, None, None]:
    """Create a post-order tree traversal iterator (Depth-First Search).

    Args:
        with_keys: Return tuples of (key, object) values where keys are
            the reference to the object node in the parent.
            Defaults to `False`.

    """
    if with_keys:
        for key, child in concepts.generic_iter_children(node, with_keys=True):
            yield from traverse_post(child, with_keys=True, __key__=key)
        yield __key__, node
    else:
        for child in concepts.generic_iter_children(node, with_keys=False):
            yield from traverse_post(child, with_keys=False)
        yield node


@utils.as_xiter
def traverse_levels(
    node: concepts.TreeNode,
    *,
    with_keys: bool = False,
    __key__: Optional[Any] = None,
    __queue__: Optional[List] = None,
) -> Generator[concepts.TreeIterationItem, None, None]:
    """Create a tree traversal iterator by levels (Breadth-First Search).

    Args:
        with_keys: Return tuples of (key, object) values where keys are
            the reference to the object node in the parent.
            Defaults to `False`.

    """
    __queue__ = __queue__ or []
    if with_keys:
        yield __key__, node
        __queue__.extend(concepts.generic_iter_children(node, with_keys=True))
        if __queue__:
            key, child = __queue__.pop(0)
            yield from traverse_levels(child, with_keys=True, __key__=key, __queue__=__queue__)
    else:
        yield node
        __queue__.extend(concepts.generic_iter_children(node, with_keys=False))
        if __queue__:
            child = __queue__.pop(0)
            yield from traverse_levels(child, with_keys=False, __queue__=__queue__)


def traverse_tree(
    node: concepts.TreeNode,
    traversal_order: TraversalOrder = TraversalOrder.PRE_ORDER,
    *,
    with_keys: bool = False,
) -> utils.XIterator[concepts.TreeIterationItem]:
    """Create a tree traversal iterator.

    Args:
        traversal_order: Tree nodes traversal order.
        with_keys: Return tuples of (key, object) values where keys are
            the reference to the object node in the parent.
            Defaults to `False`.

    """
    assert isinstance(traversal_order, TraversalOrder)
    iterator = globals()[f"traverse_{traversal_order.value}"](node=node, with_keys=with_keys)
    assert isinstance(iterator, utils.XIterator)

    return iterator


def select_from(
    node: concepts.TreeNode,
    traversal_order: TraversalOrder = TraversalOrder.PRE_ORDER,
    *,
    by_type: Optional[Union[Type, Sequence[Type]]] = None,
) -> utils.XIterator[concepts.TreeIterationItem]:
    """Create a tree traversal iterator."""

    iterator = traverse_tree(node, traversal_order)
    if by_type is not None:
        iterator = iterator.filter(utils.isinstancechecker(by_type))

    return iterator
