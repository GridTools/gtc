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

"""Definitions of basic Eve concepts."""


from __future__ import annotations

import collections.abc
import functools

import pydantic

from . import type_definitions, utils
from ._typing import (
    Any,
    AnyNoArgCallable,
    ClassVar,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    no_type_check,
)
from .type_definitions import NOTHING, IntEnum, PositiveInt, Str, StrEnum


# -- Attributes and fields --
class AttributeMetadataDict(TypedDict, total=False):
    info: pydantic.fields.FieldInfo


NodeAttributeMetadataDict = Dict[str, AttributeMetadataDict]


class FieldKind(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


class FieldConstraintsDict(TypedDict, total=False):
    vtype: Union[VType, Tuple[VType, ...]]


class FieldMetadataDict(TypedDict, total=False):
    constraints: FieldConstraintsDict
    kind: FieldKind
    definition: pydantic.fields.ModelField


NodeChildrenMetadataDict = Dict[str, FieldMetadataDict]


_EVE_METADATA_KEY = "_EVE_META_"


def field(
    default: Any = NOTHING,
    *,
    default_factory: Optional[AnyNoArgCallable] = None,
    kind: Optional[FieldKind] = None,
    constraints: Optional[FieldConstraintsDict] = None,
    schema_config: Dict[str, Any] = None,
) -> pydantic.fields.FieldInfo:
    metadata = {}
    for key in ["kind", "constraints"]:
        value = locals()[key]
        if value:
            metadata[key] = value
    kwargs = schema_config or {}
    kwargs[_EVE_METADATA_KEY] = metadata

    if default is NOTHING:
        field_info = pydantic.Field(default_factory=default_factory, **kwargs)
    else:
        field_info = pydantic.Field(default, default_factory=default_factory, **kwargs)
    assert isinstance(field_info, pydantic.fields.FieldInfo)

    return field_info


in_field = functools.partial(field, kind=FieldKind.INPUT)
out_field = functools.partial(field, kind=FieldKind.OUTPUT)


# -- Models --
class BaseModelConfig:
    extra = "forbid"


class FrozenModelConfig(BaseModelConfig):
    allow_mutation = False


class Model(pydantic.BaseModel):
    class Config(BaseModelConfig):
        pass


class FrozenModel(pydantic.BaseModel):
    class Config(FrozenModelConfig):
        pass


# -- Nodes --
_EVE_NODE_IMPL_SUFFIX = "_"
_EVE_NODE_ATTR_SUFFIX = "_attr_"

AnyNode = TypeVar("AnyNode", bound="BaseNode")
ValueNode = Union[bool, bytes, int, float, str, IntEnum, StrEnum]
LeafNode = Union[AnyNode, ValueNode]
TreeNode = Union[AnyNode, Union[List[LeafNode], Dict[Any, LeafNode], Set[LeafNode]]]


class NodeMetaclass(pydantic.main.ModelMetaclass):
    """Custom metaclass for Node classes (inherits from pydantic metaclass).

    Customize the creation of Node classes adding Eve specific attributes.

    """

    @no_type_check
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Optional preprocessing of class namespace before creation:
        #

        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Postprocess created class:
        # Add metadata class members
        attributes_metadata = {}
        children_metadata = {}
        for name, model_field in cls.__fields__.items():
            if name.endswith(_EVE_NODE_ATTR_SUFFIX):
                attributes_metadata[name] = {"definition": model_field}
            elif not name.endswith(_EVE_NODE_IMPL_SUFFIX):
                children_metadata[name] = {
                    "definition": model_field,
                    **model_field.field_info.extra.get(_EVE_METADATA_KEY, {}),
                }

        cls.__node_attributes__ = attributes_metadata
        cls.__node_children__ = children_metadata

        return cls


class BaseNode(pydantic.BaseModel, metaclass=NodeMetaclass):
    """Base class representing an IR node.

    It is currently implemented as a pydantic Model with some extra features.

    Field values should be either:

        * builtin types: `bool`, `bytes`, `int`, `float`, `str`
        * other :class:`Node` subclasses
        * other :class:`pydantic.BaseModel` subclasses
        * supported collections (:class:`List`, :class:`Dict`, :class:`Set`)
            of any of the previous items

    Field naming scheme:

        * Field names starting with "_" are ignored by pydantic and Eve. They
            will not be considered as `fields` and thus none of the pydantic
            features will work (type coercion, validators, etc.).
        * Field names ending with "_" are ignored only by Eve, not by pydantic.
            This means that all pydantic features will work on these fields,
            but they will be invisible for Eve. They are reserved for internal Eve
            use and should not be defined by regular users.
        * Field names ending with "_attr_" are considered implementation fields
            not children. They are intended to be defined by users when needed,
            typically to cache derived, non-essential information on the node.

    """

    __node_attributes__: ClassVar[NodeAttributeMetadataDict]
    __node_children__: ClassVar[NodeChildrenMetadataDict]

    # Node fields
    #: Unique node-id (meta-attribute)
    id_attr_: Optional[Str] = None

    @pydantic.validator("id_attr_", pre=True, always=True)
    def _id_attr_validator(cls: Type[AnyNode], v: Optional[str]) -> str:  # type: ignore  # validators are classmethods
        if v is None:
            v = utils.UIDGenerator.sequential_id(prefix=cls.__qualname__)
        if not isinstance(v, str):
            raise TypeError(f"id_attr_ is not an 'str' instance ({type(v)})")
        return v

    def iter_attributes(self) -> Generator[Tuple[str, Any], None, None]:
        for name, _ in self.__fields__.items():
            if name.endswith(_EVE_NODE_ATTR_SUFFIX):
                yield name, getattr(self, name)

    def iter_children(self) -> Generator[Tuple[str, Any], None, None]:
        for name, _ in self.__fields__.items():
            if not (name.endswith(_EVE_NODE_ATTR_SUFFIX) or name.endswith(_EVE_NODE_IMPL_SUFFIX)):
                yield name, getattr(self, name)

    def iter_children_nodes(self) -> Generator[Any, None, None]:
        for _, node in self.iter_children():
            yield node

    class Config(BaseModelConfig):
        pass


class Node(BaseNode):
    """Default public name for a base node class."""

    pass


class FrozenNode(Node):
    """Default public name for an inmutable base node class."""

    class Config(FrozenModelConfig):
        pass


KeyValue = Tuple[Union[int, str], Any]
TreeIterationItem = Union[Any, Tuple[KeyValue, Any]]


def generic_iter_children(
    node: TreeNode, *, with_keys: bool = False
) -> Iterable[Union[Any, Tuple[KeyValue, Any]]]:
    """Create an iterator to traverse values as Eve tree nodes.

    Args:
        with_keys: Return tuples of (key, object) values where keys are
            the reference to the object node in the parent.
            Defaults to `False`.

    """

    children_iterator: Iterable[Union[Any, Tuple[KeyValue, Any]]] = iter(())
    if isinstance(node, Node):
        children_iterator = node.iter_children() if with_keys else node.iter_children_nodes()
    elif isinstance(node, collections.abc.Sequence) and not isinstance(
        node, type_definitions.ATOMIC_COLLECTION_TYPES
    ):
        children_iterator = enumerate(node) if with_keys else iter(node)
    elif isinstance(node, collections.abc.Set):
        children_iterator = zip(node, node) if with_keys else iter(node)  # type: ignore  # problems with iter(Set)
    elif isinstance(node, collections.abc.Mapping):
        children_iterator = node.items() if with_keys else node.values()

    return children_iterator


# -- Misc --
class VType(FrozenModel):

    # VType fields
    #: Unique name
    name: Str

    def __init__(self, name: str) -> None:
        super().__init__(name=name)


class SourceLocation(FrozenModel):
    """Source code location (line, column, source)."""

    line: PositiveInt
    column: PositiveInt
    source: Str

    def __init__(self, line: int, column: int, source: str) -> None:
        super().__init__(line=line, column=column, source=source)

    def __str__(self) -> str:
        src = self.source or ""
        return f"<{src}: Line {self.line}, Col {self.column}>"
