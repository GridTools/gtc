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
import types

import pydantic
import pydantic.generics

from . import type_definitions, utils
from .type_definitions import NOTHING, IntEnum, Str, StrEnum
from .typingx import (
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
    TypedDict,
    TypeVar,
    Union,
    no_type_check,
)


# -- Fields --
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
    underscore_attrs_are_private = True
    # TODO(egparedes): setting 'underscore_attrs_are_private' to True breaks sphinx-autodoc


class FrozenModelConfig(BaseModelConfig):
    allow_mutation = False


class Model(pydantic.BaseModel):
    class Config(BaseModelConfig):
        pass


class FrozenModel(pydantic.BaseModel):
    class Config(FrozenModelConfig):
        pass


# -- Nodes --
AnyNode = TypeVar("AnyNode", bound="BaseNode")
ValueNode = Union[bool, bytes, int, float, str, IntEnum, StrEnum]
LeafNode = Union[AnyNode, ValueNode]
TreeNode = Union[AnyNode, Union[List[LeafNode], Dict[Any, LeafNode], Set[LeafNode]]]


def _is_data_annotation_name(name: str) -> bool:
    return name.endswith("_") and not name.endswith("__") and not name.startswith("_")


def _is_child_field_name(name: str) -> bool:
    return not name.endswith("_") and not name.startswith("_")


def _is_internal_field_name(name: str) -> bool:
    return name.endswith("__") and not name.startswith("_")


def _is_private_attr_name(name: str) -> bool:
    return name.startswith("_")


class NodeMetaclass(pydantic.main.ModelMetaclass):
    """Custom metaclass for Node classes.

    Customize the creation of Node classes adding Eve specific attributes.

    """

    @no_type_check
    def __new__(mcls, name, bases, namespace, **kwargs):
        # Optional preprocessing of class namespace before creation:
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)

        # Postprocess created class:
        # Add metadata class members
        children_metadata = {}
        for name, model_field in cls.__fields__.items():
            assert not _is_private_attr_name(name)
            if _is_data_annotation_name(name):
                raise TypeError(f"Invalid field name ('{name}') looks like a data annotation.")
            if _is_internal_field_name(name):
                raise TypeError(f"Invalid field name ('{name}') looks like an Eve internal field.")
            if _is_child_field_name(name):
                children_metadata[name] = {
                    "definition": model_field,
                    **model_field.field_info.extra.get(_EVE_METADATA_KEY, {}),
                }

        cls.__node_children__ = children_metadata

        return cls


class BaseNode(pydantic.BaseModel, metaclass=NodeMetaclass):
    """Base class representing an IR node.

    A node is currently implemented as a pydantic Model with some extra features.

    Field values should be either:

        * builtin types: `bool`, `bytes`, `int`, `float`, `str`
        * enum.Enum types
        * other :class:`Node` subclasses
        * other :class:`pydantic.BaseModel` subclasses
        * supported collections (:class:`Tuple`, :class:`List`, :class:`Dict`, :class:`Set`)
          of any of the previous items

    Naming semantics for Node members:

        * Member names starting with ``_`` (e.g. ``_private`` or ``__private__``) are
          transformed into `private attributes` by pydantic and thus ignored by Eve. Since
          none of the pydantic features will work on them (type coercion, validators, etc.),
          it is not recommended to define new pydantic private attributes in the nodes.
        * Member names ending with ``_`` (and not starting with ``_``, e.g. ``my_data_``)
          are considered node data annotations, not children nodes. They are intended
          to be used by the user, typically to cache derived, non-essential information
          on the node, and they can be assigned directly without a explicit definition
          in the class body (which will consequently trigger an error).
        * Member names ending with ``__`` (and not starting with ``_``, e.g. ``internal__``)
          are reserved for internal Eve use and should NOT be defined by regular users.
          All pydantic features will work on these fields anyway but they will be
          not visible visible in Eve nodes.

    """

    __node_children__: ClassVar[NodeChildrenMetadataDict]

    # Node private attributes
    #: Unique node-id
    __node_id__: Optional[str] = pydantic.PrivateAttr(  # type: ignore  # mypy can't find PrivateAttr
        default_factory=utils.UIDGenerator.sequential_id
    )

    #: Node analysis annotations
    __node_annotations__: Optional[Str] = pydantic.PrivateAttr(  # type: ignore  # mypy can't find PrivateAttr
        default_factory=types.SimpleNamespace
    )

    def iter_children(self) -> Generator[Tuple[str, Any], None, None]:
        for name, _ in self.__fields__.items():
            if _is_child_field_name(name):
                yield name, getattr(self, name)

    def iter_children_names(self) -> Generator[str, None, None]:
        for name, _ in self.iter_children():
            yield name

    def iter_children_values(self) -> Generator[Any, None, None]:
        for _, node in self.iter_children():
            yield node

    @property
    def data_annotations(self) -> Dict[str, Any]:
        return self.__node_annotations__.__dict__

    @property
    def private_attrs_names(self) -> Set[str]:
        return set(self.__slots__) - {"__doc__"}

    def __getattr__(self, name: str) -> Any:
        if _is_data_annotation_name(name):
            try:
                return super().__getattribute__("__node_annotations__").__getattribute__(name)
            except AttributeError as e:
                raise AttributeError(f"Invalid data annotation name: '{name}'") from e
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if _is_data_annotation_name(name):
            try:
                super().__getattribute__("__node_annotations__").__setattr__(name, value)
            except AttributeError as e:
                raise AttributeError(f"Invalid data annotation name: '{name}'") from e
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if _is_data_annotation_name(name):
            try:
                super().__getattribute__("__node_annotations__").__delattr__(name)
            except AttributeError as e:
                raise AttributeError(f"Invalid data annotation name: '{name}'") from e
        else:
            super().__delattr__(name)

    class Config(BaseModelConfig):
        pass


class GenericNode(BaseNode, pydantic.generics.GenericModel):
    """Base generic node class."""

    pass


class Node(BaseNode):
    """Default public name for a base node class."""

    pass


class FrozenNode(Node):
    """Default public name for an immutable base node class."""

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
        children_iterator = node.iter_children() if with_keys else node.iter_children_values()
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
