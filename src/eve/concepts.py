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


import abc
import functools
import itertools
import warnings

import boltons.typeutils
import pydantic
import pydantic.generics

from . import typing
from .type_definitions import IntEnum, PositiveInt, Str, StrEnum
from .typing import (
    Any,
    AnyNoArgCallable,
    ClassVar,
    Dict,
    Final,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
)


#: Marker value used to avoid confusion with `None`
#: (specially in contexts where `None` could be a valid value)
NOTHING = boltons.typeutils.make_sentinel(name="NOTHING", var_name="NOTHING")


class FieldKind(StrEnum):
    INPUT = "input"
    OUTPUT = "output"
    SYMBOL = "symbol"


class FieldConstraintsDict(TypedDict, total=False):
    vtype: Union["VType", Tuple["VType", ...]]


class FieldMetadataDict(TypedDict, total=False):
    kind: FieldKind
    constraints: FieldConstraintsDict


NodeMetadataDict = Dict[str, FieldMetadataDict]


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

    return typing.cast(pydantic.fields.FieldInfo, field_info)


in_field = functools.partial(field, kind=FieldKind.INPUT)
out_field = functools.partial(field, kind=FieldKind.OUTPUT)
symbol_field = functools.partial(field, kind=FieldKind.SYMBOL)


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


class Trait(abc.ABC):
    REGISTRY: Final[Dict[str, Type["Trait"]]] = {}
    name: ClassVar[str]

    @classmethod
    def __init_subclass__(cls, **kwargs: Any) -> None:
        assert hasattr(cls, "name") and isinstance(cls.name, str)
        Trait.REGISTRY[cls.name] = cls

    @classmethod
    @abc.abstractmethod
    def process_namespace(
        cls, namespace: Dict[str, Any], trait_names: List[str], meta_kwargs: Dict[str, Any]
    ) -> None:
        ...

    @classmethod
    @abc.abstractmethod
    def process_class(
        cls, node_class: Type["Node"], trait_names: List[str], meta_kwargs: Dict[str, Any]
    ) -> None:
        ...


_EVE_NODE_IMPL_SUFFIX = "_"

_EVE_NODE_ATTR_SUFFIX = "_attr_"


class NodeMetaclass(pydantic.main.ModelMetaclass):
    @typing.no_type_check
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Apply traits to the namespace and the class
        trait_names: List[Type[Trait]] = kwargs.pop("traits", [])
        traits = []
        for name in trait_names:
            trait = Trait.REGISTRY[name]
            trait.process_namespace(namespace, trait_names, kwargs)
            traits.append(trait)

        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        for trait in traits:
            trait.process_class(cls, trait_names, kwargs)

        # Add custom class members
        fields_metadata = {}
        for name, field in cls.__fields__.items():
            if not (name.endswith(_EVE_NODE_ATTR_SUFFIX) or name.endswith(_EVE_NODE_IMPL_SUFFIX)):
                fields_metadata[name] = field.field_info.extra.get(_EVE_METADATA_KEY, {})

        cls.__node_metadata__ = fields_metadata
        cls.__node_traits__ = trait_names

        return cls


AnyNode = TypeVar("AnyNode", bound="BaseNode")
ValueNode = Union[bool, bytes, int, float, str, IntEnum, StrEnum]
LeafNode = Union[AnyNode, ValueNode]
TreeNode = Union[AnyNode, Union[List[LeafNode], Dict[Any, LeafNode], Set[LeafNode]]]


class BaseNode(pydantic.BaseModel, metaclass=NodeMetaclass):
    """Base class representing an IR node.

    Field values should be either:

        * builtin types: `bool`, `bytes`, `int`, `float`, `str`
        * other :class:`Node` subclasses
        * other :class:`pydantic.BaseModel` subclasses
        * supported collections (:class:`List`, :class:`Dict`, :class:`Set`)
            of any of the previous items

    Field names may not start with "_". Field names ending with "_" are
    considered implementation helpers and will not appear in the node iterators.
    Field names ending with "_attr_" are considered meta-attributes of the node,
    not children.
    """

    __node_metadata__: NodeMetadataDict

    # Node fields
    #: Unique node-id (meta-attribute)
    id_attr_: Optional[Str] = None

    @pydantic.validator("id_attr_", pre=True, always=True)
    def _id_attr_validator(cls: Type[AnyNode], v: Optional[str]) -> str:  # type: ignore  # validators are classmethods
        if v is None:
            v = UIDGenerator.get_unique_id(prefix=cls.__qualname__)
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

    def select(self, *, kind: Optional[FieldKind] = None) -> Generator[Tuple[str, Any], None, None]:
        for name, _ in self.__fields__.items():
            if not (name.endswith(_EVE_NODE_ATTR_SUFFIX) or name.endswith(_EVE_NODE_IMPL_SUFFIX)):
                if kind and self.__node_metadata__.get("kind", None) == kind:
                    yield name, getattr(self, name)

    class Config(BaseModelConfig):
        pass


class GenericNode(BaseNode, pydantic.generics.GenericModel):
    pass


class Node(BaseNode):
    pass


class FrozenNode(Node):
    """Base inmutable node class."""

    class Config(FrozenModelConfig):
        pass


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


class UIDGenerator:
    """Simple unique id generator using a counter."""

    #: Constantly increasing counter for generation of unique ids
    __counter = itertools.count(1)

    @classmethod
    def get_unique_id(cls, *, prefix: Optional[str] = None) -> str:
        """Generate a new globally unique id (for the current session)."""
        count = next(cls.__counter)
        return f"{prefix}_{count}" if prefix else f"{count}"

    @classmethod
    def reset(cls, start: int = 1) -> None:
        """Reset global generator counter.

        Notes:
            If the new start value is lower than the last generated UID, new
            IDs are not longer guaranteed to be unique.

        """
        if start < next(cls.__counter):
            warnings.warn("Unsafe reset of global UIDGenerator", RuntimeWarning)
        cls.__counter = itertools.count(start)
