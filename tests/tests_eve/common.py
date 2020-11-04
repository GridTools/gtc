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


import enum
import random
import string
from typing import Collection, Dict, List, Mapping, Optional, Sequence, Set, Type, TypeVar

from pydantic import Field, validator  # noqa: F401

from eve.concepts import FrozenNode, Node, VType
from eve.type_definitions import Bool, Bytes, Float, Int, IntEnum, SourceLocation, Str, StrEnum


T = TypeVar("T")
S = TypeVar("S")


class Factories:
    STR_LEN = 6

    @classmethod
    def make_bool(cls) -> bool:
        return True

    @classmethod
    def make_int(cls) -> int:
        return 1

    @classmethod
    def make_neg_int(cls) -> int:
        return -2

    @classmethod
    def make_pos_int(cls) -> int:
        return 2

    @classmethod
    def make_float(cls) -> float:
        return 1.1

    @classmethod
    def make_str(cls, length: Optional[int] = None) -> str:
        length = length or cls.STR_LEN
        return string.ascii_letters[:length]

    @classmethod
    def make_member(cls, values: Sequence[T]) -> T:
        return values[0]

    @classmethod
    def make_collection(
        cls,
        item_type: Type[T],
        collection_type: Type[Collection[T]] = list,
        length: Optional[int] = None,
    ) -> Collection[T]:
        length = length or cls.STR_LEN

        maker_attr_name = f"make_{item_type.__name__}"
        if hasattr(cls, maker_attr_name):
            maker = getattr(cls, maker_attr_name)
        else:

            def maker():
                return item_type()

        return collection_type([maker() for _ in range(length)])  # type: ignore

    @classmethod
    def make_mapping(
        cls,
        key_type: Type[S],
        value_type: Type[T],
        mapping_type: Type[Mapping[S, T]] = dict,
        length: Optional[int] = None,
    ) -> Mapping[S, T]:
        length = length or cls.STR_LEN

        key_maker_attr_name = f"make_{key_type.__name__}"
        if hasattr(cls, key_maker_attr_name):
            key_maker = getattr(cls, key_maker_attr_name)
        else:

            def key_maker():
                return key_type()

        value_maker_attr_name = f"make_{value_type.__name__}"
        if hasattr(cls, value_maker_attr_name):
            value_maker = getattr(cls, value_maker_attr_name)
        else:

            def value_maker():
                return value_type()

        return mapping_type({key_maker(): value_maker() for _ in range(length)})  # type: ignore


class RandomFactories(Factories):
    MIN_INT = -9999
    MAX_INT = 9999
    MIN_FLOAT = -999.0
    MAX_FLOAT = 999.09

    @classmethod
    def make_bool(cls) -> bool:
        return random.choice([True, False])

    @classmethod
    def make_int(cls) -> int:
        return random.randint(cls.MIN_INT, cls.MAX_INT)

    @classmethod
    def make_neg_int(cls) -> int:
        return random.randint(cls.MIN_INT, 1)

    @classmethod
    def make_pos_int(cls) -> int:
        return random.randint(1, cls.MAX_INT)

    @classmethod
    def make_float(cls) -> float:
        return cls.MIN_FLOAT + random.random() * (cls.MAX_FLOAT - cls.MIN_FLOAT)

    @classmethod
    def make_str(cls, length: Optional[int] = None) -> str:
        length = length or cls.STR_LEN
        return "".join(random.choice(string.ascii_letters) for _ in range(length))

    @classmethod
    def make_member(cls, values: Sequence[T]) -> T:
        return random.choice(values)


@enum.unique
class IntKind(IntEnum):
    """Sample int Enum."""

    MINUS = -1
    ZERO = 0
    PLUS = 1


@enum.unique
class StrKind(StrEnum):
    """Sample string Enum."""

    FOO = "foo"
    BLA = "bla"
    FIZ = "fiz"
    FUZ = "fuz"


SimpleVType = VType("simple")


class EmptyNode(Node):
    pass


class LocationNode(Node):
    loc: SourceLocation


class SimpleNode(Node):
    int_value: Int
    bool_value: Bool
    float_value: Float
    str_value: Str
    bytes_value: Bytes
    int_kind: IntKind
    str_kind: StrKind


class SimpleNodeWithOptionals(Node):
    int_value: Int
    float_value: Optional[Float]
    str_value: Optional[Str]


class SimpleNodeWithLoc(Node):
    int_value: Int
    float_value: Float
    str_value: Str
    loc: Optional[SourceLocation]


class SimpleNodeWithCollections(Node):
    int_value: Int
    int_list: List[Int]
    str_set: Set[Str]
    str_to_int_dict: Dict[Str, Int]
    loc: Optional[SourceLocation]


class SimpleNodeWithAbstractCollections(Node):
    int_value: Int
    int_sequence: Sequence[Int]
    str_set: Set[Str]
    str_to_int_mapping: Mapping[Str, Int]
    loc: Optional[SourceLocation]


class CompoundNode(Node):
    int_value: Int
    location: LocationNode
    simple: SimpleNode
    simple_loc: SimpleNodeWithLoc
    simple_opt: SimpleNodeWithOptionals
    other_simple_opt: Optional[SimpleNodeWithOptionals]


class FrozenSimpleNode(FrozenNode):
    int_value: Int
    bool_value: Bool
    float_value: Float
    str_value: Str
    bytes_value: Bytes
    int_kind: IntKind
    str_kind: StrKind


# -- Maker functions --
def make_source_location(fixed: bool = False) -> SourceLocation:
    factories = Factories if fixed else RandomFactories
    line = factories.make_pos_int()
    column = factories.make_pos_int()
    str_value = factories.make_str()
    source = f"file_{str_value}.py"

    return SourceLocation(line=line, column=column, source=source)


def make_empty_node(fixed: bool = False) -> LocationNode:
    return EmptyNode()


def make_location_node(fixed: bool = False) -> LocationNode:
    return LocationNode(loc=make_source_location(fixed))


def make_simple_node(fixed: bool = False) -> SimpleNode:
    factories = Factories if fixed else RandomFactories
    int_value = factories.make_int()
    bool_value = factories.make_bool()
    float_value = factories.make_float()
    str_value = factories.make_str()
    bytes_value = factories.make_str().encode()
    int_kind = IntKind.PLUS if fixed else factories.make_member([*IntKind])
    str_kind = StrKind.BLA if fixed else factories.make_member([*StrKind])

    return SimpleNode(
        int_value=int_value,
        bool_value=bool_value,
        float_value=float_value,
        str_value=str_value,
        bytes_value=bytes_value,
        int_kind=int_kind,
        str_kind=str_kind,
    )


def make_simple_node_with_optionals(fixed: bool = False) -> SimpleNodeWithOptionals:
    factories = Factories if fixed else RandomFactories
    int_value = factories.make_int()
    float_value = factories.make_float()

    return SimpleNodeWithOptionals(int_value=int_value, float_value=float_value)


def make_simple_node_with_loc(fixed: bool = False) -> SimpleNodeWithLoc:
    factories = Factories if fixed else RandomFactories
    int_value = factories.make_int()
    float_value = factories.make_float()
    str_value = factories.make_str()
    loc = make_source_location(fixed)

    return SimpleNodeWithLoc(
        int_value=int_value, float_value=float_value, str_value=str_value, loc=loc
    )


def make_simple_node_with_collections(fixed: bool = False) -> SimpleNodeWithCollections:
    factories = Factories if fixed else RandomFactories
    int_value = factories.make_int()
    int_list = factories.make_collection(int, length=3)
    str_set = factories.make_collection(str, set, length=3)
    str_to_int_dict = factories.make_mapping(key_type=str, value_type=int, length=3)
    loc = make_source_location(fixed)

    return SimpleNodeWithCollections(
        int_value=int_value,
        int_list=int_list,
        str_set=str_set,
        str_to_int_dict=str_to_int_dict,
        loc=loc,
    )


def make_simple_node_with_abstractcollections(
    fixed: bool = False,
) -> SimpleNodeWithAbstractCollections:
    factories = Factories if fixed else RandomFactories
    int_value = factories.make_int()
    int_sequence = factories.make_collection(int, collection_type=tuple, length=3)
    str_set = factories.make_collection(str, set, length=3)
    str_to_int_mapping = factories.make_mapping(key_type=str, value_type=int, length=3)

    return SimpleNodeWithAbstractCollections(
        int_value=int_value,
        int_sequence=int_sequence,
        str_set=str_set,
        str_to_int_mapping=str_to_int_mapping,
    )


def make_compound_node(fixed: bool = False) -> CompoundNode:
    factories = Factories if fixed else RandomFactories
    return CompoundNode(
        int_value=factories.make_int(),
        location=make_location_node(),
        simple=make_simple_node(),
        simple_loc=make_simple_node_with_loc(),
        simple_opt=make_simple_node_with_optionals(),
        other_simple_opt=None,
    )


def make_frozen_simple_node(fixed: bool = False) -> FrozenSimpleNode:
    factories = Factories if fixed else RandomFactories
    int_value = factories.make_int()
    bool_value = factories.make_bool()
    float_value = factories.make_float()
    str_value = factories.make_str()
    bytes_value = factories.make_str().encode()
    int_kind = IntKind.PLUS if fixed else factories.make_member([*IntKind])
    str_kind = StrKind.BLA if fixed else factories.make_member([*StrKind])

    return FrozenSimpleNode(
        int_value=int_value,
        bool_value=bool_value,
        float_value=float_value,
        str_value=str_value,
        bytes_value=bytes_value,
        int_kind=int_kind,
        str_kind=str_kind,
    )


# -- Makers of invalid nodes --
def make_invalid_location_node(fixed: bool = False) -> LocationNode:
    return LocationNode(loc=SourceLocation(line=0, column=-1, source="<str>"))


def make_invalid_at_int_simple_node(fixed: bool = False) -> SimpleNode:
    factories = Factories if fixed else RandomFactories
    int_value = factories.make_float()
    bool_value = factories.make_bool()
    float_value = factories.make_float()
    bytes_value = factories.make_str().encode()
    str_value = factories.make_str()
    int_kind = IntKind.PLUS if fixed else factories.make_member([*IntKind])
    str_kind = StrKind.BLA if fixed else factories.make_member([*StrKind])

    return SimpleNode(
        int_value=int_value,
        bool_value=bool_value,
        float_value=float_value,
        str_value=str_value,
        bytes_value=bytes_value,
        int_kind=int_kind,
        str_kind=str_kind,
    )


def make_invalid_at_float_simple_node(fixed: bool = False) -> SimpleNode:
    factories = Factories if fixed else RandomFactories
    int_value = factories.make_int()
    bool_value = factories.make_bool()
    float_value = factories.make_int()
    str_value = factories.make_str()
    bytes_value = factories.make_str().encode()
    int_kind = IntKind.PLUS if fixed else factories.make_member([*IntKind])
    str_kind = StrKind.BLA if fixed else factories.make_member([*StrKind])

    return SimpleNode(
        int_value=int_value,
        bool_value=bool_value,
        float_value=float_value,
        str_value=str_value,
        bytes_value=bytes_value,
        int_kind=int_kind,
        str_kind=str_kind,
    )


def make_invalid_at_str_simple_node(fixed: bool = False) -> SimpleNode:
    factories = Factories if fixed else RandomFactories
    int_value = factories.make_int()
    bool_value = factories.make_bool()
    float_value = factories.make_float()
    str_value = factories.make_float()
    bytes_value = factories.make_str().encode()
    int_kind = IntKind.PLUS if fixed else factories.make_member([*IntKind])
    str_kind = StrKind.BLA if fixed else factories.make_member([*StrKind])

    return SimpleNode(
        int_value=int_value,
        bool_value=bool_value,
        float_value=float_value,
        str_value=str_value,
        bytes_value=bytes_value,
        int_kind=int_kind,
        str_kind=str_kind,
    )


def make_invalid_at_bytes_simple_node(fixed: bool = False) -> SimpleNode:
    factories = Factories if fixed else RandomFactories
    int_value = factories.make_int()
    bool_value = factories.make_bool()
    float_value = factories.make_float()
    str_value = factories.make_float()
    bytes_value = [1, "2", (3, 4)]
    int_kind = IntKind.PLUS if fixed else factories.make_member([*IntKind])
    str_kind = StrKind.BLA if fixed else factories.make_member([*StrKind])

    return SimpleNode(
        int_value=int_value,
        bool_value=bool_value,
        float_value=float_value,
        str_value=str_value,
        bytes_value=bytes_value,
        int_kind=int_kind,
        str_kind=str_kind,
    )


def make_invalid_at_enum_simple_node(fixed: bool = False) -> SimpleNode:
    factories = Factories if fixed else RandomFactories
    int_value = factories.make_int()
    bool_value = factories.make_bool()
    float_value = factories.make_float()
    str_value = factories.make_float()
    bytes_value = factories.make_str().encode()
    int_kind = IntKind.PLUS if fixed else factories.make_member([*IntKind])
    str_kind = StrKind.BLA if fixed else factories.make_member([*StrKind])

    return SimpleNode(
        int_value=int_value,
        bool_value=bool_value,
        float_value=float_value,
        str_value=str_value,
        bytes_value=bytes_value,
        int_kind=int_kind,
        str_kind=str_kind,
    )
