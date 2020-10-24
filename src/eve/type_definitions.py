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

"""Definitions of useful field and general types."""


from __future__ import annotations

import enum
import functools
import re

import boltons.typeutils
import xxhash
from boltons.typeutils import classproperty  # noqa: F401
from pydantic import (  # noqa: F401
    NegativeFloat,
    NegativeInt,
    PositiveFloat,
    PositiveInt,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    validator,
)

from ._typing import Any, Callable, Dict, Generator, Mapping, Optional, Type


#: Marker value used to avoid confusion with `None`
#: (specially in contexts where `None` could be a valid value)
NOTHING = boltons.typeutils.make_sentinel(name="NOTHING", var_name="NOTHING")

#: Marker value used by as a sentinel value to delete items
DELETE = boltons.typeutils.make_sentinel(name="DELETE", var_name="DELETE")


#: Collection types considered as single elements
ATOMIC_COLLECTION_TYPES = (str, bytes, bytearray)


#: Typing definitions for `__get_validators__()` methods (defined but not exported in `pydantic.typing`)
PydanticCallableGenerator = Generator[Callable[..., Any], None, None]


#: :class:`bool subclass for strict field definition
Bool = StrictBool  # noqa: F401
#: :class:`bytes subclass for strict field definition
Bytes = bytes  # noqa: F401
#: :class:`float` subclass for strict field definition
Float = StrictFloat  # noqa: F401
#: :class:`int` subclass for strict field definition
Int = StrictInt  # noqa: F401
#: :class:`str` subclass for strict field definition
Str = StrictStr  # noqa: F401


class Enum(enum.Enum):
    """Basic :class:`enum.Enum` subclass with strict type validation."""

    @classmethod
    def __get_validators__(cls) -> PydanticCallableGenerator:
        yield cls._strict_type_validator

    @classmethod
    def _strict_type_validator(cls, v: Any) -> Enum:
        if not isinstance(v, cls):
            raise TypeError(f"Invalid value type [expected: {cls}, received: {v.__class__}]")
        return v


class IntEnum(enum.IntEnum):
    """Basic :class:`enum.IntEnum` subclass with strict type validation."""

    @classmethod
    def __get_validators__(cls) -> PydanticCallableGenerator:
        yield cls._strict_type_validator

    @classmethod
    def _strict_type_validator(cls, v: Any) -> IntEnum:
        if not isinstance(v, cls):
            raise TypeError(f"Invalid value type [expected: {cls}, received: {v.__class__}]")
        return v


class StrEnum(str, enum.Enum):
    """Basic :class:`enum.Enum` subclass with strict type validation and supporting string operations."""

    @classmethod
    def __get_validators__(cls) -> PydanticCallableGenerator:
        yield cls._strict_type_validator

    @classmethod
    def _strict_type_validator(cls, v: Any) -> StrEnum:
        if not isinstance(v, cls):
            raise TypeError(f"Invalid value type [expected: {cls}, received: {v.__class__}]")
        return v

    def __str__(self) -> str:
        assert isinstance(self.value, str)
        return self.value


class SymbolName(str):
    """Name of a symbol."""

    NAME_REGEX = re.compile(r"[a-zA-Z_]\w*")

    @staticmethod
    @functools.lru_cache
    def constrained(regex: str) -> Type[SymbolName]:
        xxh64 = xxhash.xxh64()
        xxh64.update(regex.encode())
        subclass_name = f"SymbolName_{xxh64.hexdigest()[-8:]}"
        namespace = dict(NAME_REGEX=regex)

        return type(subclass_name, (SymbolName,), namespace)

    @classmethod
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)  # type: ignore  # mypy issues 4335, 4660
        if not hasattr(cls, "NAME_REGEX") or not isinstance(cls.NAME_REGEX, (str, re.Pattern)):
            raise TypeError(f"Missing or invalid 'NAME_REGEX' member in '{cls.__name__}' class.")
        elif isinstance(cls.NAME_REGEX, str):
            try:
                cls.NAME_REGEX = re.compile(cls.NAME_REGEX)
            except re.error as e:
                raise TypeError(
                    f"Invalid regular expression definition in '{cls.__name__}'."
                ) from e

    @classmethod
    def __get_validators__(cls) -> PydanticCallableGenerator:
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        field_schema.update(pattern=cls.NAME_REGEX.pattern)

    @classmethod
    def validate(cls, v: Any) -> SymbolName:
        return cls(v)

    def __init__(self, name: str, *, symtable: Optional[Mapping[str, Any]] = None) -> None:
        if not isinstance(name, str):
            raise TypeError(f"Invalid string argument '{name}'.")
        if not self.NAME_REGEX.fullmatch(name):
            raise ValueError(
                f"Invalid name value '{name}' does not match re({self.NAME_REGEX.pattern})."
            )
        self._symtable = symtable

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({super().__repr__()})"


class SymbolRef(str):
    """Reference to a symbol."""

    @classmethod
    def __get_validators__(cls) -> PydanticCallableGenerator:
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> SymbolRef:
        return cls(v)

    def __init__(self, name: str, *, context: Optional[Mapping[str, Any]] = None) -> None:
        if not isinstance(name, str):
            raise TypeError(f"Invalid string argument '{name}'.")
        self._context = context

    def node(self, *, context: Optional[Mapping[str, Any]] = None) -> Any:
        if context:
            self._context = context
        assert self._context
        return self._context[self]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({super().__repr__()})"
