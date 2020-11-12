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

import abc
import collections
import enum
import functools
import re

import boltons.typeutils
import pydantic
import pydantic.typing
import xxhash
from boltons.typeutils import classproperty  # noqa: F401
from pydantic import (  # noqa: F401
    ConstrainedStr,
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

from .typingx import Any, Callable, Dict, Generator, Mapping, Optional, Type


#: Marker value used to avoid confusion with `None`
#: (specially in contexts where `None` could be a valid value)
NOTHING = boltons.typeutils.make_sentinel(name="NOTHING", var_name="NOTHING")

#: Marker value used as a sentinel value to delete items
DELETE = boltons.typeutils.make_sentinel(name="DELETE", var_name="DELETE")


#: Collection types considered as single elements
ATOMIC_COLLECTION_TYPES = (str, bytes, bytearray, collections.UserString)


class AtomicCollection(abc.ABC):
    """Abstract base class for atomic collection types."""

    ...


for t in ATOMIC_COLLECTION_TYPES:
    AtomicCollection.register(t)  # type: ignore  # mypy gets it wrong


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


class Str(abc.ABC):
    """Abstract base class for string types with strict validation."""

    @classmethod
    def __get_validators__(cls) -> pydantic.typing.CallableGenerator:
        yield cls._type_validation

    @classmethod
    def _type_validation(cls, value: Str) -> Str:
        if not isinstance(value, cls):
            raise TypeError(f"Invalid Str value: '{value}'")

        return value


Str.register(str)
Str.register(collections.UserString)


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


class SymbolName(collections.UserString):
    """Name of a symbol."""

    #: Regular expression used to validate the name string
    NAME_REGEX = re.compile(r"[a-zA-Z_]\w*")

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def constrained(regex: str) -> Type[SymbolName]:
        """Create a new SymbolName subclass using the provided string as validation RE."""

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
        return v if isinstance(v, cls) else cls(v)

    def __init__(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError(f"Invalid string argument '{name}'.")
        if not self.NAME_REGEX.fullmatch(name):
            raise ValueError(
                f"Invalid name value '{name}' does not match re({self.NAME_REGEX.pattern})."
            )
        object.__setattr__(self, "data", name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise NotImplementedError(f"Trying to modify immutable {type(self).__name__} value.")

    def __repr__(self) -> str:
        return (
            f"SymbolName('{self.data}')"
            if self.__class__.__name__ == "SymbolName"
            else f"SymbolName.constrained('{self.NAME_REGEX.pattern}')('{self.data}')"
        )


class SymbolRef(collections.UserString):
    """Reference to a symbol."""

    @classmethod
    def __get_validators__(cls) -> PydanticCallableGenerator:
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> SymbolRef:
        return v if isinstance(v, cls) else cls(v)

    def __init__(self, name: str, *, context: Optional[Mapping[str, Any]] = None) -> None:
        if not isinstance(name, str):
            raise TypeError(f"Invalid string argument '{name}'.")
        object.__setattr__(self, "data", name)
        object.__setattr__(self, "_context", context)

    def node(self, *, context: Optional[Mapping[str, Any]] = None) -> Any:
        if context:
            self._context = context
        assert self._context

        return self._context[self.data]

    def __setattr__(self, name: str, value: Any) -> None:
        raise NotImplementedError(f"Trying to modify immutable {type(self).__name__} value.")

    def __repr__(self) -> str:
        return f"{type(self).__name__}('{self.data}')"


class SourceLocation(pydantic.BaseModel):
    """Source code location (line, column, source)."""

    line: PositiveInt
    column: PositiveInt
    source: Str

    def __init__(self, line: int, column: int, source: str) -> None:
        super().__init__(line=line, column=column, source=source)

    def __str__(self) -> str:
        src = self.source or ""
        return f"<{src}: Line {self.line}, Col {self.column}>"

    class Config:
        extra = "forbid"
        allow_mutation = False
