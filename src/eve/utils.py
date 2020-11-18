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

"""General utility functions. Some functionalities are directly imported from dependencies."""


import collections.abc
import enum
import hashlib
import itertools
import pickle
import re
import uuid
import warnings

import xxhash
from boltons.iterutils import flatten, flatten_iter  # noqa: F401
from boltons.strutils import (  # noqa: F401
    a10n,
    asciify,
    format_int_list,
    iter_splitlines,
    parse_int_list,
    slugify,
    unwrap_text,
)
from boltons.typeutils import classproperty  # noqa: F401

from .type_definitions import DELETE, NOTHING
from .typingx import Any, Callable, Iterable, Iterator, List, Optional, Type, Union


def filter_map(
    func: Callable[..., Any], iterable: Iterable[Any], *, delete_sentinel: Any = DELETE
) -> Iterator[Any]:
    """Mapping function supporting elimination of items.

    Args:
        iterable: iterable object to be processed.
        delete_sentinel: sentinel object which marks items to be deleted from the
            the output (note that comparison is made by identity NOT by value).

    Notes:
        The default `delete_sentinel` value (:data:`eve.DELETE`) can also be
        accessed as :data:`filter_map.DELETE`.

    """
    for item in iterable:
        result = func(item)
        if result is not delete_sentinel:
            yield result


#: Shortcut to the global DELETE sentinel value
filter_map.DELETE = DELETE  # type: ignore


def get_item(obj: Any, key: Any, default: Any = NOTHING) -> Any:
    """Similar to :func:`operator.getitem()` accepting a default value."""

    if default is NOTHING:
        result = obj[key]
    else:
        try:
            result = obj[key]
        except (KeyError, IndexError):
            result = default

    return result


def register_subclasses(*subclasses: Type) -> Callable[[Type], Type]:
    """Class decorator to automatically register virtual subclasses.

    Example:
        >>> import abc
        >>> class MyVirtualSubclassA:
        ...     pass
        ...
        >>> class MyVirtualSubclassB:
        ...    pass
        ...
        >>> @register_subclasses(MyVirtualSubclassA, MyVirtualSubclassB)
        ... class MyBaseClass(abc.ABC):
        ...    pass
        ...
        >>> issubclass(MyVirtualSubclassA, MyBaseClass) and issubclass(MyVirtualSubclassB, MyBaseClass)
        True

    """

    def _decorator(base_cls: Type) -> Type:
        for s in subclasses:
            base_cls.register(s)
        return base_cls

    return _decorator


def shash(*args: Any, hash_algorithm: Optional[Any] = None) -> str:
    """Stable hash function.

    It provides a customizable hash function for any kind of data.
    Unlike the builtin `hash` function, it is stable (same hash value across
    interpreter reboots) and it does not use hash customizations on user
    classes (it uses `pickle` internally to get a byte stream).

    Args:
        hash_algorithm: object implementing the `hash algorithm` interface
            from :mod:`hashlib` or canonical name (`str`) of the
            hash algorithm as defined in :mod:`hashlib`.
            Defaults to :class:`xxhash.xxh64`.

    """

    if hash_algorithm is None:
        hash_algorithm = xxhash.xxh64()
    elif isinstance(hash_algorithm, str):
        hash_algorithm = hashlib.new(hash_algorithm)

    hash_algorithm.update(pickle.dumps(args))
    result = hash_algorithm.hexdigest()
    assert isinstance(result, str)

    return result


AnyWordsIterable = Union[str, Iterable[str]]


class CaseStyleConverter:
    """Utility class to convert name strings to different case styles.

    Functionality exposed through :meth:`split()`, :meth:`join()` and
    :meth:`convert()` methods.

    """

    class CASE_STYLE(enum.Enum):
        CONCATENATED = "concatenated"
        CANONICAL = "canonical"
        CAMEL = "camel"
        PASCAL = "pascal"
        SNAKE = "snake"
        KEBAB = "kebab"

    @classmethod
    def split(cls, name: str, case_style: Union[CASE_STYLE, str]) -> List[str]:
        if isinstance(case_style, str):
            case_style = cls.CASE_STYLE(case_style)
        assert isinstance(case_style, cls.CASE_STYLE)
        if case_style == cls.CASE_STYLE.CONCATENATED:
            raise ValueError("Impossible to split a simply concatenated string")

        splitter: Callable[[str], List[str]] = getattr(cls, f"split_{case_style.value}_case")
        return splitter(name)

    @classmethod
    def join(cls, words: AnyWordsIterable, case_style: Union[CASE_STYLE, str]) -> str:
        if isinstance(case_style, str):
            case_style = cls.CASE_STYLE(case_style)
        assert isinstance(case_style, cls.CASE_STYLE)
        if isinstance(words, str):
            words = [words]
        if not isinstance(words, collections.abc.Iterable):
            raise TypeError(f"'{words}' type is not a valid sequence of words")

        joiner: Callable[[AnyWordsIterable], str] = getattr(cls, f"join_{case_style.value}_case")
        return joiner(words)

    @classmethod
    def convert(
        cls, name: str, source_style: Union[CASE_STYLE, str], target_style: Union[CASE_STYLE, str]
    ) -> str:
        return cls.join(cls.split(name, source_style), target_style)

    # Following `join_...`` functions are based on:
    #    https://blog.kangz.net/posts/2016/08/31/code-generation-the-easier-way/
    #
    @staticmethod
    def join_concatenated_case(words: AnyWordsIterable) -> str:
        words = [words] if isinstance(words, str) else words
        return "".join(words).lower()

    @staticmethod
    def join_canonical_case(words: AnyWordsIterable) -> str:
        words = [words] if isinstance(words, str) else words
        return (" ".join(words)).lower()

    @staticmethod
    def join_camel_case(words: AnyWordsIterable) -> str:
        words = [words] if isinstance(words, str) else list(words)
        return words[0].lower() + "".join(word.title() for word in words[1:])

    @staticmethod
    def join_pascal_case(words: AnyWordsIterable) -> str:
        words = [words] if isinstance(words, str) else words
        return "".join(word.title() for word in words)

    @staticmethod
    def join_snake_case(words: AnyWordsIterable) -> str:
        words = [words] if isinstance(words, str) else words
        return "_".join(words).lower()

    @staticmethod
    def join_kebab_case(words: AnyWordsIterable) -> str:
        words = [words] if isinstance(words, str) else words
        return "-".join(words).lower()

    # Following `split_...`` functions are based on:
    #    https://stackoverflow.com/a/29920015/7232525
    #
    @staticmethod
    def split_canonical_case(name: str) -> List[str]:
        return name.split()

    @staticmethod
    def split_camel_case(name: str) -> List[str]:
        matches = re.finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", name)
        return [m.group(0) for m in matches]

    split_pascal_case = split_camel_case

    @staticmethod
    def split_snake_case(name: str) -> List[str]:
        return name.split("_")

    @staticmethod
    def split_kebab_case(name: str) -> List[str]:
        return name.split("-")


class UIDGenerator:
    """Simple unique id generator using different methods."""

    #: Constantly increasing counter for generation of sequential unique ids
    __counter = itertools.count(1)

    @classmethod
    def random_id(cls, *, prefix: Optional[str] = None, width: int = 8) -> str:
        """Generate a random globally unique id."""

        if width is not None and width <= 4:
            raise ValueError(f"Width must be a positive number > 4 ({width} provided).")
        u = uuid.uuid4()
        s = str(u).replace("-", "")[:width]
        return f"{prefix}_{s}" if prefix else f"{s}"

    @classmethod
    def sequential_id(cls, *, prefix: Optional[str] = None, width: Optional[int] = None) -> str:
        """Generate a sequential unique id (for the current session)."""

        if width is not None and width < 1:
            raise ValueError(f"Width must be a positive number ({width} provided).")
        count = next(cls.__counter)
        s = f"{count:0{width}}" if width else f"{count}"
        return f"{prefix}_{s}" if prefix else f"{s}"

    @classmethod
    def reset_sequence(cls, start: int = 1) -> None:
        """Reset global generator counter.

        Notes:
            If the new start value is lower than the last generated UID, new
            IDs are not longer guaranteed to be unique.

        """
        if start < next(cls.__counter):
            warnings.warn("Unsafe reset of global UIDGenerator", RuntimeWarning)
        cls.__counter = itertools.count(start)
