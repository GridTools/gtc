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


from __future__ import annotations

import collections.abc
import enum
import functools
import hashlib
import itertools
import pickle
import re
import types
import typing
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

from .type_definitions import NOTHING
from .typingx import (
    Any,
    AnyCallable,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)


try:
    import cytoolz as toolz
except ModuleNotFoundError:
    import toolz


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


# -- Iterators --

T = TypeVar("T")
S = TypeVar("S")


def as_xiter(iterator_func: Callable[..., Iterator[T]]) -> Callable[..., XIterator[T]]:
    """Wrap the provided callable to convert its output in a :class:`XIterator`."""

    @functools.wraps(iterator_func)
    def _xiterator(*args: Any, **keywords: Any) -> XIterator[T]:
        return xiter(iterator_func(*args, **keywords))

    return _xiterator


def xiter(iterable: Iterable[T]) -> XIterator[T]:
    """Create an XIterator from any iterable (like ``iter()``)."""

    if isinstance(iterable, collections.abc.Iterator):
        it = iterable
    elif isinstance(iterable, collections.abc.Iterable):
        it = iter(iterable)
    else:
        raise ValueError(f"Invalid iterable instance: '{iterable}'.")

    return XIterator(it)


xenumerate = as_xiter(enumerate)


def xiter_args(it: Iterable[T], **kwargs: Any) -> FrozenArgs:
    """Create a :class:`FrozenArgs` instance passing extra keywords for :class:`XIterator` operators."""
    return FrozenArgs(it, **kwargs)


class FrozenArgs:
    """Immutable object catching function call args and kwargs."""

    args: Tuple[Any, ...]
    kwargs: Mapping[str, Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__setattr__("args", args)
        super().__setattr__("kwargs", types.MappingProxyType(kwargs))

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError(f"{type(self).__name__} is immutable.")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(args={self.args}, kwargs={self.kwargs})"


class XIterator(collections.abc.Iterator, Iterable[T]):
    """Iterator wrapper defining operators for extra functionality.

    Features:

        - ``[]``: select elements from an iterable (equivalent to ``itertools.islice(self.iterator, start, stop, step)``).
          For detailed information check :func:`itertools.islice` reference.

            >>> it = xiter(range(10))
            >>> list(it[2:8:2])
            [2, 4, 6]

        - ``&``: filter elements with callables (equivalent to ``filter(callable, iterator)``).
          For detailed information check :func:`filter` reference.

            >>> it = xiter(range(4))
            >>> list(it & (lambda x: x % 2 == 0))
            [0, 2]

            >>> it = xiter(range(4))
            >>> list(it & (lambda x: x % 2 == 0) & (lambda x: x > 1))
            [2]

        - ``|``: apply a callable to every iterator element (equivalent to ``map(callable, iterator)``).
          For detailed information check :func:`map` reference.

            >>> it = xiter(range(3))
            >>> list(it | str)
            ['0', '1', '2']

            >>> it = xiter(range(3))
            >>> list(it | (lambda x: -x) | str)
            ['0', '-1', '-2']

            If the callable requires additional arguments, ``lambda`` of :func:`functools.partial`
            functions can be used:

            >>> def times(a, b):
            ...     return a * b
            >>> times_2 = functools.partial(times, 2)
            >>> it = xiter(range(4))
            >>> list(it | (lambda x: x + 1) | times_2)
            [2, 4, 6, 8]

            Curried functions generated by :func:`toolz.curry` will also work as expected:

            >>> @toolz.curry
            ... def mul(x, y):
            ...     return x * y
            >>> it = xiter(range(4))
            >>> list(it | (lambda x: x + 1) | mul(5))
            [5, 10, 15, 20]

        - ``@``: pick data from each item in a sequence (equivalent to ``toolz.itertoolz.pluck(index, iterator)``).
          For detailed information check :func:`toolz.itertoolz.pluck` reference.

            >>> it = xiter([('a', 1), ('b', 2), ('c', 3)])
            >>> list(it @ 0)
            ['a', 'b', 'c']

            >>> it = xiter([
            ...     dict(name="AA", age=20, country="US"),
            ...     dict(name="BB", age=30, country="UK"),
            ...     dict(name="CC", age=40, country="CH")
            ... ])
            >>> list(it @ ["name", "age"])
            [('AA', 20), ('BB', 30), ('CC', 40)]

        - ``+``: chain iterators (equivalent to ``itertools.chain(it_a, it_b)``).
          For detailed information check :func:`itertools.chain` reference.

            >>> it_a, it_b = xiter(range(2)), xiter(['a', 'b'])
            >>> list(it_a + it_b)
            [0, 1, 'a', 'b']

            >>> it_b = xiter(['a', 'b'])
            >>> list([0, 1] + it_b)
            [0, 1, 'a', 'b']

        - ``-``: diff iterators (equivalent to ``toolz.itertoolz.diff(it_a, it_b)``).
          For detailed information check :func:`toolz.itertoolz.diff` reference.

            >>> it_a, it_b = xiter([1, 2, 3]), xiter([1, 3, 5])
            >>> list(it_a - it_b)
            [(2, 3), (3, 5)]

            >>> it_b = xiter([1, 3, 5])
            >>> list([1, 2, 3] - it_b)
            [(2, 3), (3, 5)]

            Adding missing values:

            >>> it_a, it_b = xiter([1, 2, 3, 4]), xiter([1, 3, 5])
            >>> list(it_a - xiter_args(it_b, default=None))
            [(2, 3), (3, 5), (4, None)]

            >>> it_b = xiter([1, 3, 5])
            >>> list(xiter_args([1, 2, 3, 4], default=None) - it_b)
            [(2, 3), (3, 5), (4, None)]


        - ``*``: product of iterators (equivalent to ``itertools.product(it_a, it_b)``).
          For detailed information check :func:`itertools.product` reference.

            >>> it_a, it_b = xiter([0, 1]), xiter(['a', 'b'])
            >>> list(it_a * it_b)
            [(0, 'a'), (0, 'b'), (1, 'a'), (1, 'b')]

            >>> it_b = xiter(['a', 'b'])
            >>> list([0, 1] * it_b)
            [(0, 'a'), (0, 'b'), (1, 'a'), (1, 'b')]

            Product of an iterator with itself:

            >>> it_a = xiter([0, 1])
            >>> list(it_a * 3)
            [(0, 0, 0), (1, 1, 1)]

            >>> it_a = xiter([0, 1])
            >>> list(3 * it_a)
            [(0, 0, 0), (1, 1, 1)]

        - ``/``: partition iterator into tuples of length at most ``n``
          (equivalent to ``toolz.itertoolz.partition_all(n, iterator)``).
          For detailed information check :func:`toolz.itertoolz.partition_all` reference.

            >>> it = xiter(range(7))
            >>> list(it / 3)
            [(0, 1, 2), (3, 4, 5), (6,)]

        - ``//``: partition iterator into tuples of length ``n``
          (equivalent to ``toolz.itertoolz.partition(n, iterator)``).
          For detailed information check :func:`toolz.itertoolz.partition` reference.

            >>> it = xiter(range(7))
            >>> list(it // 3)
            [(0, 1, 2), (3, 4, 5)]

            >>> it = xiter(range(7))
            >>> list(it // xiter_args(3, pad=None))
            [(0, 1, 2), (3, 4, 5), (6, None, None)]

        - ``%``: take every nth item in sequence (equivalent to ``toolz.itertoolz.take_nth(n, iterator)``).
          For detailed information check :func:`toolz.itertoolz.take_nth` reference.

            >>> it = xiter(range(7))
            >>> list(it % 3)
            [0, 3, 6]

        - ``**``: zip iterators (equivalent to ``zip(it_a, it_b)``).
          For detailed information check :func:`zip` reference.

            >>> it_a = xiter(range(3))
            >>> it_b = ['a', 'b', 'c']
            >>> list(it_a ** it_b)
            [(0, 'a'), (1, 'b'), (2, 'c')]

            >>> it_a = range(3)
            >>> it_b = xiter(['a', 'b', 'c'])
            >>> list(it_a ** it_b)
            [(0, 'a'), (1, 'b'), (2, 'c')]

        - ``~``: unzip iterator (equivalent to ``zip(*iterator)``).
          For detailed information check :func:`zip` reference.

            >>> it = xiter([('a', 1), ('b', 2), ('c', 3)])
            >>> list(~it)
            [('a', 'b', 'c'), (1, 2, 3)]

    """

    iterator: Iterator[T]

    def __init__(self, it: Union[Iterable[T], Iterator[T]]) -> None:
        if not isinstance(it, collections.abc.Iterator):
            raise ValueError(f"Invalid iterator instance: '{it}'.")
        if isinstance(it, XIterator):
            self.iterator = it.iterator
        else:
            self.iterator = it

    def __getattr__(self, name: str) -> Any:
        # Forward special methods to wrapped iterator
        if name.startswith("__") and name.endswith("__"):
            return getattr(self.iterator, name)

    def __next__(self) -> T:
        return next(self.iterator)

    def __getitem__(self, index: Union[Tuple[int, ...], slice]) -> XIterator[T]:
        if not isinstance(index, slice):
            if isinstance(index, int):
                index = slice(index)
            elif isinstance(index, tuple):
                index = slice(*index)
            else:
                raise ValueError(f"Invalid slice object: '{index}'.")

        return XIterator(itertools.islice(self.iterator, index.start, index.stop, index.step))

    def __and__(self, func: Callable[..., bool]) -> XIterator[T]:
        if not callable(func):
            raise ValueError(f"Invalid function or callable: '{func}'.")
        return XIterator(filter(func, self.iterator))

    def __or__(self, func: AnyCallable) -> XIterator[Any]:
        if not callable(func):
            raise ValueError(f"Invalid function or callable: '{func}'.")
        return XIterator(map(func, self.iterator))

    def __matmul__(self, index: Union[int, str, List[Union[int, str]]]) -> XIterator[Any]:
        if isinstance(index, collections.abc.Iterable) and not isinstance(index, list):
            index = list(index)
        return XIterator(toolz.itertoolz.pluck(index, self.iterator))

    def __add__(self, other: Iterable[S]) -> XIterator[Union[T, S]]:
        if not isinstance(other, XIterator):
            other = xiter(other)
        return XIterator(itertools.chain(self.iterator, other.iterator))

    def __sub__(self, other: Union[Iterable[S], FrozenArgs]) -> XIterator[Tuple[T, S]]:
        kwargs: Dict[str, Any] = {}
        if isinstance(other, FrozenArgs):
            assert len(other.args) == 1
            kwargs = {**other.kwargs}
            unknown_options = set(kwargs.keys()) - {"default", "key"}
            if unknown_options:
                raise ValueError(f"Unknown options: {unknown_options}")
            other = other.args[0]

        if not isinstance(other, XIterator):
            other = xiter(other)  # type: ignore  # impossible to verify at compile-time
        return XIterator(toolz.itertoolz.diff(self.iterator, other.iterator, **kwargs))

    def __mul__(
        self, other: Union[Iterable[S], int]
    ) -> Union[XIterator[Tuple[T, S]], XIterator[Tuple[T, T]]]:
        if isinstance(other, int):
            if other < 0:
                raise ValueError(
                    f"Only non-negative integer numbers are accepted (provided: {other})."
                )
            return XIterator(map(lambda item: tuple([item] * other), self.iterator))  # type: ignore  # mypy gets confused with `other`
        else:
            if not isinstance(other, XIterator):
                other = xiter(other)
        return XIterator(itertools.product(self.iterator, other.iterator))

    def __truediv__(self, other: int) -> XIterator[Tuple[T, ...]]:
        if not isinstance(other, int) or other < 1:
            raise ValueError(f"Only positive integer numbers are accepted (provided: {other}).")
        return XIterator(toolz.itertoolz.partition_all(other, self.iterator))

    def __floordiv__(self, other: int) -> XIterator[Tuple[T, ...]]:
        kwargs: Dict[str, Any] = {}
        if isinstance(other, FrozenArgs):
            assert len(other.args) == 1
            kwargs = {**other.kwargs}
            unknown_options = set(kwargs.keys()) - {"pad"}
            if unknown_options:
                raise ValueError(f"Unknown options: {unknown_options}")
            other = other.args[0]

        if not isinstance(other, int) or other < 1:
            raise ValueError(f"Only positive integer numbers are accepted (provided: {other}).")
        return XIterator(toolz.itertoolz.partition(other, self.iterator, **kwargs))

    def __mod__(self, other: int) -> XIterator[T]:
        if not isinstance(other, int) or other < 1:
            raise ValueError(f"Only positive integer numbers are accepted (provided: {other}).")
        return XIterator(toolz.itertoolz.take_nth(other, self.iterator))

    def __pow__(self, other: Iterable[S]) -> XIterator[Tuple[T, S]]:
        if not isinstance(other, XIterator):
            other = xiter(other)
        return XIterator(zip(self.iterator, other.iterator))

    def __invert__(self) -> XIterator[Tuple[Any, ...]]:
        return XIterator(zip(*self.iterator))  # type: ignore  # mypy gets confused with *args

    def __radd__(self, other: Iterable[S]) -> XIterator[Union[T, S]]:
        return xiter(other).__add__(self)

    def __rsub__(self, other: Union[Iterable[S], FrozenArgs]) -> XIterator[Tuple[S, T]]:
        kwargs: Dict[str, Any] = {}
        if isinstance(other, FrozenArgs):
            assert len(other.args) == 1
            kwargs = {**other.kwargs}
            other = other.args[0]
        assert not isinstance(other, FrozenArgs)
        return xiter(other).__sub__(xiter_args(self, **kwargs))

    def __rmul__(
        self, other: Union[Iterable[S], int]
    ) -> Union[XIterator[Tuple[S, T]], XIterator[Tuple[T, T]]]:
        if isinstance(other, int):
            return self.__mul__(other)  # type: ignore  # mypy gets confused and the actual type is hard to express
        else:
            return xiter(other).__mul__(self)  # type: ignore  # mypy gets confused and the actual type is hard to express

    def __rpow__(self, other: Iterable[S]) -> XIterator[Tuple[S, T]]:
        return typing.cast(XIterator[S], xiter(other)).__pow__(self)
