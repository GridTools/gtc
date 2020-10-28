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
import string
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

from ._typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)
from .type_definitions import DELETE, NOTHING


def filter_map(
    func: Callable[..., Any], iterable: Iterable[Any], *, delete_sentinel: Any = DELETE
) -> Iterator[Any]:
    """Mapping function supporting elimination of items.

    Args:
        iterable: iterable object to be processed.
        delete_sentinel: sentinel object which marks items to be deleted from the
            the output (note that comparison is made by identity NOT by value).

    Notes:
        The default `delete_sentinel` value (`type_definitions.DELETE`) can also
        be accessed as `filter_map.DELETE`.

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
    def split(cls, name: str, case_style: CASE_STYLE) -> List[str]:
        assert isinstance(case_style, cls.CASE_STYLE)
        if case_style == cls.CASE_STYLE.CONCATENATED:
            raise ValueError("Impossible to split a simply concatenated string")

        splitter: Callable[[str], List[str]] = getattr(cls, f"split_{case_style.value}_case")
        return splitter(name)

    @classmethod
    def join(cls, words: AnyWordsIterable, case_style: CASE_STYLE) -> str:
        assert isinstance(case_style, cls.CASE_STYLE)
        if isinstance(words, str):
            words = [words]
        if not isinstance(words, collections.abc.Iterable):
            raise TypeError(f"'{words}' type is not a valid sequence of words")

        joiner: Callable[[AnyWordsIterable], str] = getattr(cls, f"join_{case_style.value}_case")
        return joiner(words)

    @classmethod
    def convert(cls, name: str, source_style: CASE_STYLE, target_style: CASE_STYLE) -> str:
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


class XStringFormatter(string.Formatter):
    """Custom :class:`string.Formatter` implementation with f-string-like functionality.

    Implementation follows as close as possible the implementation style
    of :class:`string.Formatter` in the standard library.

    Examples:
        >>> fmt = XStringFormatter()
        >>> data = [1.1, 2.22, 3.333, 4.444]

        >>> fmt.format("{';'.join(str((i, d)) for i, d in enumerate(data))}", data=data)
        '(0, 1.1);(1, 2.22);(2, 3.333);(3, 4.444)'

    Note:
        The current implementation is not 100% f-string compatible due to
        limitations in the `format` specification parser (:meth:`string.Formatter.parser`).
        Most flagrant limitation is that ``{`` and ``}`` characters are strictly forbidden
        inside expressions.

        See `PEP 498 <https://www.python.org/dev/peps/pep-0498/>`_ for more details.

    """

    class __DictLogger(dict):
        """Dumb :class:`dict` subclass logging the accessed args."""

        def __getitem__(self, key: Any) -> Any:
            self.used_args = getattr(self, "used_args", set())
            if key in self:
                self.used_args.add(key)
            return super().__getitem__(key)

    def vformat(self, format_string: str, args: Sequence, kwargs: Mapping) -> str:
        used_args: Set[Union[int, str]] = set()
        result, _ = self._vformat(format_string, args, kwargs, used_args, 2)
        self.check_unused_args(used_args, args, kwargs)  # type: ignore  # likely wrong 'used_args' type in stdlib
        return result

    def _vformat(
        self,
        format_string: str,
        args: Sequence,
        kwargs: Mapping,
        used_args: Set,
        recursion_depth: int,
        auto_arg_index: int = 0,
    ) -> Tuple[str, int]:
        if recursion_depth < 0:
            raise ValueError("Max string recursion exceeded")

        result = []
        _kwargs = self.__DictLogger({**kwargs, "__formatter_args__": args})
        _kwargs.used_args = used_args
        for literal_text, field_name, format_spec, conversion in self.parse(format_string):
            # output the literal text
            if literal_text:
                result.append(literal_text)

            # if there's a field, output it
            if field_name is not None:
                # this is some markup, find the object and do
                #  the formatting

                # handle arg indexing when empty field_names are given.
                if field_name == "":
                    if auto_arg_index is False:
                        raise ValueError(
                            "cannot switch from manual field "
                            "specification to automatic field "
                            "numbering"
                        )
                    field_name = str(auto_arg_index)
                    auto_arg_index += 1
                elif field_name.isdigit():
                    if auto_arg_index:
                        raise ValueError(
                            "cannot switch from manual field "
                            "specification to automatic field "
                            "numbering"
                        )
                    # disable auto arg incrementing, if it gets
                    # used later on, then an exception will be raised
                    auto_arg_index = False

                # given the field_name or expression, get the actual formatted value
                # if a valid arg_used is returned, add it to the used_args set (only for subclasses)
                obj, arg_used = self.get_field(field_name, args, _kwargs)
                if arg_used is not None:
                    used_args.add(arg_used)

                # do any conversion on the resulting object
                obj = self.convert_field(obj, conversion)  # type: ignore  # wrong 'conversion' type in stdlib

                # expand the format spec, if needed
                format_spec, auto_arg_index = self._vformat(
                    format_spec,  # type: ignore  # wrong 'format_spec' type in stdlib
                    args,
                    kwargs,
                    used_args,
                    recursion_depth - 1,
                    auto_arg_index=auto_arg_index,
                )

                # format the object and append to the result
                result.append(self.format_field(obj, format_spec))

        used_args -= {"__formatter_args__"}

        return "".join(result), auto_arg_index

    def get_value(self, key: Union[int, str], args: Sequence, kwargs: Mapping) -> Any:
        assert isinstance(key, str)
        result = eval(key, {}, kwargs)
        return result

    def format_field(self, value: Any, format_spec: str) -> str:
        return format(value, format_spec)

    # given a field_name, find the object it references.
    #  field_name:   the field being looked up or a python expression
    #  args, kwargs: as passed in to vformat
    def get_field(self, field_name: str, args: Sequence, kwargs: Mapping) -> Any:
        used_arg = None
        if field_name.isdigit():
            used_arg = int(field_name)
            field_name = f"__formatter_args__[{field_name}]"

        obj = self.get_value(field_name, args, kwargs)
        return obj, used_arg
