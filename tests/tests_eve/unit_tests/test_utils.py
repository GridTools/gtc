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

import copy
import dataclasses
import hashlib
from typing import Any

import pydantic
import pytest

import eve.utils


def test_filter_map():
    from eve.utils import filter_map

    LENGTH = 12

    def map_int_to_string(int_or_float):
        if not isinstance(int_or_float, int):
            return filter_map.DELETE
        else:
            return int_or_float, str(int_or_float)

    assert all(
        isinstance(i, int) and str_i == str(i)
        for i, str_i in filter_map(map_int_to_string, range(LENGTH))
    )

    odds = list(filter_map(lambda x: x if x % 2 else None, range(LENGTH), delete_sentinel=None))
    assert all(isinstance(i, int) and ((i % 2) != 0) for i in odds)
    assert len(odds) == LENGTH // 2


def test_get_item():
    from eve.utils import get_item

    mapping = {
        "true": True,
        1: True,
        "false": False,
        0: False,
    }

    sequence = [False, True, True]

    # Items in collections
    assert get_item(mapping, "true")
    assert not get_item(mapping, "false")
    assert get_item(sequence, 1)
    assert not get_item(sequence, 0)

    # Items in mapping and providing default value
    assert get_item(mapping, "true", False)
    assert not get_item(mapping, "false", True)
    assert get_item(sequence, 1, False)
    assert not get_item(sequence, 0, True)

    # Missing items in mapping and providing default value
    assert get_item(mapping, "", True)
    assert not get_item(mapping, "", False)
    assert get_item(sequence, 1000, True)
    assert not get_item(sequence, 1000, False)

    # Missing items in mapping without providing default value
    with pytest.raises(KeyError):
        assert get_item(mapping, "")
    with pytest.raises(IndexError):
        assert get_item(sequence, 1000)


def test_register_subclasses():
    import abc

    class MyVirtualSubclassA:
        pass

    class MyVirtualSubclassB:
        pass

    @eve.utils.register_subclasses(MyVirtualSubclassA, MyVirtualSubclassB)
    class MyBaseClass(abc.ABC):
        pass

    assert issubclass(MyVirtualSubclassA, MyBaseClass) and issubclass(
        MyVirtualSubclassB, MyBaseClass
    )


class ModelClass(pydantic.BaseModel):
    data: Any


@dataclasses.dataclass
class DataClass:
    data: Any


@pytest.fixture(
    params=[
        [
            1,
            "1",
            True,
            "True",
            "true",
            False,
            (True,),
            [True],
            {True},
            {True: True},
            {True: ()},
            [(True,)],
            ["true"],
        ],
        [0, False, (False,), [False], {False}],
        [(), [], frozenset()],
        [[(1,)], [[1]], [[[1]]], [1], {1}],
        [
            {"a": 0},
            {"A": 0},
            {"b": 0},
            {"a": False},
            {"b": False},
            {"a": "0"},
            {"a": "False"},
            {"a": ("False",)},
            {"a": ["False"]},
            {"a": (False,)},
            {"a": "false"},
            {"a": [0]},
            {"a": [[0]]},
            {"a": [(0,)]},
        ],
    ]
)
def unique_data_items(request):
    input_data = request.param

    yield input_data + [
        DataClass(data=input_data),
        DataClass(data=input_data[0]),
        ModelClass(data=input_data),
        ModelClass(data=input_data[0]),
    ]


@pytest.fixture(
    params=[None, hashlib.md5(), "md5", hashlib.sha1(), "sha1", hashlib.sha256(), "sha256"]
)
def hash_algorithm(request):
    yield request.param


def test_shash(unique_data_items, hash_algorithm):
    from eve.utils import shash

    # Test hash consistency
    for item in unique_data_items:
        if hasattr(hash_algorithm, "copy"):
            h1 = hash_algorithm.copy()
            h2 = hash_algorithm.copy()
        else:
            h1 = hash_algorithm
            h2 = hash_algorithm
        assert shash(item, hash_algorithm=h1) == shash(copy.deepcopy(item), hash_algorithm=h2)

    # Test hash specificity
    hashes = set(shash(item, hash_algorithm=hash_algorithm) for item in unique_data_items)
    assert len(hashes) == len(unique_data_items)


# -- CaseStyleConverter --
@pytest.fixture
def name_with_cases():
    from eve.utils import CaseStyleConverter

    cases = {
        "words": ["first", "second", "UPPER", "Title"],
        CaseStyleConverter.CASE_STYLE.CONCATENATED: "firstseconduppertitle",
        CaseStyleConverter.CASE_STYLE.CANONICAL: "first second upper title",
        CaseStyleConverter.CASE_STYLE.CAMEL: "firstSecondUpperTitle",
        CaseStyleConverter.CASE_STYLE.PASCAL: "FirstSecondUpperTitle",
        CaseStyleConverter.CASE_STYLE.SNAKE: "first_second_upper_title",
        CaseStyleConverter.CASE_STYLE.KEBAB: "first-second-upper-title",
    }

    yield cases


def test_case_style_converter(name_with_cases):
    from eve.utils import CaseStyleConverter

    words = name_with_cases.pop("words")
    for case, cased_string in name_with_cases.items():
        # Try also passing case as a string
        if len(case.value) % 2:
            case = case.value

        assert CaseStyleConverter.join(words, case) == cased_string
        if case == CaseStyleConverter.CASE_STYLE.CONCATENATED:
            with pytest.raises(ValueError, match="Impossible to split"):
                CaseStyleConverter.split(cased_string, case)
        else:
            assert [w.lower() for w in CaseStyleConverter.split(cased_string, case)] == [
                w.lower() for w in words
            ]


# -- TestUIDGenerator --
class TestUIDGenerator:
    def test_random_id(self):
        from eve.utils import UIDGenerator

        a = UIDGenerator.random_id()
        b = UIDGenerator.random_id()
        c = UIDGenerator.random_id()
        assert a != b and a != c and b != c
        assert UIDGenerator.random_id(prefix="abcde").startswith("abcde")
        assert len(UIDGenerator.random_id(width=10)) == 10
        with pytest.raises(ValueError, match="Width"):
            UIDGenerator.random_id(width=-1)
        with pytest.raises(ValueError, match="Width"):
            UIDGenerator.random_id(width=4)

    def test_sequential_id(self):
        from eve.utils import UIDGenerator

        i = UIDGenerator.sequential_id()
        assert UIDGenerator.sequential_id() != i
        assert UIDGenerator.sequential_id(prefix="abcde").startswith("abcde")
        assert len(UIDGenerator.sequential_id(width=10)) == 10
        assert not UIDGenerator.sequential_id().startswith("0")
        with pytest.raises(ValueError, match="Width"):
            UIDGenerator.sequential_id(width=-1)

    def test_reset_sequence(self):
        from eve.utils import UIDGenerator

        i = UIDGenerator.sequential_id()
        counter = int(i)
        UIDGenerator.reset_sequence(counter + 1)
        assert int(UIDGenerator.sequential_id()) == counter + 1
        with pytest.warns(RuntimeWarning, match="Unsafe reset"):
            UIDGenerator.reset_sequence(counter)
