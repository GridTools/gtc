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


from __future__ import annotations

from typing import List, Union

import enum
import random

import boltons
import factory
import pytest
import pytest_factoryboy as pytfboy


import eve
from eve import datamodel


# --- Utils ---
invalid_model_factories = []
model_factory_fixtures = []
model_fixtures = []


def register_factories():
    for name, value in dict(**globals()).items():
        if isinstance(value, type) and issubclass(value, factory.Factory):
            assert name.endswith("Factory")
            factory_fixture_name = boltons.strutils.camel2under(name)
            model_factory_fixtures.append(pytest.lazy_fixture(factory_fixture_name))

            model_fixture_name = boltons.strutils.camel2under(value._meta.model.__name__)
            if factory_fixture_name.endswith(f"{model_fixture_name}_factory"):
                model_fixture_name = factory_fixture_name.replace("_factory", "")
            if value not in invalid_model_factories:
                model_fixtures.append(pytest.lazy_fixture(model_fixture_name))

            pytfboy.register(value, model_fixture_name)


def invalid_model_factory(factory):
    invalid_model_factories.append(factory)
    return factory


# --- Definitions ---
@enum.unique
class Kind(enum.Enum):
    FOO = "foo"
    BLA = "bla"
    FIZ = "fiz"
    FUZ = "fuz"


@enum.unique
class IntKind(enum.IntEnum):
    MINUS = -1
    ZERO = 0
    PLUS = 1


class NonInstantiableModel(datamodel.DataModel, instantiable=False):
    pass


class EmptyModel(datamodel.DataModel):
    pass


class BasicFieldsModel(datamodel.DataModel):
    bool_value: bool
    int_value: int
    float_value: float
    complex_value: complex
    str_value: str
    bytes_value: bytes
    kind: Kind
    int_kind: IntKind


class BasicFieldsModelWithDefaults(datamodel.DataModel):
    bool_value: bool = True
    int_value: int = 1
    float_value: float = 2.0
    complex_value: complex = 1 + 2j
    str_value: str = "string"
    bytes_value: bytes = b"bytes"
    kind: Kind = Kind.FOO
    int_kind: IntKind = IntKind.PLUS


class CompositeModel(datamodel.DataModel):
    basic_model: BasicFieldsModel
    basic_model_with_defaults: BasicFieldsModelWithDefaults


# --- Factories ---
class EmptyModelFactory(factory.Factory):
    class Meta:
        model = EmptyModel


@invalid_model_factory
class NonInstantiableModelFactory(factory.Factory):
    class Meta:
        model = NonInstantiableModel


class BasicFieldsModelFactory(factory.Factory):
    class Meta:
        model = BasicFieldsModel

    bool_value = factory.Faker("pybool")
    int_value = factory.Faker("pyint")
    float_value = factory.Faker("pyfloat")
    complex_value = factory.LazyFunction(lambda: complex(random.random(), random.random()))
    str_value = factory.Faker("pystr")
    bytes_value = factory.LazyFunction(lambda: f"asdf{repr(random.random())}".encode())
    kind = factory.Faker("random_element", elements=Kind)
    int_kind = factory.Faker("random_element", elements=IntKind)


class FixedBasicFieldsModelFactory(BasicFieldsModelFactory):
    bool_value = True
    int_value = 1
    float_value = 1.0
    complex_value = 1 + 2j
    str_value = "a simple string"
    bytes_value = b"sequence of bytes"
    kind = Kind.FOO
    int_kind = IntKind.PLUS


class BasicFieldsModelWithDefaultsFactory(factory.Factory):
    class Meta:
        model = BasicFieldsModelWithDefaults


class CompositeModelFactory(factory.Factory):
    class Meta:
        model = CompositeModel

    basic_model = factory.SubFactory(BasicFieldsModelFactory)
    basic_model_with_defaults = factory.SubFactory(BasicFieldsModelWithDefaultsFactory)


class FixedCompositeModelFactory(factory.Factory):
    class Meta:
        model = CompositeModel

    basic_model = factory.SubFactory(FixedBasicFieldsModelFactory)
    basic_model_with_defaults = factory.SubFactory(BasicFieldsModelWithDefaultsFactory)


# --- Fixtures ---
# Register factories as fixtures using pytest_factoryboy plugin
register_factories()


@pytest.fixture(params=model_factory_fixtures)
def any_model_factory(request):
    return request.param


@pytest.fixture(params=model_fixtures)
def any_model(request):
    return request.param


# --- Tests ---
def test_datamodel_class_members(any_model):
    model = any_model
    assert hasattr(model, "__init__")
    assert hasattr(model, "is_generic") and callable(model.is_generic)
    assert hasattr(model, "__dataclass_fields__") and isinstance(model.__dataclass_fields__, tuple)
    assert hasattr(model, "__datamodel_validators__") and isinstance(
        model.__datamodel_validators__, tuple
    )


def test_non_instantiable(non_instantiable_model_factory):
    with pytest.raises(TypeError, match="Trying to instantiate"):
        non_instantiable_model_factory()


def test_default_values(basic_fields_model_with_defaults_factory):
    model = basic_fields_model_with_defaults_factory()

    assert model.bool_value is True
    assert model.int_value == 1
    assert model.float_value == 2.0
    assert model.complex_value == 1 + 2j
    assert model.str_value == "string"
    assert model.bytes_value == b"bytes"
    assert model.kind == Kind.FOO
    assert model.int_kind == IntKind.PLUS


@pytest.mark.parametrize(
    "basic_fields_model_factory", [BasicFieldsModelFactory, FixedBasicFieldsModelFactory]
)
def test_basic_type_validation(basic_fields_model_factory):
    basic_fields_model_factory()

    with pytest.raises(TypeError, match="bool_value"):
        basic_fields_model_factory(bool_value="WRONG TYPE")
    with pytest.raises(TypeError, match="int_value"):
        basic_fields_model_factory(int_value="WRONG TYPE")
    with pytest.raises(TypeError, match="float_value"):
        basic_fields_model_factory(float_value="WRONG TYPE")
    with pytest.raises(TypeError, match="complex_value"):
        basic_fields_model_factory(complex_value="WRONG TYPE")
    with pytest.raises(TypeError, match="str_value"):
        basic_fields_model_factory(str_value=1.0)
    with pytest.raises(TypeError, match="bytes_value"):
        basic_fields_model_factory(bytes_value=1.0)
    with pytest.raises(TypeError, match="kind"):
        basic_fields_model_factory(kind="WRONG TYPE")
    with pytest.raises(TypeError, match="int_kind"):
        basic_fields_model_factory(int_kind="WRONG TYPE")


@pytest.mark.parametrize("model_factory", [CompositeModelFactory, FixedCompositeModelFactory])
def test_composite_type_validation(model_factory):
    model_factory()

    with pytest.raises(TypeError, match="basic_model"):
        model_factory(basic_model="WRONG TYPE")
    with pytest.raises(TypeError, match="basic_model_with_defaults"):
        model_factory(basic_model_with_defaults="WRONG TYPE")
