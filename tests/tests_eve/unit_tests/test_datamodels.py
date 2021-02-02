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

import dataclasses
import enum
import inspect
import random
import types
import typing
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Literal,
    Mapping,
    MutableSequence,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import factory
import pytest
import pytest_factoryboy as pytfboy

from eve import datamodels, utils


T = TypeVar("T")


# --- Utils and sample data ---
example_model_classes: List[Type] = []
example_model_factories: List[factory.Factory] = []


def register_factory(
    factory_class: Optional[factory.Factory] = None,
    *,
    model_fixture: Optional[str] = None,
    collection: Optional[MutableSequence[factory.Factory]] = None,
) -> Union[factory.Factory, Callable[[factory.Factory], factory.Factory]]:
    def _decorator(factory: factory.Factory) -> factory.Factory:
        if collection is not None:
            collection.append(factory)
        return pytfboy.register(factory, model_fixture)

    return _decorator(factory_class) if factory_class is not None else _decorator


class SampleEnum(enum.Enum):
    FOO = "foo"
    BLA = "bla"


class EmptyModel(datamodels.DataModel):
    pass


@register_factory(collection=example_model_factories)
class EmptyModelFactory(factory.Factory):
    class Meta:
        model = EmptyModel


class IntModel(datamodels.DataModel):
    value: int


@register_factory(collection=example_model_factories)
class IntModelFactory(factory.Factory):
    class Meta:
        model = IntModel

    value = 1


class AnyModel(datamodels.DataModel):
    value: Any


@register_factory(collection=example_model_factories)
class AnyModelFactory(factory.Factory):
    class Meta:
        model = AnyModel

    value = ("any", "value")


class GenericModel(datamodels.DataModel, Generic[T]):
    value: T


@register_factory(collection=example_model_factories)
class GenericModelFactory(factory.Factory):
    class Meta:
        model = GenericModel

    value = "generic value"


example_model_classes = [f._meta.model for f in example_model_factories]


# --- Tests ---
# Test generation
@pytest.mark.parametrize("model_class", example_model_classes)
def test_datamodel_class_members(model_class):
    assert hasattr(model_class, "__init__")
    assert hasattr(model_class, "__datamodel_fields__")
    assert isinstance(model_class.__datamodel_fields__, utils.FrozenNamespace)
    assert hasattr(model_class, "__datamodel_params__")
    assert isinstance(model_class.__datamodel_params__, utils.FrozenNamespace)
    assert hasattr(model_class, "__datamodel_validators__")
    assert isinstance(model_class.__datamodel_validators__, tuple)


@pytest.mark.parametrize("model_class", example_model_classes)
def test_devtools_compatibility(model_class):
    assert hasattr(model_class, "__pretty__")
    assert callable(model_class.__pretty__)


@pytest.mark.parametrize("model_class", example_model_classes)
def test_dataclass_compatibility(model_class):
    assert hasattr(model_class, "__dataclass_fields__") and isinstance(
        model_class.__dataclass_fields__, tuple
    )
    assert set(model_class.__datamodel_fields__.keys()) == set(
        f.name for f in model_class.__dataclass_fields__
    )
    assert dataclasses.is_dataclass(model_class)


def test_init():
    @datamodels.datamodel
    class Model:
        value: int
        enum_value: SampleEnum
        list_value: List[int]

    model = Model(value=1, enum_value=SampleEnum.FOO, list_value=[1, 2, 3])
    assert model.value == 1
    assert model.enum_value == SampleEnum.FOO
    assert model.list_value == [1, 2, 3]


# Test field specification
def test_class_vars():
    @datamodels.datamodel
    class Model:
        value: Any
        default_value: Any = None

        class_var: ClassVar[int] = 0

    field_names = set(Model.__datamodel_fields__.keys())
    assert field_names == {"value", "default_value"}
    assert hasattr(Model, "class_var")
    assert Model.class_var == 0


def test_default_values():
    @datamodels.datamodel
    class Model:
        bool_value: bool = True
        int_value: int
        enum_value: SampleEnum = SampleEnum.FOO
        any_value: Any = datamodels.field(default="ANY")

    model = Model(int_value=1)

    assert model.bool_value is True
    assert model.int_value == 1
    assert model.enum_value == SampleEnum.FOO
    assert model.any_value == "ANY"


def test_default_factories():
    @datamodels.datamodel
    class Model:
        list_value: List[int] = datamodels.field(default_factory=list)
        dict_value: Dict[str, int] = datamodels.field(default_factory=lambda: {"one": 1})

    model = Model()

    assert model.list_value == []
    assert model.dict_value == {"one": 1}


def test_field_metadata():
    @datamodels.datamodel
    class Model:
        value: int = datamodels.field(metadata={"my_metadata": "META"})

    assert isinstance(Model.__datamodel_fields__.value.metadata, types.MappingProxyType)
    assert Model.__datamodel_fields__.value.metadata["my_metadata"] == "META"


# Test datamodel options
def test_non_instantiable():
    @datamodels.datamodel(instantiable=False)
    class NonInstantiableModel:
        value: Any

    assert NonInstantiableModel.__datamodel_params__.instantiable is False
    with pytest.raises(TypeError, match="Trying to instantiate"):
        NonInstantiableModel()

    class NonInstantiableModel2(datamodels.DataModel, instantiable=False):
        value: Any

    assert NonInstantiableModel2.__datamodel_params__.instantiable is False
    with pytest.raises(TypeError, match="Trying to instantiate"):
        NonInstantiableModel2()


# def test_field_redefinition():
#     class IntModel(datamodels.DataModel):
#         value: int

#     model = basic_fields_model
#     field_values = {
#         field.name: getattr(model, field.name) for field in BasicFieldsModel.__dataclass_fields__
#     }

#     # Redefinition with same type
#     class SubModel(BasicFieldsModel):
#         int_value: int

#     SubModel(**field_values)

#     # Redefinition with different type
#     class SubModel(BasicFieldsModel):
#         int_value: float

#     with pytest.raises(TypeError, match="int_value"):
#         SubModel(**field_values)

#     new_field_values = {**field_values}
#     new_field_values.pop("int_value")
#     SubModel(**new_field_values, int_value=1.0)
