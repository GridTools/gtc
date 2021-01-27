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
    ClassVar,
    Dict,
    Generic,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import boltons
import factory
import pytest
import pytest_factoryboy as pytfboy

from eve import datamodels, utils


# --- Utils ---
invalid_model_factories = []
model_factory_fixtures = []
model_instance_fixtures = []


def register_factories():
    """Register factoryboy factory classes as pytest fixtures."""

    for name, value in dict(**globals()).items():
        if isinstance(value, type) and issubclass(value, factory.Factory):
            assert name.endswith("Factory")
            factory_fixture_name = boltons.strutils.camel2under(name)
            model_factory_fixtures.append(pytest.lazy_fixture(factory_fixture_name))

            model_fixture_name = boltons.strutils.camel2under(value._meta.model.__name__)
            if factory_fixture_name.endswith(f"{model_fixture_name}_factory"):
                model_fixture_name = factory_fixture_name.replace("_factory", "")
            if value not in invalid_model_factories:
                model_instance_fixtures.append(pytest.lazy_fixture(model_fixture_name))

            pytfboy.register(value, model_fixture_name)


def invalid_model_factory(factory):
    invalid_model_factories.append(factory)
    return factory


# --- Models, factories and fixtures ---
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


# NonInstantiableModel
class NonInstantiableModel(datamodels.DataModel, instantiable=False):
    pass


@invalid_model_factory
class NonInstantiableModelFactory(factory.Factory):
    class Meta:
        model = NonInstantiableModel


# EmptyModel
class EmptyModel(datamodels.DataModel):
    pass


class EmptyModelFactory(factory.Factory):
    class Meta:
        model = EmptyModel


# BasicFieldsModel
class BasicFieldsModel(datamodels.DataModel):
    bool_value: bool
    int_value: int
    float_value: float
    complex_value: complex
    str_value: str
    bytes_value: bytes
    kind: Kind
    int_kind: IntKind
    any_value: Any

    class_var: ClassVar[int] = 0


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
    any_value = factory.Faker(
        "random_element", elements=[None, 1, 1.0, 1j, "one", [], {}, tuple(), set(), lambda x: x]
    )


class FixedBasicFieldsModelFactory(BasicFieldsModelFactory):
    bool_value = True
    int_value = 1
    float_value = 1.0
    complex_value = 1 + 2j
    str_value = "a simple string"
    bytes_value = b"sequence of bytes"
    kind = Kind.FOO
    int_kind = IntKind.PLUS
    any_value = "Any"


# BasicFieldsModelWithDefaults
class BasicFieldsModelWithDefaults(datamodels.DataModel):
    bool_value: bool = True
    int_value: int = 1
    float_value: float = 2.0
    complex_value: complex = 1 + 2j
    str_value: str = "string"
    bytes_value: bytes = b"bytes"
    kind: Kind = Kind.FOO
    int_kind: IntKind = IntKind.PLUS
    any_value: Any = None

    class_var: ClassVar[int] = 0


class BasicFieldsModelWithDefaultsFactory(factory.Factory):
    class Meta:
        model = BasicFieldsModelWithDefaults


# AdvancedFieldsModel
class AdvancedFieldsModel(datamodels.DataModel):
    str_list: List[str]
    int_set: Set[int]
    float_sequence: Sequence[float]
    int_float_dict: Dict[int, float]
    str_float_map: Mapping[str, float]
    int_float_tuple: Tuple[int, float]
    int_tuple: Tuple[int, ...]
    int_float_str_union: Union[int, float, str]
    opt_float: Optional[float]
    opt_int_kind: Optional[IntKind]
    opt_int_str_union: Optional[Union[int, str]]
    tuple_with_opt_union: Tuple[int, Optional[Union[int, str]]]
    five_literal: Literal[5]
    true_literal: Literal[True]
    nested_dict: Dict[Union[int, str], List[Optional[Tuple[str, str, int]]]]

    class_var: ClassVar[Dict[str, int]] = {"a": 0}


class AdvancedFieldsModelFactory(factory.Factory):
    class Meta:
        model = AdvancedFieldsModel

    str_list = ["a", "b", "c", "d"]
    int_set = {1, 2, 3, 4}
    float_sequence = [1.1, 2.2, 3.3, 4.4]
    int_float_dict = {1: 1.1, 2: 2.2}
    str_float_map = {"pi": 3.14159}
    int_float_tuple = (1, 1.1)
    int_tuple = (1,)
    int_float_str_union = 3
    opt_float = 2.34
    opt_int_kind = IntKind.PLUS
    opt_int_str_union = "string"
    tuple_with_opt_union = (1, 2)
    five_literal = 5
    true_literal = True
    nested_dict = {"empty": [], 0: [], 1: [("a", "b", 10), None, None]}


class OtherAdvancedFieldsModelFactory(AdvancedFieldsModelFactory):
    str_list = []
    float_sequence = tuple()
    int_float_dict = {}
    str_float_map = types.MappingProxyType({"pi": 3.14159})
    int_tuple = tuple()
    int_float_str_union = "three"
    opt_float = None
    opt_int_kind = None
    opt_int_str_union = None
    tuple_with_opt_union = (1, None)
    nested_dict = {"empty": [None, None, ("a", "b", 3)]}


# CompositeModel
class CompositeModel(datamodels.DataModel):
    basic_model: BasicFieldsModel
    basic_model_with_defaults: BasicFieldsModelWithDefaults


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


# ModelWithValidators
class ModelWithValidators(datamodels.DataModel):
    bool_value: bool
    int_value: int
    even_int_value: int
    float_value: float
    str_value: str
    extra_value: Optional[Any] = None

    @datamodels.validator("bool_value")
    def _bool_value_validator(self, attribute, value):
        assert isinstance(self, ModelWithValidators)

    @datamodels.validator("int_value")
    def _int_value_validator(self, attribute, value):
        if value < 0:
            raise ValueError(f"'{attribute.name}' must be larger or equal to 0")

    @datamodels.validator("even_int_value")
    def _even_int_value_validator(self, attribute, value):
        if value % 2:
            raise ValueError(f"'{attribute.name}' must be an even number")

    @datamodels.validator("float_value")
    def _float_value_validator(self, attribute, value):
        if value > 3.14159:
            raise ValueError(f"'{attribute.name}' must be smaller or equal to 3.14159")

    @datamodels.validator("str_value")
    def _str_value_validator(self, attribute, value):
        if value == str(self.float_value):
            raise ValueError(f"'{attribute.name}' must be different to 'float_value'")

    @datamodels.validator("extra_value")
    def _extra_value_validator(self, attribute, value):
        if bool(value):
            raise ValueError(f"'{attribute.name}' must be equivalent to False")


class ModelWithValidatorsFactory(factory.Factory):
    class Meta:
        model = ModelWithValidators

    bool_value = False
    int_value = 0
    even_int_value = 2
    float_value = 0.0
    str_value = ""
    extra_value = False


# InheritedModelWithValidators
class InheritedModelWithValidators(ModelWithValidators):
    # bool_value, int_value, even_int_value -> no redefinition
    float_value: float  # redefined without decorator
    str_value: str  # redefined with decorator
    extra_value: bool  # redefined with decorator and a different type
    new_int_value: int  # added field

    @datamodels.validator("str_value")
    def _str_value_validator(self, attribute, value):
        # Redeclare using same validator function name
        if len(value) > 5:
            raise ValueError(f"'{attribute.name}' must contain 5 chars or less")

    @datamodels.validator("extra_value")
    def _another_extra_value_validator(self, attribute, value):
        # Redeclare using a different validator function name
        if value is True:
            raise ValueError(f"'{attribute.name}' must be False")

    @datamodels.validator("new_int_value")
    def _int_value_validator(self, attribute, value):
        # Using a name already existing in the superclass for the validator function
        if self.int_value != value:
            raise ValueError(f"'{attribute.name}' value must be equal to 'extra_value'")


class InheritedModelWithValidatorsFactory(factory.Factory):
    class Meta:
        model = InheritedModelWithValidators

    bool_value = False
    int_value = 0
    even_int_value = 2
    float_value = 0.0
    str_value = ""
    extra_value = False
    new_int_value = 0


# ModelWithRootValidators
class ModelWithRootValidators(datamodels.DataModel):
    int_value: int
    float_value: float
    str_value: str

    class_counter: ClassVar[int] = 0

    @datamodels.root_validator
    def _root_validator(cls, instance):
        assert cls is type(instance)
        assert issubclass(cls, ModelWithRootValidators)
        assert isinstance(instance, ModelWithRootValidators)
        cls.class_counter = 0

    @datamodels.root_validator
    def _another_root_validator(cls, instance):
        assert cls.class_counter == 0
        cls.class_counter += 1

    @datamodels.root_validator
    def _final_root_validator(cls, instance):
        assert cls.class_counter == 1
        cls.class_counter += 1

        if instance.int_value == instance.float_value:
            raise ValueError("'int_value' and 'float_value' must be different")


class ModelWithRootValidatorsFactory(factory.Factory):
    class Meta:
        model = ModelWithRootValidators

    int_value = 0
    float_value = 1.1
    str_value = ""


# InheritedModelWithRootValidators
class InheritedModelWithRootValidators(ModelWithRootValidators):
    @datamodels.root_validator
    def _root_validator(cls, instance):
        assert cls.class_counter == 2
        cls.class_counter += 10

    @datamodels.root_validator
    def _another_root_validator(cls, instance):
        assert cls.class_counter == 12
        cls.class_counter += 10

    @datamodels.root_validator
    def _final_root_validator(cls, instance):
        assert cls.class_counter == 22
        if str(instance.int_value) == instance.str_value:
            raise ValueError("'int_value' and 'str_value' must be different")


class InheritedModelWithRootValidatorsFactory(factory.Factory):
    class Meta:
        model = InheritedModelWithRootValidators

    int_value = 0
    float_value = 1.1
    str_value = ""


# GenericModel
T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U", bound=int)


class SimpleGenericModel(datamodels.DataModel, Generic[T]):
    generic_value: T
    int_value: int = 0


class SimpleGenericModelFactory(factory.Factory):
    class Meta:
        model = SimpleGenericModel

    generic_value = None
    int_value = 1


class AdvancedGenericModel(SimpleGenericModel[T], Generic[U, S, T]):
    generic_value: T
    int_value: int = 0


# @pytest.fixture(params=[int, float, str, complex])
# def instantiated_simple_generic_model_class(request):
#     return SimpleGenericModel[request.param]


# Register factories as fixtures using pytest_factoryboy plugin
register_factories()


@pytest.fixture(params=model_factory_fixtures)
def any_model_factory(request):
    return request.param


@pytest.fixture(params=model_instance_fixtures)
def any_model_instance(request):
    return request.param


# --- Tests ---
def test_datamodel_class_members(any_model_instance):
    assert hasattr(any_model_instance, "__init__")
    assert hasattr(any_model_instance, "__datamodel_fields__") and isinstance(
        any_model_instance.__datamodel_fields__, utils.FrozenNamespace
    )
    assert hasattr(any_model_instance, "__datamodel_params__") and isinstance(
        any_model_instance.__datamodel_params__, utils.FrozenNamespace
    )
    assert hasattr(any_model_instance, "__datamodel_validators__") and isinstance(
        any_model_instance.__datamodel_validators__, tuple
    )
    assert hasattr(any_model_instance, "__dataclass_fields__") and isinstance(
        any_model_instance.__dataclass_fields__, tuple
    )

    assert dataclasses.is_dataclass(any_model_instance)

    field_names = [field.name for field in any_model_instance.__dataclass_fields__]
    type_hints = typing.get_type_hints(any_model_instance.__class__)
    for name, type_hint in type_hints.items():
        if typing.get_origin(type_hint) is ClassVar:
            assert hasattr(any_model_instance, name)
            assert hasattr(any_model_instance.__class__, name)
            assert name not in field_names


def test_non_instantiable(non_instantiable_model_factory):
    with pytest.raises(TypeError, match="Trying to instantiate"):
        non_instantiable_model_factory()


def test_field_redefinition(basic_fields_model_factory):
    base_model = basic_fields_model_factory()
    field_values = {
        field.name: getattr(base_model, field.name)
        for field in BasicFieldsModel.__dataclass_fields__
    }

    # Redefinition with same type
    class SubModel(BasicFieldsModel):
        int_value: int

    SubModel(**field_values)

    # Redefinition with different type
    class SubModel(BasicFieldsModel):
        int_value: float

    with pytest.raises(TypeError, match="int_value"):
        SubModel(**field_values)

    new_field_values = {**field_values}
    new_field_values.pop("int_value")
    SubModel(**new_field_values, int_value=1.0)


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
    assert model.any_value is None


class TestTypeValidation:
    @pytest.mark.parametrize(
        "basic_fields_model_factory", [BasicFieldsModelFactory, FixedBasicFieldsModelFactory]
    )
    def test_basic_type_validation(self, basic_fields_model_factory):
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

    @pytest.mark.parametrize(
        "sample_model_factory", [AdvancedFieldsModelFactory, OtherAdvancedFieldsModelFactory]
    )
    def test_advanced_type_validation(self, sample_model_factory):
        sample_model_factory()

    def test_invalid_type_validation(self, advanced_fields_model_factory):
        advanced_fields_model_factory()

        with pytest.raises(TypeError, match="str_list"):
            advanced_fields_model_factory(str_list=("a", "b"))
        with pytest.raises(TypeError, match="str_list"):
            advanced_fields_model_factory(str_list=["a", 2])

        with pytest.raises(TypeError, match="int_set"):
            advanced_fields_model_factory(int_set={"a", "b"})
        with pytest.raises(TypeError, match="int_set"):
            advanced_fields_model_factory(int_set=[1, "2"])

        with pytest.raises(TypeError, match="float_sequence"):
            advanced_fields_model_factory(float_sequence={1.1, 2.2})
        with pytest.raises(TypeError, match="float_sequence"):
            advanced_fields_model_factory(float_sequence=[1.1, 2])

        with pytest.raises(TypeError, match="int_float_dict"):
            advanced_fields_model_factory(int_float_dict=types.MappingProxyType({1: 2.2}))
        with pytest.raises(TypeError, match="int_float_dict"):
            advanced_fields_model_factory(int_float_dict={1.1: 2.2})

        with pytest.raises(TypeError, match="str_float_map"):
            advanced_fields_model_factory(str_float_map={"one": "2.2"})

        with pytest.raises(TypeError, match="int_float_tuple"):
            advanced_fields_model_factory(int_float_tuple=[1, 2.2])
        with pytest.raises(TypeError, match="int_float_tuple"):
            advanced_fields_model_factory(int_float_tuple=(1, 2.2, 3))
        with pytest.raises(TypeError, match="int_float_tuple"):
            advanced_fields_model_factory(int_float_tuple=(1, "2.2"))

        with pytest.raises(TypeError, match="int_tuple"):
            advanced_fields_model_factory(int_tuple=[1, 2])
        with pytest.raises(TypeError, match="int_tuple"):
            advanced_fields_model_factory(int_tuple=(1.1))

        with pytest.raises(TypeError, match="int_float_str_union"):
            advanced_fields_model_factory(int_float_str_union=(1, 2))
        with pytest.raises(TypeError, match="int_float_str_union"):
            advanced_fields_model_factory(int_float_str_union=None)

        with pytest.raises(TypeError, match="opt_float"):
            advanced_fields_model_factory(opt_float=1)
        with pytest.raises(TypeError, match="opt_float"):
            advanced_fields_model_factory(opt_float="1.1")

        with pytest.raises(TypeError, match="opt_int_kind"):
            advanced_fields_model_factory(opt_int_kind=Kind.FOO)
        with pytest.raises(TypeError, match="opt_int_kind"):
            advanced_fields_model_factory(opt_int_kind=1000)

        with pytest.raises(TypeError, match="opt_int_str_union"):
            advanced_fields_model_factory(opt_int_str_union=1.1)
        with pytest.raises(TypeError, match="opt_int_str_union"):
            advanced_fields_model_factory(opt_int_str_union=(1,))

        with pytest.raises(TypeError, match="tuple_with_opt_union"):
            advanced_fields_model_factory(tuple_with_opt_union=(1,))
        with pytest.raises(TypeError, match="tuple_with_opt_union"):
            advanced_fields_model_factory(tuple_with_opt_union=(1, 1.1))

        advanced_fields_model_factory(five_literal=2 + 3)
        with pytest.raises(ValueError, match="five_literal"):
            advanced_fields_model_factory(five_literal=(5,))
        with pytest.raises(ValueError, match="five_literal"):
            advanced_fields_model_factory(five_literal="5")

        advanced_fields_model_factory(true_literal=1 == 1)
        with pytest.raises(ValueError, match="true_literal"):
            advanced_fields_model_factory(true_literal=1)
        with pytest.raises(ValueError, match="true_literal"):
            advanced_fields_model_factory(true_literal="True")

        with pytest.raises(TypeError, match="nested_dict"):
            advanced_fields_model_factory(nested_dict=types.MappingProxyType({0: []}))
        with pytest.raises(TypeError, match="nested_dict"):
            advanced_fields_model_factory(nested_dict={None: None})

    @pytest.mark.parametrize("model_factory", [CompositeModelFactory, FixedCompositeModelFactory])
    def test_composite_type_validation(self, model_factory):
        composite_model = model_factory()

        with pytest.raises(TypeError, match="basic_model"):
            model_factory(basic_model="WRONG TYPE")
        with pytest.raises(TypeError, match="basic_model_with_defaults"):
            model_factory(basic_model_with_defaults="WRONG TYPE")

        # Test that equivalent (but different) classes are not accepted
        exec_results = {}
        exec(inspect.getsource(BasicFieldsModel), globals(), exec_results)
        AltBasicFieldsModel = exec_results["BasicFieldsModel"]
        assert (
            AltBasicFieldsModel is not BasicFieldsModel and AltBasicFieldsModel != BasicFieldsModel
        )

        field_values = {
            field.name: getattr(composite_model.basic_model, field.name)
            for field in BasicFieldsModel.__dataclass_fields__
        }
        another_basic_model = BasicFieldsModel(**field_values)
        model_factory(basic_model=another_basic_model)

        alt_basic_model = AltBasicFieldsModel(**field_values)
        with pytest.raises(TypeError, match="basic_model"):
            model_factory(basic_model=alt_basic_model)


class TestFieldValidators:
    def test_field_validators(self, model_with_validators_factory):
        model_with_validators_factory(extra_value="")
        model_with_validators_factory(extra_value=0)
        model_with_validators_factory(extra_value=0.0)
        model_with_validators_factory(extra_value=[])
        model_with_validators_factory(extra_value={})

        with pytest.raises(ValueError, match="extra_value"):
            model_with_validators_factory(extra_value=1)

        with pytest.raises(ValueError, match="float_value"):
            model_with_validators_factory(float_value=100.0)

        with pytest.raises(ValueError, match="str_value"):
            model_with_validators_factory(float_value=1.0, str_value="1.0")

    def test_inherited_field_validators(self, inherited_model_with_validators_factory):
        inherited_model_with_validators_factory()

        # Verify that validator from superclasss works just fine
        with pytest.raises(ValueError, match="even_int_value"):
            inherited_model_with_validators_factory(even_int_value=1)

        # Verify that validator works even if the validator function name is
        # reused in a subclass
        with pytest.raises(ValueError, match="int_value"):
            inherited_model_with_validators_factory(int_value=-1)

        # Overwritten field definition does not reuse validator from superclass
        inherited_model_with_validators_factory(float_value=100.0)
        inherited_model_with_validators_factory(float_value=1.0, str_value="1.0")

        # Overwritten field definition uses new validator from subclass
        with pytest.raises(ValueError, match="str_value"):
            inherited_model_with_validators_factory(str_value="this is too long")

        # Overwritten field definition with new type uses validators from new definition
        inherited_model_with_validators_factory(extra_value=False)
        with pytest.raises(TypeError, match="extra_value"):
            inherited_model_with_validators_factory(extra_value=0)
        with pytest.raises(ValueError, match="extra_value"):
            inherited_model_with_validators_factory(extra_value=True)

        # Regular behavior for new field
        inherited_model_with_validators_factory(int_value=1, new_int_value=1)
        with pytest.raises(TypeError, match="new_int_value"):
            inherited_model_with_validators_factory(int_value=1, new_int_value=1.0)
        with pytest.raises(ValueError, match="new_int_value"):
            inherited_model_with_validators_factory(int_value=1, new_int_value=2)


class TestRootValidators:
    def test_root_validators(self, model_with_root_validators_factory):
        model_with_root_validators_factory()

        model_with_root_validators_factory(int_value=1, str_value="1")
        with pytest.raises(ValueError, match="float_value"):
            model_with_root_validators_factory(int_value=1, float_value=1.0)

    def test_inherited_root_validators(self, inherited_model_with_root_validators_factory):
        inherited_model_with_root_validators_factory()

        with pytest.raises(ValueError, match="str_value"):
            inherited_model_with_root_validators_factory(int_value=1, str_value="1")
        with pytest.raises(ValueError, match="float_value"):
            inherited_model_with_root_validators_factory(int_value=1, float_value=1.0)
            inherited_model_with_root_validators_factory(
                int_value=1, float_value=1.0, str_value="1"
            )


class TestGenericModels:
    @pytest.mark.parametrize("concrete_type", [int, float, str])
    def test_generic_model_instantiation(self, concrete_type):
        Model = SimpleGenericModel[concrete_type]
        assert concrete_type.__name__ in Model.__name__

        Model1 = SimpleGenericModel[concrete_type]
        Model2 = SimpleGenericModel[concrete_type]
        Model3 = SimpleGenericModel[concrete_type]

        assert (
            Model is Model1
            and Model1 is Model2
            and Model2 is Model3
            and Model3 is SimpleGenericModel[concrete_type]
        )

    @pytest.mark.parametrize(
        "value", [False, 1, 1.1, "string", [1], ("a", "b"), {1, 2, 3}, {"a": 1}]
    )
    def test_generic_field_type_validation(self, simple_generic_model_factory, value):
        simple_generic_model_factory(generic_value="")

    @pytest.mark.parametrize(
        "value", [False, 1, 1.1, "string", [1], ("a", "b"), {1, 2, 3}, {"a": 1}]
    )
    @pytest.mark.parametrize("concrete_type", [int, float, str])
    def test_concrete_field_type_validation(self, concrete_type, value):
        Model = SimpleGenericModel[concrete_type]

        if isinstance(value, concrete_type):  # concrete_type == type(value):
            model = Model(generic_value=value)
            assert model.int_value == 0
        else:
            with pytest.raises(TypeError, match="generic_value"):
                model = Model(generic_value=value)


class TestFieldFunctions:
    def test_missing_annotations(self):
        with pytest.raises(TypeError, match="other_value"):

            class Model(datamodels.DataModel):
                other_value = datamodels.field(default=None)

    def test_field_defaults(self):
        class Model(datamodels.DataModel):
            str_value: str = datamodels.field(default="DEFAULT")

        assert Model().str_value == "DEFAULT"
        assert Model(str_value="other").str_value == "other"

        with pytest.raises(TypeError, match="int_value"):

            class Model(datamodels.DataModel):
                int_value: int = datamodels.field(default="DEFAULT")

            Model()

    def test_field_default_factory(self):
        # Classic type factory
        class Model(datamodels.DataModel):
            list_value: List[int] = datamodels.field(default_factory=list)

        list_value = Model().list_value
        assert isinstance(list_value, list) and len(list_value) == 0

        # Custom function factory
        def list_factory():
            return list(i for i in range(5))

        class Model(datamodels.DataModel):
            list_value: List[int] = datamodels.field(default_factory=list_factory)

        list_value = Model().list_value
        assert (
            isinstance(list_value, list) and len(list_value) == 5 and list_value == list_factory()
        )

        # Invalid default and default_factory combination
        with pytest.raises(ValueError, match="both default and default_factory"):

            class Model(datamodels.DataModel):
                list_value: Optional[List[int]] = datamodels.field(
                    default=None, default_factory=list
                )

    @pytest.mark.parametrize("value", [5, "5", 1.1, "22"])
    def test_field_converter(self, value):
        class Model(datamodels.DataModel):
            int_value: int = datamodels.field(converter=int)

        assert Model(int_value=value).int_value == int(value)

        with pytest.raises(ValueError, match="int()"):
            assert Model(int_value="invalid")

    def test_invalid_field_converter(self):
        class OtherModel(datamodels.DataModel):
            int_value: int = datamodels.field(converter=str)

        with pytest.raises(TypeError, match="int_value"):
            OtherModel(int_value=3)

        with pytest.raises(TypeError, match="int_value"):
            OtherModel(int_value="3")

    @pytest.mark.parametrize("value", [1, 2.2, "3", "asdf"])
    def test_auto_field_converter(self, value):
        class Model(datamodels.DataModel):
            int_value: int = datamodels.field(converter=True)

        try:
            expected_value = int(value)
        except ValueError:
            with pytest.raises(ValueError):
                Model(int_value=value).int_value
        else:
            int_value = Model(int_value=value).int_value
            assert isinstance(int_value, int)
            assert int_value == expected_value
