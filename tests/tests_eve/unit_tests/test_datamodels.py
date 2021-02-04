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
import types
import typing
from typing import Set  # noqa: F401  # imported but unused (used in exec() context)
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import devtools
import factory
import pytest
import pytest_factoryboy as pytfboy

from eve import datamodels, utils


T = TypeVar("T")


# --- Utils and sample data ---
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


class SampleClass:
    pass


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


@pytest.fixture(params=example_model_factories)
def example_model_factory(request) -> datamodels.DataModelLike:  # type: ignore
    return request.param


# --- Tests ---
# Test generated members
def test_datamodel_class_members(example_model_factory):
    model = example_model_factory()
    model_class = model.__class__

    assert hasattr(model_class, "__init__")
    assert hasattr(model_class, "__datamodel_fields__")
    assert isinstance(model_class.__datamodel_fields__, utils.FrozenNamespace)
    assert hasattr(model_class, "__datamodel_params__")
    assert isinstance(model_class.__datamodel_params__, utils.FrozenNamespace)
    assert hasattr(model_class, "__datamodel_validators__")
    assert isinstance(model_class.__datamodel_validators__, tuple)


def test_devtools_compatibility(example_model_factory):
    model = example_model_factory()
    model_class = model.__class__
    formatted_string = devtools.pformat(model)

    assert hasattr(model_class, "__pretty__")
    assert callable(model_class.__pretty__)
    assert f"{model_class.__name__}(" in formatted_string
    for name in model_class.__datamodel_fields__.keys():
        assert f"{name}=" in formatted_string


def test_dataclass_compatibility(example_model_factory):
    model = example_model_factory()
    model_class = model.__class__

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

    @datamodels.datamodel
    class WrongModel:
        bool_value: bool = 1

    with pytest.raises(TypeError, match="'bool_value'"):
        WrongModel()


def test_default_factories():
    @datamodels.datamodel
    class Model:
        list_value: List[int] = datamodels.field(default_factory=list)
        dict_value: Dict[str, int] = datamodels.field(default_factory=lambda: {"one": 1})

    model = Model()

    assert model.list_value == []
    assert model.dict_value == {"one": 1}

    @datamodels.datamodel
    class WrongModel:
        list_value: List[int] = datamodels.field(default_factory=tuple)

    with pytest.raises(TypeError, match="'list_value'"):
        WrongModel()


# Test field specification
sample_type_data = [
    ("bool", [True, False], [1, "True"]),
    ("int", [1, -1], [1.0, True, "1"]),
    ("float", [1.0], [1, "1.0"]),
    ("str", ["", "one"], [1, ("one",)]),
    ("complex", [1j], [1, 1.0, "1j"]),
    ("bytes", [b"bytes", b""], ["string", ["a"]]),
    ("typing.Any", ["any"], tuple()),
    ("typing.Literal[1, 1.0, True]", [1, 1.0, True], [False]),
    ("typing.Tuple[int, str]", [(3, "three")], [(), (3, 3)]),
    ("typing.Tuple[int, ...]", [(1, 2, 3), ()], [3, (3, "three")]),
    ("typing.List[int]", ([1, 2, 3], []), (1, [1.0])),
    ("typing.Set[int]", ({1, 2, 3}, set()), (1, [1], (1,), {1: None})),
    ("typing.Dict[int, str]", ({}, {3: "three"}), ([(3, "three")], 3, "three", [])),
    ("typing.Sequence[int]", ([1, 2, 3], [], (1, 2, 3), tuple()), (1, [1.0], {1})),
    ("typing.MutableSequence[int]", ([1, 2, 3], []), ((1, 2, 3), tuple(), 1, [1.0], {1})),
    ("typing.Set[int]", ({1, 2, 3}, set()), (1, [1], (1,), {1: None})),
    ("typing.Union[int, float, str]", [1, 3.0, "one"], [[1], [], 1j]),
    ("typing.Optional[int]", [1, None], [[1], [], 1j]),
    (
        "typing.Dict[Union[int, float, str], Union[Tuple[int, Optional[float]], Set[int]]]",
        [{1: (2, 3.0)}, {1.0: (2, None)}, {"1": {1, 2}}],
        [{(1, 1.0, "1"): set()}, {1: [1]}, {"1": (1,)}],
    ),
]


@pytest.mark.parametrize(["type_hint", "valid_values", "wrong_values"], sample_type_data)
def test_field_type_hint(type_hint: Type, valid_values: Sequence[Any], wrong_values: Sequence[Any]):
    context = {}
    exec(
        f"""
@datamodels.datamodel
class Model:
    value: {type_hint}
""",
        globals(),
        context,
    )
    Model = context["Model"]

    assert eval(Model.__datamodel_fields__.value.type) == eval(type_hint)

    for value in valid_values:
        Model(value=value)

    for value in wrong_values:
        with pytest.raises((TypeError, ValueError), match="'value'"):
            Model(value=value)


def test_custom_class_type_hint():
    @datamodels.datamodel
    class Model1:
        value: SampleEnum

    Model1(value=SampleEnum.FOO)

    class SampleClassChild(SampleClass):
        pass

    class SampleClassGrandChild(SampleClassChild):
        pass

    @datamodels.datamodel
    class Model2:
        value: SampleClass

    Model2(value=SampleClass())
    Model2(value=SampleClassChild())
    Model2(value=SampleClassGrandChild())

    class AnotherClass:
        pass

    with pytest.raises(TypeError, match="value"):
        Model1(value=AnotherClass())
    with pytest.raises(TypeError, match="value"):
        Model2(value=AnotherClass())


def test_field_redefinition():
    class Model(datamodels.DataModel):
        value: int = 0

    assert Model().value == 0

    # Redefinition with same type
    class ChildModel1(Model):
        value: int = 1

    assert ChildModel1().value == 1

    # Redefinition with different type
    class ChildModel2(Model):
        value: float = 2.2

    assert ChildModel2().value == 2.2

    with pytest.raises(TypeError, match="value"):
        assert ChildModel2(value=2)


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


def test_missing_annotations():
    with pytest.raises(TypeError, match="other_value"):

        class Model(datamodels.DataModel):
            other_value = datamodels.field(default=None)


# Test custom validators
class ModelWithValidators(datamodels.DataModel):
    bool_value: bool = False
    int_value: int = 0
    even_int_value: int = 2
    float_value: float = 0.0
    str_value: str = ""
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
            raise ValueError(f"'{attribute.name}' must be lower or equal to 3.14159")

    @datamodels.validator("str_value")
    def _str_value_validator(self, attribute, value):
        # This kind of validation should arguably happen in a root_validator, but
        # since float_value is defined before str_value, it should have been
        # already validated at this point
        if value == str(self.float_value):
            raise ValueError(f"'{attribute.name}' must be different to 'float_value'")

    @datamodels.validator("extra_value")
    def _extra_value_validator(self, attribute, value):
        if bool(value):
            raise ValueError(f"'{attribute.name}' must be equivalent to False")


class ChildModelWithValidators(ModelWithValidators):
    pass


@pytest.mark.parametrize("model_class", [ModelWithValidators, ChildModelWithValidators])
def test_field_validators(model_class: datamodels.DataModelLike):

    with pytest.raises(ValueError, match="int_value"):
        model_class(int_value=-1)

    model_class(even_int_value=-2)
    with pytest.raises(ValueError, match="even_int_value"):
        model_class(even_int_value=1)

    model_class(float_value=-2.0)
    with pytest.raises(ValueError, match="float_value"):
        model_class(float_value=4.0)

    with pytest.raises(ValueError, match="str_value"):
        model_class(float_value=1.0, str_value="1.0")

    with pytest.raises(ValueError, match="extra_value"):
        model_class(extra_value=1)

    model_class(extra_value=False)
    model_class(extra_value=0)
    model_class(extra_value="")
    model_class(extra_value=[])
    with pytest.raises(ValueError, match="extra_value"):
        model_class(extra_value=1)


def test_new_field_validators_in_subclass():
    class ChildModelWithValidators(ModelWithValidators):
        @datamodels.validator("int_value")
        def _int_value_validator(self, attribute, value):
            if value > 10:
                raise ValueError(f"'{attribute.name}' must be lower or equal to 10")

    ChildModelWithValidators(int_value=1)
    with pytest.raises(ValueError, match="int_value"):
        ChildModelWithValidators(int_value=-1)
    with pytest.raises(ValueError, match="int_value"):
        ChildModelWithValidators(int_value=11)


def test_field_validators_in_overwritten_field_in_subclass():
    class ChildModelWithValidators(ModelWithValidators):
        int_value: int = 0

        @datamodels.validator("int_value")
        def _int_value_validator(self, attribute, value):
            if value > 10:
                raise ValueError(f"'{attribute.name}' must be lower or equal to 10")

    ChildModelWithValidators(int_value=1)
    ChildModelWithValidators(int_value=-1)  # Test new field definition
    with pytest.raises(ValueError, match="int_value"):
        ChildModelWithValidators(int_value=11)

    # Overwrite field definition with a new type
    class AnotherChildModelWithValidators(ModelWithValidators):
        extra_value: float

        @datamodels.validator("extra_value")
        def _extra_value_validator(self, attribute, value):
            if value < 0.0:
                raise ValueError(f"'{attribute.name}' must be a positive number")

    AnotherChildModelWithValidators(extra_value=0.0)
    AnotherChildModelWithValidators(extra_value=1.0)
    with pytest.raises(ValueError, match="extra_value"):
        AnotherChildModelWithValidators(extra_value=-1.0)


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


class ChildModelWithRootValidators(ModelWithRootValidators):
    pass


@pytest.mark.parametrize("model_class", [ModelWithRootValidators, ChildModelWithRootValidators])
def test_root_validators(model_class: datamodels.DataModelLike):
    model_class(int_value=0, float_value=1.1, str_value="")
    with pytest.raises(ValueError, match="float_value"):
        model_class(int_value=1, float_value=1.0, str_value="")


def test_root_validators_in_subclasses():
    class Model(ModelWithRootValidators):
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

    with pytest.raises(ValueError, match="str_value"):
        Model(int_value=1, float_value=0.0, str_value="1")
    with pytest.raises(ValueError, match="float_value"):
        Model(int_value=1, float_value=1.0, str_value="1")
    with pytest.raises(ValueError, match="float_value"):
        Model(int_value=1, float_value=1.0, str_value="1")


# Test field options
def test_field_init():
    @datamodels.datamodel
    class Model:
        hidden_value: int = datamodels.field(init=False)
        value: int
        hidden_value_with_default: int = datamodels.field(default=10, init=False)

    model = Model(value=1)
    assert model.value == 1
    assert model.hidden_value_with_default == 10

    model.hidden_value = -1
    assert model.hidden_value == -1


def test_field_metadata():
    @datamodels.datamodel
    class Model:
        value: int = datamodels.field(metadata={"my_metadata": "META"})

    assert isinstance(Model.__datamodel_fields__.value.metadata, types.MappingProxyType)
    assert Model.__datamodel_fields__.value.metadata["my_metadata"] == "META"


# Test datamodel options
def test_frozen():
    import attr

    @datamodels.datamodel(frozen=True)
    class FrozenModel:
        value: Any = None

    assert FrozenModel.__datamodel_params__.frozen is True
    with pytest.raises(attr.exceptions.FrozenInstanceError):
        FrozenModel().value = 1

    class FrozenModel2(datamodels.DataModel, frozen=True):
        value: Any = None

    assert FrozenModel2.__datamodel_params__.frozen is True
    with pytest.raises(attr.exceptions.FrozenInstanceError):
        FrozenModel2().value = 1


def test_hash():
    string_value = "this is a really long string to avoid string interning 1234567890 +:?.!"

    # Safe mutable case: no hash
    @datamodels.datamodel
    class MutableModel:
        value: Any = None

    with pytest.raises(TypeError, match="unhashable type"):
        hash(MutableModel(value=string_value))

    # Unsafe mutable case: mutable hash
    @datamodels.datamodel(unsafe_hash=True)
    class MutableModelWithHash:
        value: Any = None

    mutable_model = MutableModelWithHash(value=string_value)
    hash_value = hash(mutable_model)
    assert hash_value == hash(MutableModelWithHash(value=string_value))

    # This is the (expected) wrong behaviour with a 'unsafe_hash=True',
    # because hash value should not change during lifetime of an object
    mutable_model.value = 42
    assert hash_value != hash(mutable_model)

    # Safe frozen case: proper hash
    @datamodels.datamodel(frozen=True)
    class FrozenModel:
        value: Any = None

    assert hash(MutableModelWithHash(value=string_value)) == hash(
        MutableModelWithHash(value=string_value)
    )


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


# Test generic models
@pytest.mark.parametrize(
    "concrete_type",
    [int, List[float], Tuple[int, ...], Optional[int], Union[int, float]],
)
def test_generic_model_instantiation_name(concrete_type: Type):
    Model = datamodels.concretize(GenericModel, concrete_type)
    assert Model.__name__.startswith(GenericModel.__name__)
    assert Model.__name__ != GenericModel.__name__

    Model = datamodels.concretize(GenericModel, concrete_type, class_name="MyNewConcreteClass")
    assert Model.__name__ == "MyNewConcreteClass"


@pytest.mark.parametrize(
    "concrete_type",
    [int, List[float], Tuple[int, ...], Optional[int], Union[int, float]],
)
def test_generic_model_alias(concrete_type: Type):
    Model = datamodels.concretize(GenericModel, concrete_type)

    assert GenericModel[concrete_type].__class__ is Model
    assert typing.get_origin(GenericModel[concrete_type]) is Model

    class SubModel(GenericModel[concrete_type]):
        ...

    assert SubModel.__base__ is Model


@pytest.mark.parametrize(
    "concrete_type",
    [int, List[float], Tuple[int, ...], Optional[int], Union[int, float]],
)
def test_generic_model_instantiation_cache(concrete_type):
    Model1 = datamodels.concretize(GenericModel, concrete_type)
    Model2 = datamodels.concretize(GenericModel, concrete_type)
    Model3 = datamodels.concretize(GenericModel, concrete_type)

    assert (
        Model1 is Model2
        and Model2 is Model3
        and Model3 is datamodels.concretize(GenericModel, concrete_type)
    )


def test_basic_generic_field_type_validation():
    class GenericModel(datamodels.DataModel, Generic[T]):
        value: T

    GenericModel(value=1)
    GenericModel(value="value")
    GenericModel(value=(1.0, "value"))
    GenericModel(value=None)

    class PartialGenericModel(datamodels.DataModel, Generic[T]):
        value: List[T]

    PartialGenericModel(value=[])
    PartialGenericModel(value=[1])
    PartialGenericModel(value=["value"])
    PartialGenericModel(value=[1.0, "value"])
    PartialGenericModel(value=[(1.0, "value")])
    PartialGenericModel(value=[None])
    with pytest.raises(TypeError, match="'value'"):
        PartialGenericModel(value=1)
    with pytest.raises(TypeError, match="'value'"):
        PartialGenericModel(value=(1, 2))


@pytest.mark.parametrize(["type_hint", "valid_values", "wrong_values"], sample_type_data)
def test_concrete_field_type_validation(
    type_hint: Type, valid_values: Sequence[Any], wrong_values: Sequence[Any]
):
    Model = GenericModel[eval(type_hint)].__class__

    for value in valid_values:
        Model(value=value)

    for value in wrong_values:
        with pytest.raises((TypeError, ValueError), match="'value'"):
            Model(value=value)
