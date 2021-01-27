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
# version. See the LICENSE.txt file at the top-l directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Data Model class and related utils."""


from __future__ import annotations

import collections
import dataclasses
import sys
import types
import typing
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import attr

from eve.typingx import NonDataDescriptorProto

from . import utils
from .concepts import NOTHING


# Typing
T = TypeVar("T")
V = TypeVar("V")


class _AttrClassProto(Protocol):
    __attrs_attrs__: ClassVar[Tuple[attr.Attribute, ...]]


class _DataClassProto(Protocol):
    __dataclass_fields__: ClassVar[Tuple[dataclasses.Field, ...]]


class BaseDataModelProto(_AttrClassProto, _DataClassProto, Protocol):
    __datamodel_fields__: ClassVar[Tuple[dataclasses.Field, ...]]
    __datamodel_params__: ClassVar[utils.FrozenNamespace]
    __datamodel_validators__: ClassVar[
        Tuple[NonDataDescriptorProto[BaseDataModelProto, RootValidatorFunc], ...]
    ]

    def __post_init__(self: BaseDataModelProto) -> None:
        ...


class GenericDataModelProto(BaseDataModelProto):
    __args__: ClassVar[Tuple[Union[Type, TypeVar]]]
    __parameters__: ClassVar[Tuple[TypeVar]]


DataModelProto = Union[BaseDataModelProto, GenericDataModelProto]

ValidatorFunc = Callable[[DataModelProto, attr.Attribute, Any], None]
RootValidatorFunc = Callable[[Type[DataModelProto], DataModelProto], None]


# Implementation
_ATTR_SETTINGS = types.MappingProxyType({"auto_attribs": True, "slots": False, "kw_only": True})
_FIELD_VALIDATOR_TAG = "_FIELD_VALIDATOR_TAG"
_MODEL_FIELDS = "__datamodel_fields__"
_MODEL_PARAMS = "__datamodel_params__"
_ROOT_VALIDATOR_TAG = "__ROOT_VALIDATOR_TAG"
_ROOT_VALIDATORS = "__datamodel_validators__"


class _SENTINEL:
    ...


AUTO_CONVERTER = _SENTINEL()


# -- Validators --
class _TupleValidator:
    """Implementation of attr.s type validator for Tuple typings."""

    validators: Tuple[attr._ValidatorType, ...]
    tuple_type: Type[Tuple]

    def __init__(
        self, validators: Tuple[attr._ValidatorType, ...], tuple_type: Type[Tuple]
    ) -> None:
        self.validators = validators
        self.tuple_type = tuple_type

    def __call__(self, instance: _AttrClassProto, attribute: attr.Attribute, value: Any) -> None:
        if not isinstance(value, self.tuple_type):
            raise TypeError(
                f"In '{attribute.name}' validation, got '{value}' that is a {type(value)} instead of {self.tuple_type}."
            )
        if len(value) != len(self.validators):
            raise TypeError(
                f"In '{attribute.name}' validation, got '{value}' tuple which contains {len(value)} elements instead of {len(self.validators)}."
            )

        _i = None
        item_value = ""
        try:
            for _i, (item_value, item_validator) in enumerate(zip(value, self.validators)):
                item_validator(instance, attribute, item_value)
        except Exception as e:
            raise TypeError(
                f"In '{attribute.name}' validation, tuple '{value}' contains invalid value '{item_value}' at position {_i}."
            ) from e


class _OrValidator:
    """Implementation of attr.s validator composing multiple validators together using OR."""

    validators: Tuple[attr._ValidatorType, ...]
    error_type: Type[Exception]

    def __init__(
        self, validators: Tuple[attr._ValidatorType, ...], error_type: Type[Exception]
    ) -> None:
        self.validators = validators
        self.error_type = error_type

    def __call__(self, instance: _AttrClassProto, attribute: attr.Attribute, value: Any) -> None:
        passed = False
        for v in self.validators:
            try:
                v(instance, attribute, value)
                passed = True
                break
            except Exception:
                pass

        if not passed:
            raise self.error_type(
                f"In '{attribute.name}' validation, provided value '{value}' fails for all the possible validators."
            )


class _LiteralValidator:
    """Implementation of attr.s type validator for Literal typings."""

    literal: Any

    def __init__(self, literal: Any) -> None:
        self.literal = literal

    def __call__(self, instance: _AttrClassProto, attribute: attr.Attribute, value: Any) -> None:
        if isinstance(self.literal, bool):
            valid = value is self.literal
        else:
            valid = value == self.literal
        if not valid:
            raise ValueError(
                f"Provided value '{value}' field does not match {self.literal} during '{attribute.name}' validation."
            )


def empty_attrs_validator() -> attr._ValidatorType:
    """Create an attr.s empty validator which always succeeds."""

    def _empty_validator(instance: _AttrClassProto, attribute: attr.Attribute, value: Any) -> None:
        pass

    return _empty_validator


def or_attrs_validator(
    *validators: attr._ValidatorType, error_type: Type[Exception]
) -> attr._ValidatorType:
    """Create an attr.s validator combinator where only one of the validators needs to pass."""
    # vals: Tuple[_AttrsValidatorFunc] = tuple(utils.flatten(validators))
    if len(validators) == 1:
        return validators[0]
    else:
        return _OrValidator(validators, error_type=error_type)


def literal_type_attrs_validator(*type_args: Type) -> attr._ValidatorType:
    """Create an attr.s strict type validator for Literal typings."""
    return or_attrs_validator(*(_LiteralValidator(t) for t in type_args), error_type=ValueError)


def tuple_type_attrs_validator(*type_args: Type, tuple_type: Type = tuple) -> attr._ValidatorType:
    """Create an attr.s strict type validator for Tuple typings."""
    if len(type_args) == 2 and (type_args[1] is Ellipsis):
        member_type_hint = type_args[0]
        return attr.validators.deep_iterable(
            member_validator=strict_type_attrs_validator(member_type_hint),
            iterable_validator=attr.validators.instance_of(tuple_type),
        )
    else:
        return _TupleValidator(tuple(strict_type_attrs_validator(t) for t in type_args), tuple_type)


def union_type_attrs_validator(*type_args: Type) -> attr._ValidatorType:
    """Create an attr.s strict type validator for Union typings."""
    if len(type_args) == 2 and (type_args[1] is type(None)):  # noqa: E721  # use isinstance()
        non_optional_validator = strict_type_attrs_validator(type_args[0])
        return attr.validators.optional(non_optional_validator)
    else:
        return or_attrs_validator(
            *(strict_type_attrs_validator(t) for t in type_args), error_type=TypeError
        )


def strict_type_attrs_validator(type_hint: Type) -> attr._ValidatorType:
    """Create an attr.s strict type validator for a specific typing hint."""
    origin_type = typing.get_origin(type_hint)
    type_args = typing.get_args(type_hint)

    if isinstance(type_hint, type):
        assert not type_args
        return attr.validators.instance_of(type_hint)
    elif isinstance(type_hint, typing.TypeVar):
        if type_hint.__bound__:
            return attr.validators.instance_of(type_hint.__bound__)
        else:
            return empty_attrs_validator()
    elif type_hint is Any:
        return empty_attrs_validator()
    elif origin_type is typing.Literal:
        return literal_type_attrs_validator(*type_args)
    elif origin_type is typing.Union:
        return union_type_attrs_validator(*type_args)
    elif isinstance(origin_type, type):
        # Deal with generic collections
        if issubclass(origin_type, tuple):
            return tuple_type_attrs_validator(*type_args, tuple_type=origin_type)
        elif issubclass(origin_type, (collections.abc.Sequence, collections.abc.Set)):
            assert len(type_args) == 1
            member_type_hint = type_args[0]
            return attr.validators.deep_iterable(
                member_validator=strict_type_attrs_validator(member_type_hint),
                iterable_validator=attr.validators.instance_of(origin_type),
            )
        elif issubclass(origin_type, collections.abc.Mapping):
            assert len(type_args) == 2
            key_type_hint, value_type_hint = type_args
            return attr.validators.deep_mapping(
                key_validator=strict_type_attrs_validator(key_type_hint),
                value_validator=strict_type_attrs_validator(value_type_hint),
                mapping_validator=attr.validators.instance_of(origin_type),
            )
    else:
        raise TypeError(f"Type description '{type_hint}' is not supported.")

    assert False  # noqa


# -- DataModel --
def _collect_field_validators(cls: Type, *, delete_tag: bool = True) -> Dict[str, ValidatorFunc]:
    result = {}
    for _, member in cls.__dict__.items():
        if hasattr(member, _FIELD_VALIDATOR_TAG):
            field_name = getattr(member, _FIELD_VALIDATOR_TAG)
            result[field_name] = member
            if delete_tag:
                delattr(member, _FIELD_VALIDATOR_TAG)

    return result


def _collect_root_validators(cls: Type, *, delete_tag: bool = True) -> List[RootValidatorFunc]:
    result = []
    for base in reversed(cls.__mro__[1:]):
        for validator in getattr(base, _ROOT_VALIDATORS, []):
            if validator not in result:
                result.append(validator)

    for _, member in cls.__dict__.items():
        if hasattr(member, _ROOT_VALIDATOR_TAG):
            result.append(member)
            if delete_tag:
                delattr(member, _ROOT_VALIDATOR_TAG)

    return result


def _get_attribute_from_bases(
    name: str, mro: Tuple[Type, ...], annotations: Optional[Dict[str, Any]] = None
) -> Optional[attr.Attribute]:
    for base in mro:
        for base_field_attrib in getattr(base, "__attrs_attrs__", []):
            if base_field_attrib.name == name:
                if annotations is not None:
                    annotations[name] = base.__annotations__[name]
                return typing.cast(attr.Attribute, base_field_attrib)

    return None


def _substitute_typevars(
    type_hint: Type, type_params_map: Mapping[TypeVar, Union[Type, TypeVar]]
) -> Union[Type, TypeVar]:
    if isinstance(type_hint, typing.TypeVar):
        assert type_hint in type_params_map
        return type_params_map[type_hint]
    elif getattr(type_hint, "__parameters__", []):
        return type_hint[tuple(type_params_map[tp] for tp in type_hint.__parameters__)]
    else:
        return type_hint


def _make_counting_attr_from_attr(
    field_attr: attr.Attribute, *, include_type: bool = False, **kwargs: Any
) -> Any:  # attr.s lies on purpose in some typings
    members = [
        "default",
        "validator",
        "repr",
        "eq",
        "order",
        "hash",
        "init",
        "metadata",
        "converter",
        "kw_only",
        "on_setattr",
    ]
    if include_type:
        members.append("type")

    return attr.ib(**{key: getattr(field_attr, key) for key in members}, **kwargs)  # type: ignore  # too hard for mypy


def _make_dataclass_field_from_attr(field_attr: attr.Attribute) -> dataclasses.Field:
    MISSING = getattr(dataclasses, "MISSING", NOTHING)
    default = MISSING
    default_factory = MISSING
    if isinstance(field_attr.default, attr.Factory):  # type: ignore  # attr.s lies on purpose in some typings
        default_factory = field_attr.default.factory  # type: ignore  # attr.s lies on purpose in some typings
    elif field_attr.default is not attr.NOTHING:
        default = field_attr.default

    assert field_attr.eq == field_attr.order  # dataclasses.compare == (attr.eq and attr.order)

    dataclasses_field = dataclasses.Field(  # type: ignore  # dataclasses.Field signature seems invisible to mypy
        default=default,
        default_factory=default_factory,
        init=field_attr.init,
        repr=field_attr.repr if not callable(field_attr.repr) else None,
        hash=field_attr.hash,
        compare=field_attr.eq,
        metadata=field_attr.metadata,
    )
    dataclasses_field.name = field_attr.name
    assert field_attr.type is not None
    dataclasses_field.type = field_attr.type

    return dataclasses_field


def _make_non_instantiable_init() -> Callable[..., None]:
    def __init__(self: DataModelProto, *args: Any, **kwargs: Any) -> None:
        raise TypeError(f"Trying to instantiate '{type(self).__name__}' abstract class.")

    return __init__


def _make_post_init(has_post_init: bool) -> Callable[[DataModelProto], None]:
    if has_post_init:

        def __attrs_post_init__(self: DataModelProto) -> None:
            if attr._config._run_validators is True:  # type: ignore  # attr._config is not visible for mypy
                for validator in type(self).__datamodel_validators__:
                    validator.__get__(self)(self)
                self.__post_init__()

    else:

        def __attrs_post_init__(self: DataModelProto) -> None:
            if attr._config._run_validators is True:  # type: ignore  # attr._config is not visible for mypy
                for validator in type(self).__datamodel_validators__:
                    validator.__get__(self)(self)

    return __attrs_post_init__


def _make_data_model_class_getitem() -> classmethod:
    def __class_getitem__(
        cls: DataModelProto, args: Union[Type, Tuple[Type]]
    ) -> Type[DataModelProto]:
        type_args = args if isinstance(args, tuple) else (args,)
        return concretize(cls, *type_args, as_generic_alias=True)

    return classmethod(__class_getitem__)


def _make_datamodel(
    cls: Type,
    *,
    init: bool,  # noqa: A002   # shadowing 'init' python builtin
    repr: bool,  # noqa: A002   # shadowing 'repr' python builtin
    eq: bool,
    order: bool,
    unsafe_hash: bool,
    frozen: bool,
    instantiable: bool,
) -> Type:
    """Actual implementation of the Data Model creation.

    See :func:`datamodel` for the description of the parameters.
    """
    if "__annotations__" not in cls.__dict__:
        cls.__annotations__ = {}
    annotations = cls.__dict__["__annotations__"]
    mro_bases: Tuple[Type, ...] = cls.__mro__[1:]
    resolved_annotations = typing.get_type_hints(cls)

    # Create attrib definitions with automatic type validators (and converters)
    # for the annotated fields. The original annotations are used for iteration
    # since the resolved annotations may also contain superclasses' annotations
    for key in annotations:
        type_hint = resolved_annotations[key]
        if typing.get_origin(type_hint) is not ClassVar:
            type_validator = strict_type_attrs_validator(type_hint) if init else None
            if key not in cls.__dict__:
                setattr(cls, key, attr.ib(validator=type_validator))
            elif not isinstance(cls.__dict__[key], attr._make._CountingAttr):  # type: ignore  # attr._make is not visible for mypy
                setattr(cls, key, attr.ib(default=cls.__dict__[key], validator=type_validator))
            else:
                # A field() function has been used to customize the definition:
                # prepend the type validator to the list of provided validators (if any)
                cls.__dict__[key]._validator = (
                    type_validator
                    if cls.__dict__[key]._validator is None
                    else attr._make.and_(type_validator, cls.__dict__[key]._validator)  # type: ignore  # attr._make is not visible for mypy
                )

                if cls.__dict__[key].converter is AUTO_CONVERTER:
                    cls.__dict__[key].converter = _make_type_coercer(type_hint)

    # All fields should be annotated with type hints
    for key, value in cls.__dict__.items():
        if isinstance(value, attr._make._CountingAttr) and (  # type: ignore  # attr._make is not visible for mypy
            key not in annotations or typing.get_origin(resolved_annotations[key]) is ClassVar
        ):
            raise TypeError(f"Missing type annotation in '{key}' field.")

    root_validators = _collect_root_validators(cls)
    field_validators = _collect_field_validators(cls)

    # Add collected field validators
    for field_name, field_validator in field_validators.items():
        field_attrib = cls.__dict__.get(field_name, None)
        if not field_attrib:
            # Field has not been defined in the current class namespace,
            # look for field definition in the base classes.
            base_field_attr = _get_attribute_from_bases(field_name, mro_bases, annotations)
            if base_field_attr:
                # Create a new field in the current class cloning the existing
                # definition and add the new validator (attrs recommendation)
                field_attrib = _make_counting_attr_from_attr(
                    base_field_attr,
                )
                setattr(cls, field_name, field_attrib)
            else:
                raise TypeError(f"Validator assigned to non existing '{field_name}' field.")

        # Add field validator using field_attr.validator
        assert isinstance(field_attrib, attr._make._CountingAttr)  # type: ignore  # attr._make is not visible for mypy
        field_attrib.validator(field_validator)

    setattr(cls, _ROOT_VALIDATORS, tuple(root_validators))

    # Update class with attr.s features
    if init:
        if "__init__" in cls.__dict__:
            raise TypeError(
                "datamodel(init=True) is incompatible with custom '__init__' methods, use '__post_init__' instead."
            )

        has_post_init = "__post_init__" in cls.__dict__
        if not instantiable:
            cls.__init__ = _make_non_instantiable_init()
        else:
            # For dataclasses emulation, __attrs_post_init__ calls __post_init__ (if it exists)
            cls.__attrs_post_init__ = _make_post_init(has_post_init)

    cls.__class_getitem__ = _make_data_model_class_getitem()

    hash_arg = None if not unsafe_hash else True
    new_cls = attr.define(  # type: ignore  # attr.define is not visible for mypy
        **_ATTR_SETTINGS, init=init, repr=repr, eq=eq, order=order, frozen=frozen, hash=hash_arg
    )(cls)
    assert new_cls is cls

    # Final postprocessing
    setattr(
        cls,
        _MODEL_PARAMS,
        utils.FrozenNamespace(
            init=init,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=frozen,
            instantiable=instantiable,
        ),
    )
    setattr(
        cls,
        _MODEL_FIELDS,
        utils.FrozenNamespace(
            **{field_attr.name: field_attr for field_attr in cls.__attrs_attrs__}
        ),
    )
    cls.__dataclass_fields__ = tuple(  # dataclasses emulation
        _make_dataclass_field_from_attr(field_attr) for field_attr in cls.__attrs_attrs__
    )

    return cls


def is_datamodel(obj: Any) -> bool:
    """Returns True if `obj` is a Data Model class or an instance of a Data Model."""
    cls = obj if isinstance(obj, type) else obj.__class__
    return hasattr(cls, _MODEL_FIELDS)


def is_generic(datamodel: Union[DataModelProto, Type[DataModelProto]]) -> bool:
    """Returns True if `obj` is a generic Data Model class or an instance of a generic Data Model."""
    if not is_datamodel(datamodel):
        raise ValueError(f"Invalid datamodel instance or class: '{datamodel}'.")

    return len(getattr(datamodel, "__parameters__", [])) > 0


def is_instantiable(datamodel: Type[DataModelProto]) -> bool:
    """Returns True if `obj` is a generic Data Model class or an instance of a generic Data Model."""
    if not (isinstance(type) and is_datamodel(datamodel)):
        raise ValueError(f"Invalid datamodel class: '{datamodel}'.")

    params = getattr(datamodel, _MODEL_PARAMS)
    assert hasattr(params, "instantiable") and isinstance(params.instantiable, bool)
    return params.instantiable


def get_fields(
    datamodel: Union[DataModelProto, Type[DataModelProto]]
) -> Tuple[dataclasses.Field, ...]:
    """Return a tuple describing the fields of this Data Model."""
    if is_datamodel():
        return tuple(getattr(datamodel, _MODEL_FIELDS))
    else:
        raise TypeError(f"Invalid dataclass type or instance: '{datamodel}'.")


@utils.optional_lru_cache(maxsize=None, typed=True)
def concretize(
    datamodel_cls: Type[GenericDataModelProto],
    /,
    *type_args: Type,
    class_name: Optional[str] = None,
    module: Optional[str] = None,
    overwrite_definition: bool = True,
    support_pickling: bool = True,
    as_generic_alias: bool = False,
) -> Type[DataModelProto]:
    """Generate a new concrete subclass of a generic Data Model.

    Arguments:
        datamodel_cls: Generic Data Model to be subclassed.
        type_args: Type defintitions replacing the `TypeVars` in
            ``datamodel_cls.__parameters__``.

    Keyword Arguments:
        class_name: Name of the new concrete class. The default value is the
            same of the generic Data Model replacing the `TypeVars` by the provided
            `type_args` in the name.
        module: Value of the ``__module__`` attribute of the new class.
            The default value is the name of the module containing the generic Data Model.
        overwrite_definition: If ``True`` (the default) a previous
            definition of the class in the target module will be overwritten.
        support_pickling: If ``True`` (the default) support for pickling will be added
            by actually inserting the new class into the target `module`.
        as_generic_alias: If ``True`` (the default is ``False``), it will return an
            instance compatible with the class aliases returned by the `typing` module
            machinery for normal classes annotated with :class:``typing.Generic``.
    """

    if not is_generic(datamodel_cls):
        raise TypeError(f"'{datamodel_cls.__name__}' is not a generic model class.")
    if not all(isinstance(t, (type, typing.TypeVar)) for t in type_args):
        raise TypeError(
            f"Only 'type' and 'typing.TypeVar' values can be passed as arguments "
            f"to instantiate a generic model class (received: {type_args})."
        )
    if len(type_args) != len(datamodel_cls.__parameters__):
        raise TypeError(
            f"Instantiating '{datamodel_cls.__name__}' generic model with a wrong number of parameters "
            f"({len(type_args)} used, {len(datamodel_cls.__parameters__)} expected)."
        )

    # Get actual types for generic fields
    type_params_map = dict(zip(datamodel_cls.__parameters__, type_args))
    concrete_annotations = {
        f_name: _substitute_typevars(f_type, type_params_map)
        for f_name, f_type in typing.get_type_hints(datamodel_cls).items()
    }

    # Create new concrete class
    if not class_name:
        arg_names = [
            type_params_map[tp_var].__name__ if tp_var in type_params_map else tp_var.__name__
            for tp_var in datamodel_cls.__parameters__
        ]
        class_name = f"{datamodel_cls.__name__}__{'_'.join(arg_names)}"

    namespace = {
        "__annotations__": concrete_annotations,
        "__module__": module if module else datamodel_cls.__module__,
    }

    concrete_cls = type(class_name, (datamodel_cls,), namespace)
    assert concrete_cls.__module__ == module or not module

    # For pickling to work, the new class has to be added to the proper module
    if support_pickling:
        reference_module_globals = sys.modules[concrete_cls.__module__].__dict__
        if (
            overwrite_definition is False
            and reference_module_globals.get(class_name, concrete_cls) is not concrete_cls
        ):
            raise TypeError(
                f"Existing '{class_name}' symbol in module '{module}'"
                "contains a reference to a different object."
            )
        reference_module_globals[class_name] = concrete_cls

    if _MODEL_FIELDS not in concrete_cls.__dict__:
        # If original model does not inherit from GenericModel,
        # _make_datamodel() hasn't been called yet
        params = getattr(datamodel_cls, _MODEL_PARAMS)
        concrete_cls = _make_datamodel(
            concrete_cls,
            **{
                name: getattr(params, name)
                for name in ("init", "repr", "eq", "order", "unsafe_hash", "frozen", "instantiable")
            },
        )

    if as_generic_alias:
        # To emulate the typing._GenericAlias mechanism, an object compatible
        # with `typing._GenericAlias` is returned. Basically, a `typing._GenericAlias`
        # instance is a class proxy which stores the concrete types passed at
        # creation (`__args__`) and keeps a reference to the original class (`__origin__`).
        # Since this proxy implements a `__mro__entries` method (PEP 560), when this
        # object is found in the list of bases of a new class, it is substituted by the
        # actual class and thus the new type is created as usual.
        # The `GenericTypeAlias` object returned here works exactly in the same way,
        # but stores the new concrete class in `__origin__`, instead the original
        # generic one. For the full picture check also:
        #     https://www.python.org/dev/peps/pep-0526/
        #     https://www.python.org/dev/peps/pep-0560/)
        return GenericTypeAlias(concrete_cls, type_args)
    else:
        return concrete_cls


def validator(name: str) -> Callable[[Callable], Callable]:
    """Decorator to define a custom field validator for a specific field.

    Arguments:
        name: Name of the field to be validated by the decorated function.
    """
    assert isinstance(name, str)

    def _field_validator_maker(func: Callable) -> Callable:
        setattr(func, _FIELD_VALIDATOR_TAG, name)
        return func

    return _field_validator_maker


def root_validator(func: Callable, /) -> classmethod:
    """Decorator to define a custom root validator."""

    cls_method = classmethod(func)
    setattr(cls_method, _ROOT_VALIDATOR_TAG, None)
    return cls_method


def fields(
    datamodel: Union[DataModelProto, Type[DataModelProto]], *, as_dataclass: bool = False
) -> utils.FrozenNamespace:
    """Return the field meta-information of a Data Model.

    Arguments:
        datamodel: A Data Model class or instance.

    Keyword Arguments:
        as_dataclass: If ``True`` (the default is ``False``), field information is returned
            as :class:`dataclass.Field` instances instead of :class:`attr.Attribute`.

    Examples:
        >>> from typing import List
        >>> @datamodel
        ... class Model:
        ...     amount: int = 1
        ...     name: str
        ...     numbers: List[float]
        >>> fields(Model)  # doctest:+ELLIPSIS
        FrozenNamespace(amount=Attribute(name='amount', default=1, ...),\
 name=Attribute(name='name', default=NOTHING, ...),\
 numbers=Attribute(name='numbers', default=NOTHING, ...))
        >>> fields(Model, as_dataclass=True)  # doctest:+ELLIPSIS
        FrozenNamespace(amount=Field(name='amount',type='int',default=1,...),\
 name=Field(name='name',type='str',default=<dataclasses._MISSING_TYPE object ...),\
 numbers=Field(name='numbers',type='List[float]',default=<dataclasses._MISSING_TYPE object ...))

    """
    if not is_datamodel(datamodel):
        raise ValueError(f"Invalid datamodel instance or class: '{datamodel}'.")

    if as_dataclass:
        return utils.FrozenNamespace(**{f.name: f for f in datamodel.__dataclass_fields__})
    else:
        return getattr(datamodel, _MODEL_FIELDS)


def field(
    *,
    default: Any = NOTHING,
    default_factory: Callable[[None], Any] = NOTHING,
    converter: Callable = None,
    init: bool = True,
    repr: bool = True,  # noqa: A002   # shadowing 'repr' python builtin
    hash: Optional[bool] = None,  # noqa: A002   # shadowing 'hash' python builtin
    compare: bool = True,
    metadata: Optional[Mapping[Any, Any]] = None,
) -> Any:  # attr.s lies on purpose in some typings
    """Return an object to identify dataclass fields.

    Keyword Arguments:
        default: If provided, this will be the default value for this field.
            This is needed because the ``field()`` call itself replaces the
            normal position of the default value.
        default_factory: If provided, it must be a zero-argument callable that will
            be called when a default value is needed for this field. Among other
            purposes, this can be used to specify fields with mutable default values.
            It is an error to specify both `default` and `default_factory`.
        init: If ``True`` (the default), this field is included as a parameter to the
            generated ``__init__()`` method.
        repr: If ``True`` (the default), this field is included in the string returned
            by the generated ``__repr__()`` method.
        hash: This can be a ``bool`` or ``None``. If ``True``, this field is included
            in the generated ``__hash__()`` method. If ``None`` (the default), use the
            value of `compare`, which would normally be the expected behavior: a field
            should be considered in the `hash` if it’s used for comparisons.
            Setting this value to anything other than ``None`` is `discouraged`.
        compare: If ``True`` (the default), this field is included in the
            generated equality and comparison methods (__eq__(), __gt__(), et al.).
        metadata: An arbitrary mapping, not used at all by Data Models, and provided
            only as a third-party extension mechanism. Multiple third-parties can each
            have their own key, to use as a namespace in the metadata.

    Note:
        Currently implemented using `attr.ib` (https://www.attrs.org/).
    """

    defaults_kwargs = {}
    if default is not NOTHING:
        defaults_kwargs["default"] = default
    if default_factory is not NOTHING:
        if "default" in defaults_kwargs:
            raise ValueError("Cannot specify both default and default_factory.")
        defaults_kwargs["factory"] = default_factory

    if isinstance(converter, bool):
        converter = AUTO_CONVERTER if converter is True else None

    return attr.ib(  # type: ignore  # attr.s lies on purpose in some typings
        **defaults_kwargs,
        converter=converter,
        init=init,
        repr=repr,
        hash=hash,
        eq=compare,
        order=compare,
        metadata=metadata,
    )


def datamodel(
    cls: Type = None,
    /,
    *,
    init: bool = True,  # noqa: A002   # shadowing 'init' python builtin
    repr: bool = True,  # noqa: A002   # shadowing 'repr' python builtin
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    instantiable: bool = True,
) -> Union[Type, Callable[[Type], Type]]:
    """Add dunder and validation methods to the passed class.

    Examines PEP 526 __annotations__ to determine field types and creates
    strict type validation functions for the fields.

    Arguments:
        cls: Original class definition.

    Keyword Arguments:
        init: If ``True`` (the default), a ``__init__()`` method with validation
            will be generated. If the class already defines ``__init__()``,
            an error is raised.
        repr: If ``True`` (the default), a ``__repr__()`` method will be generated.
            If the class already defines ``__repr__()``, it will be overwritten.
        eq: If ``True`` (the default), ``__eq__()`` and ``__ne__()`` methods will be
            generated. This method compares the class as if it were a tuple of its
            fields. Both instances in the comparison must be of identical type.
        order:  If ``True`` (the default is ``False``), add ``__lt__()``, ``__le__()``,
            ``__gt__()``, and ``__ge__()`` methods that behave like `eq` above and
            allow instances to be ordered. If ``None`` mirror value of `eq`.
        unsafe_hash: If ``False`` (the default), a ``__hash__()`` method is generated
            in a safe way according to how ``eq`` and ``frozen`` are set, or set to
            ``None`` (disabled) otherwise. If ``True``, a ``__hash__()`` method is
            generated anyway (use with care). See :func:`dataclasses.dataclass` for
            the complete explanation.
        frozen: If ``True`` (the default is ``False``), assigning to fields will generate
            an exception. This emulates read-only frozen instances. The ``__setattr__()``
            and ``__delattr__()`` methods should not be defined in the class.
        instantiable: If ``False`` (default is ``True``) the class will contain an
            invalid ``__init__()`` method that raises an exception.

    Note:
        Currently implemented using `attr.s` (https://www.attrs.org/).
    """

    def _decorator(cls: Type) -> Type:
        return _make_datamodel(
            cls,
            init=init,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=frozen,
            instantiable=instantiable,
        )

    # This works for both @datamodel or @datamodel() decorations
    return _decorator(cls) if cls is not None else _decorator


class DataModel:
    """Base class to automatically convert any subclass into a Data Model.

    Inheriting from this class is equivalent to apply the :func:`datamodel`
    decorator to a class, except that all descendants will be also converted
    automatically in Data Models (which does not happen when explicitly
    applying the decorator).

    See :func:`datamodel` for the description of the parameters.
    """

    @classmethod
    def __init_subclass__(
        cls,
        /,
        *,
        init: bool = True,
        repr: bool = True,  # noqa: A002   # shadowing 'repr' python builtin
        eq: bool = True,
        order: bool = False,
        unsafe_hash: bool = False,
        frozen: bool = False,
        instantiable: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)  # type: ignore  # super() does not need to be object
        _make_datamodel(
            cls,
            init=init,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=frozen,
            instantiable=instantiable,
        )


class GenericTypeAlias(typing._GenericAlias, _root=True):  # type: ignore  # typing._GenericAlias is not visible for mypy
    """Custom GenericAlias to abstract :class:`typing._GenericAlias` implementation details."""

    def __getitem__(self, args: Union[Type, Tuple[Type]]) -> Type[DataModelProto]:
        type_args = args if isinstance(args, tuple) else (args,)
        return concretize(self.__origin__, *type_args, as_generic_alias=True)

    @property  # type: ignore  # Read-only property cannot override read-write property
    def __class__(self) -> Type:
        assert isinstance(self.__origin__, type)
        return self.__origin__


# -- WIP --
def _make_type_coercer(type_hint: Type[T]) -> Callable[[Any], T]:
    # TODO: implement this method
    return type_hint if isinstance(type_hint, type) else lambda x: x  # type: ignore


# def _frozen_setattr(instance, attribute, value):
#     raise attr.exceptions.FrozenAttributeError(
#         f"Trying to modify immutable '{attribute.name}' attribute in '{type(instance).__name__}' instance."
#     )


# def _valid_setattr(instance, attribute, value):
#     print("SET", attribute, value)
