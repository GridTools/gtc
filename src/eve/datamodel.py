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

"""DataModel class and utils."""


from __future__ import annotations

import abc
import collections
import dataclasses
import functools
import typing
from typing import Any, Callable, ClassVar, Dict, Generic, List, Tuple, Type, get_origin

import attr


try:
    # For perfomance reasons, try to use cytoolz when possible (using cython)
    import cytoolz as toolz
except ModuleNotFoundError:
    # Fall back to pure Python toolz
    import toolz

from . import utils  # isort:skip
from .concepts import NOTHING  # isort:skip


def _frozen_setattr(instance, attribute, value):
    raise attr.exceptions.FrozenAttributeError(
        f"Trying to modify immutable '{attribute.name}' attribute in '{type(instance).__name__}' instance."
    )


def _valid_setattr(instance, attribute, value):
    print("SET", attribute, value)


try:
    BasesMeta = (typing._ProtocolMeta, abc.ABCMeta)
except Exception:
    BasesMeta = (abc.ABCMeta,)


class DataModelMeta(*BasesMeta):

    __ATTR_SETTINGS = dict(auto_attribs=True, slots=False, kw_only=True)
    __FIELD_VALIDATOR_TAG = "__field_validator_tag"
    __FIELD_CONVERTER_TAG = "__field_converter_tag"
    __ROOT_VALIDATOR_TAG = "__root_validator_tag"
    __ROOT_VALIDATORS_NAME = "__datamodel_validators__"
    __TYPEVARS = "__datamodel_typevars__"

    @typing.no_type_check
    def __new__(
        mcls: Type[abc.ABCMeta],
        name: str,
        bases: Tuple[Type],
        namespace: Dict[str, Any],
        **kwargs: Any,
    ):
        # Direct return path for bare DataModel class
        print(f"{namespace.get('__qualname__')=}")
        if namespace.get("__qualname__") == "DataModel":
            return super().__new__(mcls, name, bases, namespace, **kwargs)

        # Create a plain version of the Python class without magic and replace it later
        # by the enhanced version. This is just a workaround to get the correct mro and
        # resolved type annotations, which is not possible before the creation of the class
        tmp_plain_cls = super().__new__(mcls, name, bases, namespace)
        resolved_annotations = typing.get_type_hints(tmp_plain_cls)

        # Create attr.ibs for annotated fields (excluding ClassVars)
        annotations = namespace.setdefault("__annotations__", {})
        for key in annotations:
            type_hint = resolved_annotations[key]
            if typing.get_origin(type_hint) is not ClassVar:
                type_validator = _make_strict_type_validator(type_hint)
                if name not in namespace:
                    namespace[key] = attr.ib(validator=type_validator)
                elif not isinstance(namespace[key], attr._make._CountingAttr):
                    namespace[key] = attr.ib(default=namespace[key], validator=type_validator)
                else:
                    assert False

        # Collect special methods (converters and validators)
        field_converters = {}
        field_validators = {}
        root_validators = []

        # Collect root validators from bases
        for base in reversed(tmp_plain_cls.__mro__[1:]):
            for validator in getattr(base, mcls.__ROOT_VALIDATORS_NAME, []):
                if validator not in root_validators:
                    root_validators.append(validator)

        # Collect converters and root and field validators in namespace
        for _, member in namespace.items():
            if hasattr(member, mcls.__FIELD_CONVERTER_TAG):
                field_name = getattr(member, mcls.__FIELD_CONVERTER_TAG)
                delattr(member, mcls.__FIELD_CONVERTER_TAG)
                field_converters[field_name] = member
            elif hasattr(member, mcls.__FIELD_VALIDATOR_TAG):
                field_name = getattr(member, mcls.__FIELD_VALIDATOR_TAG)
                delattr(member, mcls.__FIELD_VALIDATOR_TAG)
                field_validators[field_name] = member
            elif hasattr(member, mcls.__ROOT_VALIDATOR_TAG):
                delattr(member, mcls.__ROOT_VALIDATOR_TAG)
                root_validators.append(member)

        # Add collected fields validators to fields' attr.ibs
        for field_name, field_validator in field_validators.items():
            field_attrib = namespace.get(field_name, None)
            if not field_attrib:
                # Field has not been defined in the current class namespace,
                # look for field definition in the base classes
                found = False
                for base in tmp_plain_cls.__mro__[1:]:
                    for base_field_attrib in getattr(base, "__attrs_attrs__", []):
                        if base_field_attrib.name == field_name:
                            # Clone the existing definition and add the new validator (attrs recommendation)
                            annotations[field_name] = base.__annotations__[field_name]
                            field_attrib = namespace[field_name] = mcls._make_attrib_from_attr(
                                base_field_attrib, include_type=False
                            )
                            found = True
                            break
                    if found:
                        break

                if not found:
                    raise TypeError(f"Validator assigned to non existing '{field_name}' field.")

            # Add field validator
            assert isinstance(field_attrib, attr._make._CountingAttr)
            field_attrib.validator(field_validator)

        namespace[mcls.__ROOT_VALIDATORS_NAME] = tuple(root_validators)

        # Add custom __attrs_post_init__ (calling __post_init__ if it exits for dataclasses emulation)
        has_custom_init = "__post_init__" in namespace
        namespace["__attrs_post_init__"] = mcls._make_attrs_post_init(has_custom_init)

        # Add extra custom methods
        namespace["is_generic"] = mcls._make_is_generic()

        # Create final Python class and convert it into an attrs class
        plain_cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        cls = attr.define(**mcls.__ATTR_SETTINGS)(plain_cls)

        # Add dataclasses compatibility
        dataclass_fields = []
        for field_attr in cls.__attrs_attrs__:
            dataclass_fields.append(mcls._make_dataclass_field_from_attr(field_attr))
        setattr(cls, "__dataclass_fields__", tuple(dataclass_fields))

        return cls

    @utils.optional_lru_cache(maxsize=None, typed=True)
    def __getitem__(cls, type_args):
        if not cls.is_generic():
            raise TypeError(f"'{cls.__name__}' is not a generic model class.")
        if not all(isinstance(t, (type, typing.TypeVar)) for t in type_args):
            raise TypeError(
                f"Only 'type' and 'typing.TypeVar' values can be passed as arguments "
                f"to instantiate a generic model class (received: {type_args})."
            )
        if len(type_args) > len(cls.__parameters__):
            raise TypeError(
                f"Instantiating '{cls.__name__}' generic model with too many parameters "
                f"({len(type_args)} used, {len(cls.__parameters__)} expected)."
            )

        type_params_map = dict(zip(cls.__parameters__, type_args))

        # Get actual types for generic fields
        concrete_annotations = {}
        for f_name, f_type in typing.get_type_hints(cls).items():
            if isinstance(f_type, typing.TypeVar) and f_type in type_params_map:
                concrete_annotations[f_name] = type_params_map[f_type]
                continue

            origin = typing.get_origin(f_type)
            if origin not in (None, ClassVar) and getattr(f_type, "__parameters__", None):
                concrete_type_args = tuple([type_params_map[p] for p in f_type.__parameters__])
                concrete_annotations[f_name] = f_type[concrete_type_args]

        concrete_name = f"{cls.__name__}__{'_'.join(t.__name__ for t in type_args)}__"
        # print(concrete_name, concrete_annotations)

        return type(concrete_name, (cls,), dict(__annotations__=concrete_annotations),)

    @staticmethod
    def tag_field_converter(__name: str):
        assert isinstance(__name, str)

        def _field_converter_maker(func):
            setattr(func, DataModelMeta.__FIELD_CONVERTER_TAG, __name)
            return func

        return _field_converter_maker

    @staticmethod
    def tag_field_validator(__name: str):
        assert isinstance(__name, str)

        def _field_validator_maker(func):
            setattr(func, DataModelMeta.__FIELD_VALIDATOR_TAG, __name)
            return func

        return _field_validator_maker

    @staticmethod
    def tag_root_validator(func):
        cls_method = classmethod(func)
        setattr(cls_method, DataModelMeta.__ROOT_VALIDATOR_TAG, None)
        return cls_method

    @staticmethod
    def _make_attrs_post_init(has_custom_init=False):
        if has_custom_init:

            def __attrs_post_init__(self):
                if attr._config._run_validators is True:
                    cls = type(self)
                    for validator in cls.__datamodel_validators__:
                        validator.__get__(cls)(self)
                    self.__post_init__()

        else:

            def __attrs_post_init__(self):
                if attr._config._run_validators is True:
                    cls = type(self)
                    for validator in cls.__datamodel_validators__:
                        validator.__get__(cls)(self)

        return __attrs_post_init__

    @staticmethod
    def _make_is_generic():
        def is_generic(cls):
            return len(cls.__parameters__) > 0 if hasattr(cls, "__parameters__") else False

        return classmethod(is_generic)

    @staticmethod
    def _make_attrib_from_attr(field_attr, include_type=False):
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

        return attr.ib(**{key: getattr(field_attr, key) for key in members})

    @staticmethod
    def _make_dataclass_field_from_attr(field_attr):
        MISSING = getattr(dataclasses, "MISSING", getattr(dataclasses, "_MISSING", NOTHING))
        default = MISSING
        default_factory = MISSING
        if isinstance(field_attr.default, attr.Factory):
            default_factory = field_attr.default.factory
        elif field_attr.default is not attr.NOTHING:
            default = field_attr.default

        dataclasses_field = dataclasses.Field(
            default=default,
            default_factory=default_factory,
            init=field_attr.init,
            repr=field_attr.repr if not callable(field_attr.repr) else None,
            hash=field_attr.hash,
            compare=field_attr.eq,
            # compare = field_attr.eq and field_attr.order, # it would be more accurate but messes up hash=None meaning
            metadata=field_attr.metadata,
        )
        dataclasses_field.name = field_attr.name
        dataclasses_field.type = field_attr.type

        return dataclasses_field


converter = DataModelMeta.tag_field_converter
validator = DataModelMeta.tag_field_validator
root_validator = DataModelMeta.tag_root_validator


class DataModel(metaclass=DataModelMeta):
    pass


def _make_strict_type_validator(type_hint):
    origin_type = typing.get_origin(type_hint)
    type_args = typing.get_args(type_hint)

    print(f"_make_strict_type_validator({type_hint}): {type_hint=}, {origin_type=}, {type_args=}")

    if isinstance(type_hint, type) and not type_args:
        return attr.validators.instance_of(type_hint)

    elif isinstance(type_hint, typing.TypeVar):
        if type_hint.__bound__:
            return attr.validators.instance_of(type_hint.__bound__)
        else:
            return lambda _: None

    elif type_hint is Any:
        return lambda _: None

    elif origin_type is typing.Literal:
        return _make_literal_validator(type_args)

    elif origin_type is typing.Union:
        return _make_union_validator(type_args)

    elif isinstance(origin_type, type):

        if issubclass(origin_type, tuple):
            return _make_tuple_validator(type_args)

        elif issubclass(origin_type, (collections.abc.Sequence, collections.abc.Set)):
            assert len(type_args) == 1
            member_type_hint = type_args[0]
            return attr.validators.deep_iterable(
                member_validator=_make_strict_type_validator(member_type_hint),
                iterable_validator=attr.validators.instance_of(origin_type),
            )

        elif issubclass(origin_type, collections.abc.Mapping):
            assert len(type_args) == 2
            key_type_hint, value_type_hint = type_args
            return attr.validators.deep_mapping(
                key_validator=_make_strict_type_validator(key_type_hint),
                value_validator=_make_strict_type_validator(value_type_hint),
                mapping_validator=attr.validators.instance_of(origin_type),
            )

    else:
        raise TypeError(f"Type description '{type_hint}' is not supported.")


def _make_literal_validator(type_args):
    return _make_or_validator([_LiteralValidator(t) for t in type_args])


def _make_union_validator(type_args):
    if len(type_args) == 2 and (type_args[1] is type(None)):
        non_optional_validator = _make_strict_type_validator(type_args[0])
        return attr.validators.optional(non_optional_validator)
    else:
        return _make_or_validator([_make_strict_type_validator(t) for t in type_args])


def _make_tuple_validator(type_args):
    if len(type_args) == 2 and (type_args[1] is Ellipsis):
        member_type_hint = type_args[0]
        return attr.validators.deep_iterable(
            member_validator=_make_strict_type_validator(member_type_hint),
            iterable_validator=attr.validators.instance_of(tuple),
        )
    else:
        return _TupleValidator([_make_strict_type_validator(t) for t in type_args])


def _make_or_validator(*validators):
    vals = tuple(utils.flatten(validators))
    if len(vals) == 1:
        return vals[0]
    else:
        return _OrValidator(vals)


@attr.define
class _TupleValidator:
    """
    Compose many validators to a single one.
    """

    validators: Tuple[Callable]

    def __call__(self, instance, attribute, value):
        if not isinstance(value, tuple):
            raise ValueError(f"")
        if len(value) != len(self.validators):
            raise ValueError(f"Wrong number or tuple elements")

        i = None
        try:
            for i, (item_value, item_validator) in enumerate(zip(value, self.validators)):
                item_validator(instance, attribute, item_value)
        except Exception as e:
            raise ValueError(
                f"Invalid item {f'at index[{i}] ' if i is not None else ''}in the provided value '{value}' for {attribute.name} field."
            ) from e


@attr.define
class _OrValidator:
    """
    Compose many validators to a single one.
    """

    validators: Tuple[Callable]

    def __call__(self, instance, attribute, value):
        passed = False
        for v in self.validators:
            try:
                v(instance, attribute, value)
                passed = True
                break
            except Exception:
                pass

        if not passed:
            raise ValueError(
                f"Provided value '{value}' for {attribute.name} field does not verify any of the possible validators."
            )


@attr.define
class _LiteralValidator:
    """
    Literal validator.
    """

    literal: Any

    def __call__(self, instance, attribute, value):
        if not (value == self.literal or value is self.literal):
            raise ValueError(
                f"Provided value '{value}' for {attribute.name} field does not match {self.literal}."
            )


# _ValidatorType = Callable[[Any, attr.att Attribute[_T], _T], Any]
# _ConverterType = Callable[[Any], Any]


#     @_tp_cache
#     def __class_getitem__(cls, params):
#         if not isinstance(params, tuple):
#             params = (params,)
#         if not params and cls is not Tuple:
#             raise TypeError(
#                 f"Parameter list to {cls.__qualname__}[...] cannot be empty")
#         msg = "Parameters to generic types must be types."
#         params = tuple(_type_check(p, msg) for p in params)
#         if cls in (Generic, Protocol):
#             # Generic and Protocol can only be subscripted with unique type variables.
#             if not all(isinstance(p, TypeVar) for p in params):
#                 raise TypeError(
#                     f"Parameters to {cls.__name__}[...] must all be type variables")
#             if len(set(params)) != len(params):
#                 raise TypeError(
#                     f"Parameters to {cls.__name__}[...] must all be unique")
#         else:
#             # Subscripting a regular Generic subclass.
#             _check_generic(cls, params)
#         return _GenericAlias(cls, params)


# def _check_generic(cls, parameters):
#     """Check correct count for parameters of a generic cls (internal helper).
#     This gives a nice error message in case of count mismatch.
#     """
#     if not cls.__parameters__:
#         raise TypeError(f"{cls} is not a generic class")
#     alen = len(parameters)
#     elen = len(cls.__parameters__)
#     if alen != elen:
#         raise TypeError(f"Too {'many' if alen > elen else 'few'} parameters for {cls};"
#                         f" actual {alen}, expected {elen}")
