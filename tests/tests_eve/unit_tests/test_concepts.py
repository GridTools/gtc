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

import pydantic
import pytest

import eve

from .. import common_definitions


@pytest.fixture(
    params=[
        "valid_annotation_",
        "v_",
        "v3212_32_",
        "VV_VV_00_",
        "invalid_annotation__",
        "_invalid_annotation__",
        "__invalid_annotation__",
        "_",
    ]
)
def annotation_name(request):
    yield request.param


@pytest.fixture(params=["data", 0, 1.1, None, [1, 2, 3], {"a": 1, 1: "a"}, lambda x: x])
def annotation_value(request):
    yield request.param


class TestNode:
    def test_validation(self, invalid_sample_node_maker):
        with pytest.raises(pydantic.ValidationError):
            invalid_sample_node_maker()

    def test_mutability(self, sample_node):
        if "value" in sample_node.__fields__:
            sample_node.int_value = 123456

    def test_inmutability(self, frozen_sample_node):
        with pytest.raises(TypeError):
            frozen_sample_node.int_value = 123456

    def test_copy(self, sample_node):
        foo = {"a": 1, "b": 2}
        bla = [1, "bla", (1, 2)]
        const_bla = tuple(bla)
        sample_node.foo_ = foo
        sample_node.bla_ = bla
        sample_node.const_bla_ = const_bla

        node_copy = sample_node.copy()
        assert sample_node == node_copy
        assert sample_node.__node_id__ == node_copy.__node_id__
        assert sample_node.__node_annotations__ == node_copy.__node_annotations__
        assert sample_node.__node_annotations__ is node_copy.__node_annotations__
        assert node_copy.foo_ is foo
        assert node_copy.bla_ is bla
        assert node_copy.const_bla_ is const_bla

        node_copy = copy.copy(sample_node)
        assert sample_node == node_copy
        assert sample_node.__node_id__ == node_copy.__node_id__
        assert sample_node.__node_annotations__ == node_copy.__node_annotations__
        assert sample_node.__node_annotations__ is node_copy.__node_annotations__
        assert node_copy.foo_ is foo
        assert node_copy.bla_ is bla
        assert node_copy.const_bla_ is const_bla

        node_copy = copy.deepcopy(sample_node)
        assert sample_node == node_copy
        assert sample_node.__node_id__ == node_copy.__node_id__
        assert sample_node.__node_annotations__ == node_copy.__node_annotations__
        assert sample_node.__node_annotations__ is not node_copy.__node_annotations__
        assert node_copy.foo_ is node_copy.__node_annotations__.foo_
        assert node_copy.foo_ is not foo
        assert node_copy.bla_ is not bla
        assert node_copy.const_bla_ is const_bla

    def test_field_naming(self):
        class NodeWithPrivateAttrs(eve.concepts.BaseNode):
            _private_int: int = 0
            int_value: int = 1

        assert len(NodeWithPrivateAttrs().__fields__) == 1

        with pytest.raises(TypeError, match="data annotation"):

            class NodeWithDataAnnotationNamedFields(eve.concepts.BaseNode):
                int_value_: int

        with pytest.raises(TypeError, match="internal field"):

            class NodeWithInternalNamedFields(eve.concepts.BaseNode):
                int_value__: int

    def test_private_attrs(self, sample_node):
        assert all(
            eve.concepts._is_private_attr_name(name) for name in sample_node.private_attrs_names
        )
        assert sample_node.private_attrs_names >= {"__node_id__", "__node_annotations__"}

    def test_children(self, sample_node):
        children_names = set(name for name in sample_node.iter_children_names())
        field_names = set(sample_node.__fields__.keys())

        assert all(eve.concepts._is_child_field_name(name) for name in children_names)
        assert children_names <= field_names
        assert all(
            eve.concepts._is_internal_field_name(name) for name in field_names - children_names
        )

        assert all(
            name1 is name2
            for (name1, _), name2 in zip(
                sample_node.iter_children(), sample_node.iter_children_names()
            )
        )
        assert all(
            node1 is node2
            for (_, node1), node2 in zip(
                sample_node.iter_children(), sample_node.iter_children_values()
            )
        )

    def test_node_annotations(self, sample_node, annotation_name, annotation_value):
        if eve.concepts._is_data_annotation_name(annotation_name):
            setattr(sample_node, annotation_name, annotation_value)
            assert getattr(sample_node, annotation_name) == annotation_value
        else:
            with pytest.raises(ValueError, match=f'has no field "{annotation_name}"'):
                setattr(sample_node, annotation_name, annotation_value)

    def test_node_metadata(self, sample_node):
        assert all(name in sample_node.__node_children__ for name, _ in sample_node.iter_children())
        assert all(
            isinstance(metadata, dict)
            and isinstance(metadata["definition"], pydantic.fields.ModelField)
            for metadata in sample_node.__node_children__.values()
        )

    def test_unique_id(self, sample_node_maker):
        node_a = sample_node_maker()
        node_b = sample_node_maker()
        node_c = sample_node_maker()

        assert node_a.__node_id__ != node_b.__node_id__ != node_c.__node_id__

    def test_custom_id(self, source_location, sample_node_maker):
        custom_id = "my_custom_id"
        my_node = common_definitions.LocationNode(loc=source_location)
        other_node = sample_node_maker()
        my_node.__node_id__ = custom_id

        assert my_node.__node_id__ == custom_id
        assert my_node.__node_id__ != other_node.__node_id__
