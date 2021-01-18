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

import functools
import itertools

from toolz.functoolz import compose

import eve
from gtc import common
from gtc.unstructured import gtir2, usid2


_C_TYPES = {
    common.DataType.FLOAT64: "double",
    common.DataType.FLOAT32: "float",
    common.DataType.INT32: "::std::int32_t",
    common.DataType.UINT32: "::std::uint32_t",
    common.DataType.BOOLEAN: "bool",
}

_LITERAL_CONVERTERS = {
    common.DataType.FLOAT64: compose(str, float),
    common.DataType.FLOAT32: lambda x: str(float(x)) + "f",
    common.DataType.INT32: compose(str, int),
    common.DataType.UINT32: lambda x: str(int(x)) + "u",
    common.DataType.BOOLEAN: lambda x: str(bool(x)).lower(),
}


def _loc2str(x: common.LocationType):
    return x.name.lower()


class _PrimaryCompositeExtractor(eve.NodeVisitor):
    def visit_Literal(self, src: gtir2.Literal, **kwargs):
        return set()

    def visit_FieldAccess(self, src: gtir2.FieldAccess, primary):
        return {src.name} if src.location == primary else set()

    def visit_SparseFieldAccess(self, src: gtir2.SparseFieldAccess, primary):
        assert src.primary == primary
        return {src.name}

    def visit_BinaryOp(self, src: gtir2.BinaryOp, **kwargs):
        return self.visit(src.left, **kwargs) | self.visit(src.right, **kwargs)

    def visit_NeighborReduce(self, src: gtir2.NeighborReduce, primary):
        assert src.location.primary == primary
        return {src.location.connectivity} | self.visit(src.body, primary=primary)

    def visit_Assign(self, src: gtir2.Assign, **kwargs):
        return self.visit(src.left, **kwargs) | self.visit(src.right, **kwargs)

    def visit_Stencil(self, src: gtir2.Stencil):
        return functools.reduce(
            lambda acc, e: acc | self.visit(e, primary=src.location.name), src.body, set()
        )


_extract_primary_composite = _PrimaryCompositeExtractor().visit


class _SecondaryCompositeExtractor(eve.NodeVisitor):
    def visit_SparseFieldAccess(self, src: gtir2.SparseFieldAccess, **kwargs):
        return set()

    def visit_Literal(self, src: gtir2.Literal, **kwargs):
        return set()

    def visit_FieldAccess(self, src: gtir2.FieldAccess, secondary, **kwargs):
        return {src.name} if src.location == secondary else {}

    def visit_BinaryOp(self, src: gtir2.BinaryOp, **kwargs):
        return self.visit(src.left, **kwargs) | self.visit(src.right, **kwargs)

    def visit_NeighborReduce(self, src: gtir2.NeighborReduce):
        return {src.location.connectivity: self.visit(src.body, secondary=src.location.name)}


_extract_secondary_composite = _SecondaryCompositeExtractor().visit


def _merge_dicts_of_sets(*srcs):
    res = {}
    for k in set(itertools.chain(*srcs)):
        res[k] = set()
        for src in srcs:
            if k in src:
                res[k] = res[k] | src[k]
    return res


class _SecondaryCompositesExtractor(eve.NodeVisitor):
    def visit_FieldAccess(self, src: gtir2.FieldAccess):
        return {}

    def visit_Literal(self, src: gtir2.Literal):
        return {}

    def visit_Assign(self, src: gtir2.Assign):
        return self.visit(src.right)

    def visit_NeighborReduce(self, src: gtir2.NeighborReduce):
        return _extract_secondary_composite(src)

    def visit_BinaryOp(self, src: gtir2.BinaryOp):
        return _merge_dicts_of_sets(self.visit(src.left), self.visit(src.right))

    def visit_Stencil(self, src: gtir2.Stencil):
        return _merge_dicts_of_sets(*(self.visit(e) for e in src.body)).items()


_extract_secondary_composites = _SecondaryCompositesExtractor().visit


class _Visitor(eve.NodeVisitor):
    def visit_FieldAccess(self, src: gtir2.FieldAccess):
        return usid2.FieldAccess(name=src.name, location=src.location)

    def visit_SparseFieldAccess(self, src: gtir2.SparseFieldAccess):
        return usid2.FieldAccess(name=src.name, location=src.primary)

    def visit_Literal(self, src: gtir2.Literal):
        return usid2.Literal(value=_LITERAL_CONVERTERS[src.dtype](src.value))

    def visit_BinaryOp(self, src: gtir2.BinaryOp):
        return usid2.BinaryOp(op=src.op, left=self.visit(src.left), right=self.visit(src.right))

    def visit_NeighborReduce(self, src: gtir2.NeighborReduce):
        return usid2.NeighborReduce(
            op=src.op,
            dtype=_C_TYPES[src.dtype],
            connectivity=src.location.connectivity,
            primary=src.location.primary,
            secondary=src.location.name,
            body=self.visit(src.body),
        )

    def visit_Assign(self, src: gtir2.Assign):
        return usid2.Assign(left=self.visit(src.left), right=self.visit(src.right))

    def visit_Stencil(self, src: gtir2.Stencil):
        return usid2.Kernel(
            location_type=_loc2str(src.location.location_type),
            primary=usid2.Composite(name=src.location.name, items=_extract_primary_composite(src)),
            secondaries=tuple(
                usid2.Composite(name=name, items=items)
                for name, items in _extract_secondary_composites(src)
            ),
            body=tuple(self.visit(e) for e in src.body),
        )

    def visit_Connectivity(self, src: gtir2.Connectivity):
        return usid2.Connectivity(
            name=src.name, max_neighbors=src.max_neighbors, has_skip_values=src.has_skip_values
        )

    def visit_Field(self, src: gtir2.Field):
        return usid2.Field(name=src.name)

    def visit_SparseField(self, src: gtir2.SparseField):
        return usid2.SparseField(name=src.name, connectivity=src.connectivity)

    def visit_Computation(self, src: gtir2.Computation):
        return usid2.Computation(
            name=src.name,
            connectivities=tuple(self.visit(e) for e in src.connectivities),
            args=tuple(self.visit(e) for e in src.args),
            temporaries=tuple(
                usid2.Temporary(
                    name=e.name, location_type=_loc2str(e.location_type), dtype=_C_TYPES[e.dtype]
                )
                for e in src.temporaries
            ),
            kernels=tuple(self.visit(e) for e in src.stencils),
        )


transform = _Visitor().visit
