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

import collections
import functools
import inspect
import sys

from gt_frontend import built_in_types

import eve
from gtc import common
from gtc.unstructured import gtir2, gtscript_ast2, py_to_gtscript2


def _is_conncetivity_annotation(src):
    return hasattr(src, "__supertype__") and issubclass(
        src.__supertype__, built_in_types.Connectivity
    )


def _is_field_annotation(src):
    return issubclass(src, built_in_types.Field)


def _is_sparse_field_annotation(src):
    return issubclass(src, built_in_types.SparseField)


def _is_legit_annotation(src):
    return (
        _is_conncetivity_annotation(src)
        or _is_field_annotation(src)
        or _is_sparse_field_annotation(src)
    )


def _extract_params(fun):
    res = tuple(p for _, p in inspect.signature(fun).parameters.items())
    for p in res:
        if not _is_legit_annotation(p.annotation):
            raise RuntimeError(f"unexpected type annotation: {p}")
    return res


def _extract_connectivity(param):
    args = param.annotation.__supertype__.args
    assert len(args) > 3
    return gtir2.Connectivity(
        name=param.name,
        primary=args[0],
        secondary=args[1],
        max_neighbors=args[2],
        has_skip_values=args[3] if len(args) > 3 else True,
    )


def _extract_arg(param, connectivity_type_to_name):
    args = param.annotation.args
    if _is_field_annotation(param.annotation):
        assert len(args) == 2
        return gtir2.Field(name=param.name, location_type=args[0], dtype=args[1])
    else:
        assert len(args) == 2
        return gtir2.SparseField(
            name=param.name, connectivity=connectivity_type_to_name[args[0]], dtype=args[1]
        )


_SORTED_TYPES = (
    common.DataType.INVALID,
    common.DataType.AUTO,
    common.DataType.FLOAT64,
    common.DataType.FLOAT32,
    common.DataType.INT32,
    common.DataType.UINT32,
    common.DataType.BOOLEAN,
)


def _common_type(lhs: common.DataType, rhs: common.DataType) -> common.DataType:
    for t in _SORTED_TYPES:
        if lhs == t or rhs == t:
            return t
    raise RuntimeError(f"unsupported types {lhs}, {rhs}")


class _DeduceExprTypeVisotor(eve.NodeVisitor):
    def visit_Literal(self, src: gtir2.Literal, tbl):
        return src.dtype

    def visit_NeighborReduce(self, src: gtir2.NeighborReduce, tbl):
        return src.dtype

    def visit_FieldAccess(self, src: gtir2.FieldAccess, tbl):
        return tbl[src.name].dtype

    def visit_SparseFieldAccess(self, src: gtir2.SparseFieldAccess, tbl):
        return tbl[src.name].dtype

    def visit_BinaryOp(self, src: gtir2.BinaryOp, tbl):
        return _common_type(self.visit(src.left, tbl=tbl), self.visit(src.right, tbl=tbl))


def _deduce_expr_type(src: gtir2.Expr, tbl):
    return _DeduceExprTypeVisotor().visit(src, tbl=tbl)


_PY_TYPE_TO_DATA_TYPE = {
    int: common.DataType.INT32,
    float: common.DataType.FLOAT64 if sys.float_info.mant_dig >= 53 else common.DataType.FLOAT32,
}


class _Visitor(eve.NodeVisitor):
    def visit_Constant(self, src: gtscript_ast2.Constant, **kwargs):
        return gtir2.Literal(value=src.value, dtype=_PY_TYPE_TO_DATA_TYPE[type(src.value)])

    def visit_Call(self, src: gtscript_ast2.Call, tbl, location):
        assert isinstance(location, gtir2.PrimaryLocation)
        assert len(src.args) == 1
        generator_exp = src.args[0]
        assert isinstance(generator_exp, gtscript_ast2.GeneratorExp)
        assert len(generator_exp.generators) == 1
        generator = generator_exp.generators[0]
        target = generator.target
        assert isinstance(target, gtscript_ast2.Name)
        secondary = gtir2.SecondaryLocation(
            name=target.name,
            connectivity=self.visit(generator.iter_, tbl=tbl, location=location),
            primary=location.name,
        )
        body = self.visit(generator_exp.elt, tbl=tbl, location=secondary)
        return gtir2.NeighborReduce(
            op=gtir2.ReduceOperator(src.func.name),
            location=secondary,
            dtype=_deduce_expr_type(body, tbl),
            body=body,
        )

    def visit_Name(self, src: gtscript_ast2.Name, tbl, location, is_target=False):
        if src.name in tbl:
            field = tbl[src.name]
            if isinstance(field, gtir2.Field):
                if isinstance(location, gtir2.PrimaryLocation):
                    assert field.location_type == location.location_type
                    loc = location.name
                else:
                    assert isinstance(location, gtir2.SecondaryLocation)
                    if field.location_type == tbl[location.connectivity].primary:
                        loc = location.primary
                    elif field.location_type == tbl[location.connectivity].secondary:
                        loc = location.name
                    else:
                        raise RuntimeError(f"invalid field access {src}")
                return gtir2.FieldAccess(name=field.name, location=loc)
            elif isinstance(field, gtir2.SparseField):
                assert isinstance(location, gtir2.SecondaryLocation)
                assert field.connectivity == location.connectivity
                return gtir2.SparseFieldAccess(
                    name=field.name, primary=location.primary, secondary=location.name
                )
            elif isinstance(field, gtir2.Connectivity):
                assert isinstance(location, gtir2.PrimaryLocation)
                assert field.primary == location.location_type
                return field.name
            else:
                raise RuntimeError(f"invalid access {src} to {field}")
        elif is_target:
            assert isinstance(location, gtir2.PrimaryLocation)
            return gtir2.FieldAccess(name=src.name, location=location.name)
        else:
            raise RuntimeError(f"unbound name {src}")

    def visit_Subscript(self, src: gtscript_ast2.Subscript, location, **kwargs):
        res = self.visit(src.value, location=location, **kwargs)
        if isinstance(res, gtir2.FieldAccess):
            assert isinstance(src.index, gtscript_ast2.Name)
            assert src.index.name == res.location
        elif isinstance(res, gtir2.SparseFieldAccess):
            assert isinstance(src.index, collections.Sequence)
            assert len(src.index) == 2
            assert src.index[0].name == res.primary
            assert src.index[1].name == res.secondary
        elif isinstance(res, str):
            assert src.index.name == location.name
        return res

    def visit_BinOp(self, src: gtscript_ast2.BinOp, **kwargs):
        return gtir2.BinaryOp(
            op=common.BinaryOperator(src.op),
            left=self.visit(src.left, **kwargs),
            right=self.visit(src.right, **kwargs),
        )

    def visit_Assign(self, src: gtscript_ast2.Assign, tbl, primary):
        assert len(src.targets) == 1
        target = src.targets[0]
        left = self.visit(target, tbl=tbl, location=primary, is_target=True)
        right = self.visit(src.value, tbl=tbl, location=primary)
        return (
            gtir2.Assign(left=left, right=right),
            ()
            if left.name in tbl
            else (
                gtir2.Field(
                    name=left.name,
                    location_type=primary.location_type,
                    dtype=_deduce_expr_type(right, tbl),
                ),
            ),
        )

    def visit_With(self, src: gtscript_ast2.With, tbl):
        loop_order = None
        location = None
        for item in src.items:
            assert isinstance(item.expr, gtscript_ast2.Call)
            f = item.expr
            if f.func.name == "computation":
                assert len(f.args) == 1
                assert isinstance(f.args[0], gtscript_ast2.Name)
                assert loop_order is None
                loop_order = common.LoopOrder[f.args[0].name]
            elif f.func.name == "location":
                assert len(f.args) == 1
                assert isinstance(f.args[0], gtscript_ast2.Name)
                assert location is None
                assert item.var is not None
                location = gtir2.PrimaryLocation(
                    name=item.var.name, location_type=common.LocationType[f.args[0].name]
                )
            elif f.func.name == "interval":
                pass
            else:
                raise RuntimeError(f"unexpected withitem: {item}")
        assert all(isinstance(s, gtscript_ast2.Assign) for s in src.body)

        def folder(acc, stmt):
            assign, temporaries = self.visit(
                stmt, tbl={**tbl, **{t.name: t for t in acc[1]}}, primary=location
            )
            return acc[0] + (assign,), acc[1] + temporaries

        body, temporaries = functools.reduce(folder, src.body, ((), ()))
        return gtir2.Stencil(loop_order=loop_order, location=location, body=body), temporaries


def _transform(src: gtscript_ast2.Function, fun_params):
    connectivity_params = tuple(p for p in fun_params if _is_conncetivity_annotation(p.annotation))
    connectivity_type_to_name = dict((p.annotation, p.name) for p in connectivity_params)
    if len(connectivity_params) > len(connectivity_type_to_name):
        raise RuntimeError(
            "the types of the conncetivities within computation should be all different"
        )
    connectivities = tuple(_extract_connectivity(p) for p in connectivity_params)
    args = tuple(
        _extract_arg(p, connectivity_type_to_name)
        for p in fun_params
        if not _is_conncetivity_annotation(p.annotation)
    )
    assert all(isinstance(s, gtscript_ast2.With) for s in src.body)
    tbl = {e.name: e for e in connectivities + args}

    def folder(acc, stmt):
        stencil, temporaries = _Visitor().visit(stmt, tbl={**tbl, **{t.name: t for t in acc[1]}})
        return acc[0] + (stencil,), acc[1] + temporaries

    stencils, temporaries = functools.reduce(folder, src.body, ((), ()))
    return gtir2.Computation(
        name=src.name,
        connectivities=connectivities,
        args=args,
        temporaries=temporaries,
        stencils=stencils,
    )


def transform(src):
    return _transform(py_to_gtscript2.transform(src), _extract_params(src))
