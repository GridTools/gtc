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

import eve
from gtc.unstructured import gtir2
from gtc.unstructured.gtir_passes2 import rename_location


def _merge(lhs, rhs):
    t = rename_location.transform(rhs.location.name, lhs.location.name)
    return gtir2.Stencil(
        loop_order=lhs.loop_order,
        location=lhs.location,
        body=lhs.body + tuple(t(e) for e in rhs.body),
    )


def _folder(body, cur):
    if len(body) == 0:
        return (cur,)
    last = body[-1]
    return (
        body[0:-1] + (_merge(last, cur),)
        if last.loop_order == cur.loop_order
        and last.location.location_type == cur.location.location_type
        else body + (cur,)
    )


class _Visitor(eve.NodeVisitor):
    def visit_Computation(self, src: gtir2.Computation):
        return gtir2.Computation(
            name=src.name,
            connectivities=src.connectivities,
            args=src.args,
            temporaries=src.temporaries,
            stencils=functools.reduce(_folder, src.stencils, ()),
        )


transform = _Visitor().visit
