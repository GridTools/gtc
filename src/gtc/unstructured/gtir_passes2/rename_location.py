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

import eve
from gtc.unstructured import gtir2


class _Translator(eve.NodeTranslator):
    def visit_FieldAccess(self, src: gtir2.FieldAccess, renamer):
        return gtir2.FieldAccess(name=src.name, location=renamer(src.location))

    def visit_SparseFieldAccess(self, src: gtir2.SparseFieldAccess, renamer):
        return gtir2.SparseFieldAccess(
            name=src.name, primary=renamer(src.primary), secondary=renamer(src.secondary)
        )

    def visit_SecondaryLocation(self, src: gtir2.SecondaryLocation, renamer):
        return gtir2.SecondaryLocation(
            name=src.name, connectivity=src.connectivity, primary=renamer(src.primary),
        )

    def visit_NeighborReduce(self, src: gtir2.NeighborReduce, **kwarg):
        return gtir2.NeighborReduce(
            op=src.op,
            dtype=src.dtype,
            location=self.visit(src.location, **kwarg),
            body=self.visit(src.body, **kwarg),
        )

    def visit_BinaryOp(self, src: gtir2.BinaryOp, **kwarg):
        return gtir2.BinaryOp(
            op=src.op, left=self.visit(src.left, **kwarg), right=self.visit(src.right, **kwarg)
        )

    def visit_Assign(self, src: gtir2.Assign, **kwarg):
        return gtir2.Assign(
            left=self.visit(src.left, **kwarg), right=self.visit(src.right, **kwarg)
        )


def transform(old, new):
    return (
        lambda x: x
        if old == new
        else lambda src: _Translator().visit(src, lambda x: new if x == old else x)
    )
