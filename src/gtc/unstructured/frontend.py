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


from toolz.functoolz import compose

from gtc.unstructured import gtir_to_usid2, py_to_gtir2, usid2_codegen
from gtc.unstructured.gtir_passes2 import merge_stencils


def _impl(codegen):
    return compose(
        codegen, gtir_to_usid2.transform, merge_stencils.transform, py_to_gtir2.transform
    )


naive = _impl(usid2_codegen.naive)
gpu = _impl(usid2_codegen.gpu)
