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


"""Global configuration of test generation and execution with pytest."""


import pytest

from . import common


NODE_MAKERS = []
INVALID_NODE_MAKERS = []


# Automatic creation of pytest fixtures from maker functions in .common module
for key, value in common.__dict__.items():
    if key.startswith("make_"):
        name = key[5:]
        exec(
            f"""
@pytest.fixture
def {name}_maker():
    return common.make_{name}

@pytest.fixture
def {name}({name}_maker):
    return {name}_maker()
"""
        )

        if "node" in key:
            if "invalid" in key:
                INVALID_NODE_MAKERS.append(value)
            else:
                NODE_MAKERS.append(value)


@pytest.fixture
def sample_dialect():
    return common.TestDialect


@pytest.fixture
def sample_vtype():
    return common.SimpleVType


@pytest.fixture(params=NODE_MAKERS)
def sample_node_maker(request):
    return request.param


@pytest.fixture(params=NODE_MAKERS)
def sample_node(request):
    return request.param()


@pytest.fixture(params=INVALID_NODE_MAKERS)
def invalid_sample_node_maker(request):
    return request.param
