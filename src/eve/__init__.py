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

"""Eve: a stencil toolchain in pure Python."""

# flake8: noqa  # disable flake8 because of non-used imports warnings
# Disable isort to avoid circular imports
from .version import __version__, __versioninfo__  # isort:skip

from . import _typing  # isort:skip
from . import exceptions, type_definitions  # isort:skip
from . import utils  # isort:skip
from . import concepts  # isort:skip
from . import iterators, traits, visitors  # isort:skip
from . import codegen, tree_utils  # isort:skip


from .concepts import (
    FieldKind,
    FrozenModel,
    FrozenNode,
    Model,
    Node,
    SourceLocation,
    VType,
    field,
    in_field,
    out_field,
)
from .iterators import traverse_tree
from .traits import SymbolTableTrait
from .tree_utils import FindNodes
from .type_definitions import (
    DELETE,
    NOTHING,
    Bool,
    Bytes,
    Enum,
    Float,
    Int,
    IntEnum,
    NegativeFloat,
    NegativeInt,
    PositiveFloat,
    PositiveInt,
    Str,
    StrEnum,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    SymbolName,
)
from .visitors import NodeModifier, NodeTranslator, NodeVisitor
