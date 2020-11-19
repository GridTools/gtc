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

"""Definitions of specific Eve exceptions."""

from .typingx import Any, Dict


class EveError:
    """Base class for Eve-specific exceptions.

    Notes:
        This base class has to be always inherited together with a standard
    exception.

    """

    message_template = "Generic Eve error (info: {info}) "
    info: Dict[str, Any]

    def __init__(self, message: str = None, **kwargs: Any) -> None:
        super().__init__(  # type: ignore  # when inherited together with another exception the super() call works
            message or type(self).message_template.format(**self.info, info=self.info)
        )
        self.info = kwargs


class EveTypeError(EveError, TypeError):
    message_template = "Invalid or unexpected type (info: {info})"


class EveValueError(EveError, ValueError):
    message_template = "Invalid value (info: {info})"


class EveRuntimeError(EveError, RuntimeError):
    message_template = "Runtime error (info: {info})"
