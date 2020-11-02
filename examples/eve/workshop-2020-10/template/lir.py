# -*- coding: utf-8 -*-
from typing import List  # noqa

from pydantic import validator  # noqa

from eve import Node, Str
from eve.codegen import FormatTemplate, MakoTemplate, TemplatedGenerator  # noqa


class Expr(Node):
    pass


class Literal(Expr):
    value: Str


class BinaryOp(Expr):
    left: Expr
    right: Expr
    op: Str


# TODO your IR here


# TODO your code generator here


class LIR_to_cpp(TemplatedGenerator):
    Literal = FormatTemplate("{value}")
    BinaryOp = FormatTemplate("({left}{op}{right})")
