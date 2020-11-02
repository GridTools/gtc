from eve import Node, Str
from pydantic import validator
from typing import List


class Expr(Node):
    pass


class Literal(Expr):
    value: Str


class BinaryOp(Expr):
    left: Expr
    right: Expr
    op: Str


# TODO your IR here

from eve.codegen import FormatTemplate, TemplatedGenerator
from eve.codegen import MakoTemplate

# TODO your code generator here


class LIR_to_cpp(TemplatedGenerator):
    Literal = FormatTemplate("{value}")
    BinaryOp = FormatTemplate("({left}{op}{right})")
