# -*- coding: utf-8 -*-
from typing import List

from eve import Node, Str


class Expr(Node):
    pass


class Literal(Expr):
    value: Str


class BinaryOp(Expr):
    left: Expr
    right: Expr
    op: Str


class Offset(Node):
    i: int
    j: int

    @classmethod
    def zero(cls):
        return cls(i=0, j=0)


class FieldAccess(Expr):
    name: Str
    offset: Offset


class Stmt(Node):
    pass


class AssignStmt(Stmt):
    left: FieldAccess
    right: Expr


class FieldParam(Node):
    name: Str


class Stencil(Node):
    name: Str
    params: List[FieldParam]
    body: List[AssignStmt]
