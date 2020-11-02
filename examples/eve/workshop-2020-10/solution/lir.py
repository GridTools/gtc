from eve import Node, Str
from pydantic import validator
from typing import List


class Expr(Node):
    pass


class Literal:
    value: int


class FunctionName(Expr):
    value: Str


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


class AssignStmt(Node):
    left: FieldAccess
    right: Expr

    @validator("left")
    def no_offset_in_assignment_lhs(cls, v):
        if v.offset.i != 0 or v.offset.j != 0:
            raise ValueError("Lhs of assignment must not have an offset")
        return v


class FieldDecl(Node):
    name: Str


# relative to domain
class Indent(Node):
    left: int
    right: int

    @classmethod
    def zero(cls):
        return cls(left=0, right=0)


class HorizontalLoop(Node):
    i_indent: Indent
    j_indent: Indent
    body: List[AssignStmt]


class Fun(Node):
    name: Str
    params: List[FieldDecl]
    horizontal_loops: List[HorizontalLoop]


from eve.codegen import FormatTemplate, TemplatedGenerator
from eve.codegen import MakoTemplate


class LIR_to_cpp(TemplatedGenerator):
    Literal = FormatTemplate("{value}")
    BinaryOp = FormatTemplate("({left}{op}{right})")

    Offset = FormatTemplate("[i+{i}][j+{j}]")

    FieldAccess = FormatTemplate("{name}{offset}")

    AssignStmt = FormatTemplate("{left} = {right};")

    FieldDecl = FormatTemplate("Field& {name}")

    # TODO use visit for Indent?

    # Using domain and i, j by hard-coded string is not a good design!
    HorizontalLoop = MakoTemplate(
        """for(std::size_t i = ${_this_node.i_indent.left}; i < domain[0] - ${_this_node.i_indent.right}; ++i) {
            for(std::size_t j = ${_this_node.j_indent.left}; j < domain[1] - ${_this_node.j_indent.right}; ++j) {
                ${''.join(body)}
            }
        }"""
    )

    Fun = MakoTemplate(
        """void ${name}(Domain domain, ${','.join(params)}){
        ${''.join(horizontal_loops)}
        }"""
    )
