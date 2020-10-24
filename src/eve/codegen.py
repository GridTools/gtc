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

"""Tools for source code generation."""


from __future__ import annotations

import abc
import collections.abc
import contextlib
import os
import string
import sys
import textwrap
import types
from subprocess import PIPE, Popen

import black
import jinja2
from mako import template as mako_tpl

from . import type_definitions, utils
from ._typing import (
    Any,
    Callable,
    ClassVar,
    Collection,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)
from .concepts import Node, TreeNode
from .visitors import NodeVisitor


try:
    import clang_format

    _CLANG_FORMAT_AVAILABLE = True
    del clang_format
except ImportError:
    _CLANG_FORMAT_AVAILABLE = False


SourceFormatter = Callable[[str], str]

#: Global dict storing registered formatters
SOURCE_FORMATTERS: Dict[str, SourceFormatter] = {}


def register_formatter(language: str) -> Callable[[SourceFormatter], SourceFormatter]:
    """Decorator to register source code formatters for specific languages."""

    def _decorator(formatter: SourceFormatter) -> SourceFormatter:
        if language in SOURCE_FORMATTERS:
            raise ValueError(f"Another formatter for language '{language}' already exists")

        assert callable(formatter)
        SOURCE_FORMATTERS[language] = formatter

        return formatter

    return _decorator


@register_formatter("python")
def format_python_source(
    source: str,
    *,
    line_length: int = 100,
    target_versions: Optional[Set[str]] = None,
    string_normalization: bool = True,
) -> str:
    """Format Python source code using black formatter."""

    target_versions = target_versions or f"{sys.version_info.major}{sys.version_info.minor}"
    target_versions = set(black.TargetVersion[f"PY{v.replace('.', '')}"] for v in target_versions)

    formatted_source = black.format_str(
        source,
        mode=black.FileMode(
            line_length=line_length,
            target_versions=target_versions,
            string_normalization=string_normalization,
        ),
    )
    assert isinstance(formatted_source, str)

    return formatted_source


if _CLANG_FORMAT_AVAILABLE:

    @register_formatter("cpp")
    def format_cpp_source(
        source: str,
        *,
        style: Optional[str] = None,
        fallback_style: Optional[str] = None,
        sort_includes: bool = False,
    ) -> str:
        """Format C++ source code using clang-format."""

        args = ["clang-format"]
        if style:
            args.append(f"--style={style}")
        if fallback_style:
            args.append(f"--fallback-style={style}")
        if sort_includes:
            args.append("--sort-includes")

        p = Popen(args, stdout=PIPE, stdin=PIPE, encoding="utf8")
        formatted_source, _ = p.communicate(input=source)
        assert isinstance(formatted_source, str)

        return formatted_source


def format_source(language: str, source: str, *, skip_errors: bool = True, **kwargs: Any) -> str:
    """Format source code if a formatter exists for the specific language."""

    formatter = SOURCE_FORMATTERS.get(language, None)
    try:
        if formatter:
            return formatter(source, **kwargs)  # type: ignore # Callable does not support **kwargs
        else:
            raise RuntimeError(f"Missing formatter for '{language}' language")
    except Exception as e:
        if skip_errors:
            return source
        else:
            raise RuntimeError(
                f"Something went wrong when trying to format '{language}' source code"
            ) from e


class Name:
    """Text formatter with different case styles for symbol names in source code."""

    words: List[str]

    @classmethod
    def from_string(cls, name: str, case_style: utils.CaseStyleConverter.CASE_STYLE) -> Name:
        return cls(utils.CaseStyleConverter.split(name, case_style))

    def __init__(self, words: utils.AnyWordsIterable) -> None:
        if isinstance(words, str):
            words = [words]
        if not isinstance(words, collections.abc.Iterable):
            raise TypeError(
                f"Identifier definition ('{words}') type is not a valid sequence of words"
            )

        words = [*words]
        if not all(isinstance(item, str) for item in words):
            raise TypeError(
                f"Identifier definition ('{words}') type is not a valid sequence of words"
            )

        self.words = words

    def as_case(self, case_style: utils.CaseStyleConverter.CASE_STYLE) -> str:
        return utils.CaseStyleConverter.join(self.words, case_style)


AnyTextSequence = Union[Sequence[str], "TextBlock"]


class TextBlock:
    """A block of source code represented as a sequence of text lines.

    Check the provided context manager creator method (:meth:`indented`)
    for simple `indent - append - dedent` workflows.

    Args:
        indent_level: Initial indentation level.
        indent_size: Number of characters per indentation level.
        indent_char: Character used in the indentation.
        end_line: Character or string used as new-line separator.

    """

    def __init__(
        self,
        *,
        indent_level: int = 0,
        indent_size: int = 4,
        indent_char: str = " ",
        end_line: str = "\n",
    ) -> None:
        if not isinstance(indent_char, str) or len(indent_char) != 1:
            raise ValueError("'indent_char' must be a single-character string")
        if not isinstance(end_line, str):
            raise ValueError("'end_line' must be a string")

        self.indent_level = indent_level
        self.indent_size = indent_size
        self.indent_char = indent_char
        self.end_line = end_line
        self.lines: List[str] = []

    def append(self, new_line: str, *, update_indent: int = 0) -> TextBlock:
        if update_indent > 0:
            self.indent(update_indent)
        elif update_indent < 0:
            self.dedent(-update_indent)

        self.lines.append(self.indent_str + new_line)

        return self

    def extend(self, new_lines: AnyTextSequence, *, dedent: bool = False) -> TextBlock:
        assert isinstance(new_lines, (collections.abc.Sequence, TextBlock))

        if dedent:
            if isinstance(new_lines, TextBlock):
                new_lines = textwrap.dedent(new_lines.text).splitlines()
            else:
                new_lines = textwrap.dedent("\n".join(new_lines)).splitlines()

        elif isinstance(new_lines, TextBlock):
            new_lines = new_lines.lines

        for line in new_lines:
            self.append(line)

        return self

    def empty_line(self, count: int = 1) -> TextBlock:
        self.lines.extend([""] * count)
        return self

    def indent(self, steps: int = 1) -> TextBlock:
        self.indent_level += steps
        return self

    def dedent(self, steps: int = 1) -> TextBlock:
        assert self.indent_level >= steps
        self.indent_level -= steps
        return self

    @contextlib.contextmanager
    def indented(self, steps: int = 1) -> Iterator[TextBlock]:
        """Context manager creator for temporary indentation of sources.

        This context manager simplifies the usage of indent/dedent in
        common `indent - append - dedent` workflows.

        Examples:
            >>> block = TextBlock();
            >>> block.append('first line')  # doctest: +ELLIPSIS
            <...>
            >>> with block.indented():
            ...     block.append('second line');  # doctest: +ELLIPSIS
            <...>
            >>> block.append('third line')  # doctest: +ELLIPSIS
            <...>
            >>> print(block.text)
            first line
                second line
            third line

        """
        self.indent(steps)
        yield self
        self.dedent(steps)

    @property
    def text(self) -> str:
        """Single string with the whole block contents."""
        lines = ["".join([str(item) for item in line]) for line in self.lines]
        return self.end_line.join(lines)

    @property
    def indent_str(self) -> str:
        """Indentation string for new lines (in the current state)."""
        return self.indent_char * (self.indent_level * self.indent_size)

    def __iadd__(self, source_line: Union[str, AnyTextSequence]) -> TextBlock:
        if isinstance(source_line, str):
            return self.append(source_line)
        else:
            return self.extend(source_line)

    def __len__(self) -> int:
        return len(self.lines)

    def __str__(self) -> str:
        return self.text


TemplateT = TypeVar("TemplateT", bound="Template")


class Template(abc.ABC):
    """Abstract class defining the Template interface.

    Subclassess adapting this interface to different template engines will
    only need to implement the abstract methods.
    """

    @classmethod
    def from_file(cls: Type[TemplateT], file_path: Union[str, os.PathLike]) -> Template:
        if cls is Template:
            raise RuntimeError("This method can only be called in concrete Template subclasses")

        with open(file_path, "r") as f:
            definition = f.read()
        return cls(definition)

    def render(self, mapping: Optional[Mapping[str, str]] = None, **kwargs: Any) -> str:
        """Render the template.

        Args:
            mapping: A mapping whose keys match the template placeholders.
            **kwargs: placeholder values might be also provided as
                keyword arguments, and they will take precedence over ``mapping``
                values for duplicated keys.

        """
        if not mapping:
            mapping = {}
        if kwargs:
            mapping = {**mapping, **kwargs}

        return self.render_template(**mapping)

    @abc.abstractmethod
    def __init__(self, definition: Any, **kwargs: Any) -> None:
        pass

    @abc.abstractmethod
    def render_template(self, **kwargs: Any) -> str:
        pass


class FormatTemplate(Template):
    """Template adapter for :class:`StringFormatter`."""

    definition: str

    _formatter_: ClassVar[string.Formatter] = utils.XStringFormatter()

    def __init__(self, definition: str, **kwargs: Any) -> None:
        self.definition = definition

    def render_template(self, **kwargs: Any) -> str:
        return self._formatter_.format(self.definition, **kwargs)


class StringTemplate(Template):
    """Template adapter for `string.Template`."""

    definition: string.Template

    def __init__(self, definition: Union[str, string.Template], **kwargs: Any) -> None:
        if isinstance(definition, str):
            definition = string.Template(definition)
        assert isinstance(definition, string.Template)
        self.definition = definition

    def render_template(self, **kwargs: Any) -> str:
        return self.definition.substitute(**kwargs)


class JinjaTemplate(Template):
    """Template adapter for `jinja2.Template`."""

    definition: jinja2.Template

    def __init__(self, definition: Union[str, jinja2.Template], **kwargs: Any) -> None:
        if isinstance(definition, str):
            definition = jinja2.Template(definition)
        assert isinstance(definition, jinja2.Template)
        self.definition = definition

    def render_template(self, **kwargs: Any) -> str:
        return self.definition.render(**kwargs)


class MakoTemplate(Template):
    """Template adapter for `mako.template.Template`."""

    definition: mako_tpl.Template

    def __init__(self, definition: mako_tpl.Template, **kwargs: Any) -> None:
        if isinstance(definition, str):
            definition = mako_tpl.Template(definition)
        assert isinstance(definition, mako_tpl.Template)
        self.definition = definition

    def render_template(self, **kwargs: Any) -> str:
        result = self.definition.render(**kwargs)
        assert isinstance(result, str)
        return result


class TemplatedGenerator(NodeVisitor):
    """A code generator visitor using :class:`TextTemplate` s.

    The order followed to choose a `dump()` function for instances of
    :class:`eve.Node` is the following:

        1. A `self.visit_NODE_TYPE_NAME()` method where `NODE_TYPE_NAME`
            matches `NODE_CLASS.__name__`, and `NODE_CLASS` is the
            actual type of the node or any of its superclasses
            following MRO order.
        2. A `NODE_TYPE_NAME` class variable of type :class:`Template`,
            where `NODE_TYPE_NAME` matches `NODE_CLASS.__name__`, and
            `NODE_CLASS` is the actual type of the node or any of its
            superclasses following MRO order.

    When a template is used, the following keys will be passed to the template
    instance:

        * `**node_fields`: all the node children and attributes by name.
        * `_attrs`: a `dict` instance with the results of visiting all
            the node attributes.
        * `_children`: a `dict` instance with the results of visiting all
            the node children.
        * `_this_node`: the actual node instance (before visiting children).
        * `_this_generator`: the current generator instance.
        * `_this_module`: the generator's module instance .
        * `**kwargs`: the keyword arguments received by the visiting method.

    Class variable templates cannot be used for instances of other types
    (not :class:`eve.Node` subclasses). Step 2 will be therefore substituted
    by a call to the :meth:`self.generic_dump()` method.

    """

    _templates_: ClassVar[Mapping[str, Template]]

    @classmethod
    def __init_subclass__(cls, *, inherit_templates: bool = True, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)  # type: ignore  # mypy issues 4335, 4660
        if "_templates_" in cls.__dict__:
            raise TypeError(f"Invalid '_templates_' member in class {cls}")

        templates: Dict[str, Template] = {}
        if inherit_templates:
            for templated_gen_class in reversed(cls.__mro__[1:]):
                if (
                    issubclass(templated_gen_class, TemplatedGenerator)
                    and templated_gen_class is not TemplatedGenerator
                ):
                    templates.update(templated_gen_class._templates_)

        templates.update(
            {
                key: value
                for key, value in cls.__dict__.items()
                if isinstance(value, Template) and not key.startswith("_") and not key.endswith("_")
            }
        )

        cls._templates_ = types.MappingProxyType(templates)

    @classmethod
    def apply(cls, root: TreeNode, **kwargs: Any) -> Union[str, Collection[str]]:
        """Public method to build a class instance and visit an IR node.

        Args:
            root: An IR node.
            node_templates (optiona): see :class:`NodeDumper`.
            dump_function (optiona): see :class:`NodeDumper`.
            **kwargs (optional): custom extra parameters forwarded to
                `visit_NODE_TYPE_NAME()`.

        Returns:
            String (or collection of strings) with the dumped version of the root IR node.

        """
        return cast(Union[str, Collection[str]], cls().visit(root, **kwargs))

    @classmethod
    def generic_dump(cls, node: TreeNode, **kwargs: Any) -> str:
        """Class-specific ``dump()`` function for primitive types.

        This class could be redefined in the subclasses.
        """
        return str(node)

    def generic_visit(self, node: TreeNode, **kwargs: Any) -> Union[str, Collection[str]]:
        result: Union[str, Collection[str]] = ""
        if isinstance(node, Node):
            template, _ = self.get_template(node)
            if template:
                result = self.render_template(
                    template,
                    node,
                    self.transform_children(node, **kwargs),
                    self.transform_attrs(node, **kwargs),
                    **kwargs,
                )
        elif isinstance(node, (collections.abc.Sequence, collections.abc.Set)) and not isinstance(
            node, type_definitions.ATOMIC_COLLECTION_TYPES
        ):
            result = [self.visit(value, **kwargs) for value in node]
        elif isinstance(node, collections.abc.Mapping):
            result = {key: self.visit(value, **kwargs) for key, value in node.items()}
        else:
            result = self.generic_dump(node, **kwargs)

        return result

    def get_template(self, node: TreeNode) -> Tuple[Optional[Template], Optional[str]]:
        """Get a template for a node instance (see class documentation)."""
        template: Optional[Template] = None
        template_key: Optional[str] = None
        if isinstance(node, Node):
            for node_class in node.__class__.__mro__:
                template_key = node_class.__name__
                template = self._templates_.get(template_key, None)
                if template is not None or node_class is Node:
                    break

        return template, None if template is None else template_key

    def render_template(
        self,
        template: Template,
        node: Node,
        transformed_children: Mapping[str, Any],
        transformed_attrs: Mapping[str, Any],
        **kwargs: Any,
    ) -> str:
        """Render a template using node instance data (see class documentation)."""

        return template.render(
            **transformed_children,
            **transformed_attrs,
            _children=transformed_children,
            _attrs=transformed_attrs,
            _this_node=node,
            _this_generator=self,
            _this_module=sys.modules[type(self).__module__],
            **kwargs,
        )

    def transform_children(self, node: Node, **kwargs: Any) -> Dict[str, Any]:
        return {key: self.visit(value, **kwargs) for key, value in node.iter_children()}

    def transform_attrs(self, node: Node, **kwargs: Any) -> Dict[str, Any]:
        return {key: self.visit(value, **kwargs) for key, value in node.iter_attributes()}
