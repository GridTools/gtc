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

from eve import codegen
from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from gtc.unstructured import usid2


class _KernelCallGenerator(codegen.TemplatedGenerator):
    Composite = as_mako(
        "make_composite<${','.join(f'{i}_tag' for i in items)}>(${','.join(items)})"
    )
    Kernel = as_mako(
        "call_kernel<${id_}>(d.${location_type}${''.join(f', {c}' for c in [primary] + secondaries)});"
    )


class _Generator(codegen.TemplatedGenerator):
    def visit_Computation(self, node: usid2.Computation, **kwargs):
        return self.generic_visit(
            node, kernel_calls=tuple(_KernelCallGenerator.apply(k) for k in node.kernels), **kwargs
        )

    Literal = as_mako("${value}")
    BinaryOp = as_fmt("({left} {op} {right})")
    FieldAccess = as_fmt("field<{name}_tag>({location})")
    NeighborReduce = as_mako(
        "${op}_neighbors<${dtype}, ${connectivity}_tag>"
        + "([](auto &&${primary}, auto &&${secondary}) { return ${body}; }, ${primary}, strides, ${connectivity})"
    )
    Assign = as_fmt("{left} = {right};")
    Kernel = as_mako(
        """
        struct ${id_} {
            GT_FUNCTION auto operator()() const {
                return [](auto && ${_this_node.primary.name},
                    auto &&strides${''.join(f', auto&& {s.name}' for s in _this_node.secondaries)}) {
                    ${''.join(body)}
                };
            }
        };
        """
    )
    Temporary = as_fmt(
        "auto {name} = make_simple_tmp_storage<{dtype}>(d.{location_type}, d.k, alloc);"
    )
    Connectivity = as_mako(
        "struct ${name}_tag: connectivity<${max_neighbors}, ${has_skip_values.lower()}> {};"
    )
    Field = as_mako("struct ${name}_tag {};")
    SparseField = as_mako("struct ${name}_tag: sparse_field<${connectivity}_tag> {};")
    Computation = as_mako(
        """<%

            ts = tuple(e.name for e in _this_node.temporaries)
            cs = tuple(e.name for e in _this_node.connectivities)
            ps = tuple(e.name for e in _this_node.args)

        %>#pragma once
        #include <gridtools/usid/${backend}_helpers.hpp>
        namespace gridtools::usid::${backend}::${name}_impl_ {
        ${''.join(connectivities)}
        ${''.join(args)}
        ${''.join(f'struct {t}_tag {{}};' for t in ts)}
        ${''.join(kernels)}
        inline constexpr auto ${name} = [](domain d${''.join(f', auto&& {c}' for c in cs)}) {
            ${''.join(f'static_assert(is_sid<decltype({c}(traits_t()))>());' for c in cs)}
            return[d = std::move(d)
                ${ ''.join(f', {c} = sid::rename_dimensions<dim::n, {c}_tag>(std::forward<decltype({c})>({c})(traits_t()))' for c in cs) }]
                (${ ','.join(f'auto&& {p}' for p in ps)}) {
                ${''.join(f'static_assert(is_sid<decltype({p})>());' for p in ps)}
        % if len(temporaries) > 0:
                auto alloc = make_allocator();
                ${''.join(temporaries)}
        % endif
                ${''.join(kernel_calls)}
            };
        };
        }
        using gridtools::usid::${backend}::${name}_impl_::${name};
        """
    )


def _impl(backend):
    # TOOO(anstaf): agree on python style here
    return compose(
        lambda x: codegen.format_source("cpp", x, style="LLVM"),
        lambda x: _Generator.apply(x, backend=backend),
    )


naive = _impl("naive")
gpu = _impl("gpu")
