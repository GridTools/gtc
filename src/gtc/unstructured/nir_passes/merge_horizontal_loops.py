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

import copy
from typing import List

import networkx as nx

import eve  # noqa: F401
from eve import Node, NodeTranslator
from gtc.unstructured import nir
from gtc.unstructured.nir_passes.field_dependency_graph import generate_dependency_graph


# This is an example of an analysis pass using data annotations.


def _has_read_with_offset_after_write(graph: nx.DiGraph, **kwargs):
    return any(edge["extent"] for _, _, edge in graph.edges(data=True))


def _find_merge_candidates(root: Node):
    """Find horizontal loop merge candidates.

    Result is a List[List[int]], where the inner list contains a range of the horizontal loop indices.
    The result is stored as a node data annotation `merge_candidates_` on the VerticalLoop.
    Currently the merge sets are ordered and disjunct, see question below.

    In the following examples A, B, C, ... are loops

    TODO Question
    Should we report all possible merge candidates, example: A, B, C
     - A + B and B + C possible, but not A + B + C  -> want both candidates (currently we only return [A,B])
     - A + B + C possible, we only want A + B + C, but not A + B and B + C as candidates

    Candidates are selected as follows:
     - Different location types cannot be fused
     - Only adjacent loops are considered
       Example: if A, C can be fused but an independent B which cannot be fused (e.g. different location) is in the middle,
                we don't consider A + C for fusion
     - Read after write access
        - if the read is without offset, we can fuse
        - if the read is with offset, we cannot fuse
    """
    vertical_loops = eve.FindNodes().by_type(nir.VerticalLoop, root)
    for vloop in vertical_loops:
        candidates = []
        candidate: List[nir.HorizontalLoop] = []
        candidate_range: List[int] = [0, 0]

        for index, hloop in enumerate(vloop.horizontal_loops):
            if len(candidate) == 0:
                candidate.append(hloop)
                candidate_range[0] = index
                continue
            elif (
                candidate[-1].location_type == hloop.location_type
            ):  # same location type as previous
                dependencies = generate_dependency_graph(candidate + [hloop])
                if not _has_read_with_offset_after_write(dependencies):
                    candidate.append(hloop)
                    candidate_range[1] = index
                    continue
            # cannot merge to previous loop:
            if len(candidate) > 1:
                candidates.append(candidate_range)  # add a new merge set
            candidate = [hloop]
            candidate_range = [index, 0]

        if len(candidate) > 1:
            candidates.append(candidate_range)  # add a new merge set

        vloop.merge_candidates_ = candidates


class MergeHorizontalLoops(NodeTranslator):
    """"""

    @classmethod
    def apply(cls, root: Node, **kwargs):
        """"""
        return cls().visit(root)

    def visit_VerticalLoop(self, node: nir.VerticalLoop, **kwargs):
        new_horizontal_loops = copy.deepcopy(node.horizontal_loops)
        for merge_group in node.merge_candidates_:
            declarations = []
            statements = []
            location_type = node.horizontal_loops[merge_group[0]].location_type

            for loop in node.horizontal_loops[merge_group[0] : merge_group[1] + 1]:
                declarations += loop.stmt.declarations
                statements += loop.stmt.statements

            new_horizontal_loops = [  # noqa: E203
                nir.HorizontalLoop(
                    stmt=nir.BlockStmt(
                        declarations=declarations,
                        statements=statements,
                        location_type=location_type,
                    ),
                    location_type=location_type,
                )
            ]
        return nir.VerticalLoop(loop_order=node.loop_order, horizontal_loops=new_horizontal_loops)


def merge_horizontal_loops(root: Node):
    return MergeHorizontalLoops.apply(root)


def find_and_merge_horizontal_loops(root: Node):
    _find_merge_candidates(root)
    return merge_horizontal_loops(root)
