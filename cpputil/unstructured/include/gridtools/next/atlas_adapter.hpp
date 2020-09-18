#pragma once

#include <cassert>

#include <atlas/mesh.h>
#include <mesh/Connectivity.h>

#include <gridtools/common/array.hpp>
#include <gridtools/sid/rename_dimensions.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>

#include "unstructured.hpp"

#ifdef __CUDACC__ // TODO proper handling
#include <gridtools/storage/gpu.hpp>
using storage_trait = gridtools::storage::gpu;
#else
#include <gridtools/storage/cpu_ifirst.hpp>
using storage_trait = gridtools::storage::cpu_ifirst;
#endif

namespace atlas {
    namespace gridtools_next_impl_ {
        using namespace gridtools;
        using namespace next;

        struct edge_vertex_neighbors {
            int m_val[2];
        };

        template <class Fun>
        GT_FUNCTION void mesh_for_each_neighbor(edge_vertex_neighbors obj, Fun &&fun) {
            fun(obj.m_val[0]);
            fun(obj.m_val[1]);
        }

        inline auto get_edge_vertex_neighbor_table(Mesh const &mesh) {
            auto &&src = mesh.edges().node_connectivity();
            auto init = [&](auto row) -> edge_vertex_neighbors {
                assert(src.cols(row) == 2);
                return {src.row(row)(0), src.row(row)(1)};
            };
            return sid::rename_numbered_dimensions<::gridtools::next::edge>(
                storage::builder<storage_trait>.template type<edge_vertex_neighbors>().dimensions(src.rows()).initializer(init)());
        }

        struct vertex_edge_neighbor {
            int index;
            int sign;
        };

        GT_FUNCTION int mesh_get_neighbor_index(vertex_edge_neighbor src) { return src.index; }
        GT_FUNCTION int mesh_get_neighbor_sign(vertex_edge_neighbor src) { return src.sign; }

        using vertex_edge_neighbors = ::gridtools::array<vertex_edge_neighbor, 7>;

        template <class Fun>
        GT_FUNCTION void mesh_for_each_neighbor(vertex_edge_neighbors const &obj, Fun &&fun) {
            for (auto &&item : obj) {
                if (item.index < 0)
                    continue;
                fun(item);
            }
        }

        inline auto get_vertex_edge_neighbor_table(Mesh const &mesh) {
            auto &&src = mesh.nodes().edge_connectivity();
            auto &&e2v = mesh.edges().node_connectivity();
            assert(src.missing_value() == -1);
            assert(src.maxcols() <= 7);
            auto init = [&, is_pole_edge = [edge_flags = array::make_view<int, 1>(mesh.edges().flags())](auto e) {
                using topology_t = atlas::mesh::Nodes::Topology;
                return topology_t::check(edge_flags(e), topology_t::POLE);
            }](auto row) -> vertex_edge_neighbors {
                vertex_edge_neighbors res;
                for (int i = 0; i < 7; ++i) {
                    if (i >= src.cols(row)) {
                        res[i] = {-1, 1};
                        continue;
                    }
                    auto index = src(row, i);
                    res[i] = {index, row == e2v(index, 0) || is_pole_edge(index) ? 1 : -1};
                }
                return res;
            };
            return sid::rename_numbered_dimensions<::gridtools::next::vertex>(
                storage::builder<storage_trait>.template type<vertex_edge_neighbors>().dimensions(src.rows()).initializer(init)());
        }

    } // namespace gridtools_next_impl_

    inline auto mesh_get_neighbor_table(Mesh const &mesh, ::gridtools::next::edge, ::gridtools::next::vertex) {
        return gridtools_next_impl_::get_edge_vertex_neighbor_table(mesh);
    }

    inline auto mesh_get_neighbor_table(Mesh const &mesh, ::gridtools::next::vertex, ::gridtools::next::edge) {
        return gridtools_next_impl_::get_vertex_edge_neighbor_table(mesh);
    }

    inline auto mesh_get_location_size(Mesh const &mesh, ::gridtools::next::edge) { return mesh.edges().size(); }

    inline auto mesh_get_location_size(Mesh const &mesh, ::gridtools::next::vertex) { return mesh.nodes().size(); }
} // namespace atlas
