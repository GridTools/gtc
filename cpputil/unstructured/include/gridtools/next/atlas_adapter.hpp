#pragma once

#include <cassert>
#include <cstddef>

#include <atlas/mesh.h>
#include <mesh/Connectivity.h>

#include <gridtools/common/hymap.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/meta.hpp>
#include <gridtools/next/atlas_field_util.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/sid/rename_dimensions.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>

#include "gridtools/meta/st_contains.hpp"
#include "unstructured.hpp"

#ifdef __CUDACC__ // TODO proper handling
#include <gridtools/storage/gpu.hpp>
using storage_trait = gridtools::storage::gpu;
#else
#include <gridtools/storage/cpu_ifirst.hpp>
using storage_trait = gridtools::storage::cpu_ifirst;
#endif

namespace gridtools::next::atlas_wrappers {
    // not really a connectivity
    struct primary_connectivity {
        std::size_t size_;

        friend std::size_t connectivity_size(primary_connectivity const &conn) { return conn.size_; }
    };

    template <class Table>
    struct regular_connectivity {
        static_assert(is_sid<Table>());

        Table table_;

        friend integral_constant<int, -1> connectivity_skip_value(regular_connectivity const &) { return {}; }
        friend Table const &connectivity_neighbor_table(regular_connectivity const &obj) { return obj.table_; }
    };

    template <class Table>
    auto connectivity_size(regular_connectivity<Table> const &obj) {
        return tuple_util::get<0>(sid::get_upper_bounds(obj.table_));
    }

    template <class Table>
    auto connectivity_max_neighbors(regular_connectivity<Table> const &obj) {
        return sid::get_upper_bound<neighbor>(sid::get_upper_bounds(obj.table_));
    }

    template <class Src, class Location, class MaxNeighbors>
    auto make_regular_connectivity(Src const &src, Location, MaxNeighbors max_neighbors) {
        assert(src.missing_value() == -1);
        auto init = [&](auto row, auto col) { return col < src.cols(row) ? src.row(row)(col) : -1; };
        auto ds =
            storage::builder<storage_trait>.template type<int>().dimensions(src.rows(), max_neighbors).initializer(init)();
        auto table = sid::rename_numbered_dimensions<Location, neighbor>(std::move(ds));
        using table_t = decltype(table);
        return regular_connectivity<table_t>{std::move(table)};
    }
} // namespace gridtools::next::atlas_wrappers

namespace atlas {
    namespace gridtools_next_impl_ {
        using namespace gridtools;
        using namespace next;

        // not really a connectivity
        struct primary_connectivity {
            std::size_t size_;

            friend std::size_t connectivity_size(primary_connectivity const &conn) { return conn.size_; }
        };

        template <class Table>
        struct regular_connectivity {
            static_assert(is_sid<Table>());

            Table table_;

            friend integral_constant<int, -1> connectivity_skip_value(regular_connectivity const &) { return {}; }
            friend Table const &connectivity_neighbor_table(regular_connectivity const &obj) { return obj.table_; }
        };

        template <class Table>
        auto connectivity_size(regular_connectivity<Table> const &obj) {
            return tuple_util::get<0>(sid::get_upper_bounds(obj.table_));
        }

        template <class Table>
        auto connectivity_max_neighbors(regular_connectivity<Table> const &obj) {
            return sid::get_upper_bound<neighbor>(sid::get_upper_bounds(obj.table_));
        }

        template <class Src, class Location, class MaxNeighbors>
        auto make_regular_connectivity(Src const &src, Location, MaxNeighbors max_neighbors) {
            assert(src.missing_value() == -1);
            auto init = [&](auto row, auto col) { return col < src.cols(row) ? src.row(row)(col) : -1; };
            auto ds =
                storage::builder<storage_trait>.template type<int>().dimensions(src.rows(), max_neighbors).initializer(init)();
            auto table = sid::rename_numbered_dimensions<Location, neighbor>(std::move(ds));
            using table_t = decltype(table);
            return regular_connectivity<table_t>{std::move(table)};
        }
    } // namespace gridtools_next_impl_

    inline auto mesh_connectivity(const Mesh &mesh, ::gridtools::next::vertex from, gridtools::next::edge) {
        using namespace gridtools::literals;
        // TODO this number must passed by the user (probably wrap atlas mesh)
        return gridtools_next_impl_::make_regular_connectivity(mesh.nodes().edge_connectivity(), from, 7_c);
    }

    inline auto mesh_connectivity(Mesh const &mesh, ::gridtools::next::edge from, ::gridtools::next::vertex) {
        using namespace gridtools::literals;
        return gridtools_next_impl_::make_regular_connectivity(mesh.edges().node_connectivity(), from, 2_c);
    }

    inline auto mesh_connectivity(Mesh const &mesh, ::gridtools::next::edge) {
        return gridtools_next_impl_::primary_connectivity{std::size_t(mesh.edges().size())};
    }

    inline auto mesh_connectivity(Mesh const &mesh, ::gridtools::next::vertex) {
        return gridtools_next_impl_::primary_connectivity{std::size_t(mesh.nodes().size())};
    }
} // namespace atlas
