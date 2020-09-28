#pragma once

#include <cassert>
#include <type_traits>
#include <vector>

#include <gridtools/meta.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/sid/loop.hpp>
#include <gridtools/storage/builder.hpp>

#include "unstructured.hpp"

namespace gridtools {
    namespace next {
        namespace mesh {
            namespace concept_impl_ {
                template <class Location, class Mesh>
                auto get_location_number(Mesh const &mesh) -> decltype(mesh_get_location_number(mesh, Location())) {
                    return mesh_get_location_number(mesh, Location());
                }

                template <class Mesh, class Location, class = void>
                struct has_location : std::false_type {};

                template <class Mesh, class Location>
                struct has_location<Mesh,
                    Location,
                    void_t<decltype(mesh_get_location_number(std::decay_t<Mesh const &>(), Location()))>>
                    : std::true_type {};

                template <class Mesh>
                using locations =
                    meta::filter<meta::curry<has_location, Mesh>::template apply, meta::list<vertex, edge, cell>>;

                template <class From, class To, class... Tos>
                constexpr auto get_max_neighbors_number =
                    [](auto const &mesh) -> decltype(mesh_get_max_neighbors_number(mesh, From(), To(), Tos()...)) {
                    return mesh_get_max_neighbors_number(mesh, From(), To(), Tos()...);
                };

                template <class From, class To, class... Tos>
                constexpr auto has_skip_values =
                    [](auto const &mesh) -> decltype(mesh_has_skip_values(mesh, From(), To(), Tos()...)) {
                    return mesh_has_skip_values(mesh, From(), To(), Tos()...);
                };

                template <class Traits, class From, class To, class Src, class S, class N>
                auto invert_neighbor_table(Src &&src, S s, N n) {
                    std::vector<std::vector<int>> tbl(s);
                    auto &&u_bounds = sid::get_upper_bounds(src);
                    int i = 0;
                    sid::make_loop<dim::h>(sid::get_upper_bound<dim::h>(u_bounds))([&](auto &ptr, auto const &strides) {
                        sid::make_loop<neighbor<To>>(sid::get_upper_bound<neighbor<To>>(u_bounds))(
                            [&](auto ptr, auto &&) { tbl[*ptr].pop_back(i); })(ptr, strides);
                        ++i;
                    })(sid::get_origin(src)(), sid::get_strides(src));
                    for (auto &&neighbors : tbl)
                        neighbors.resize(n, -1);
                    // TODO(anstaf): is the order within neighbors OK?
                    return storage::builder<Traits>.template type<int>().dimensions(s, n).initializer(
                        [&](auto h, auto n) { return tbl[h][n]; })();
                }

                template <class Traits, class Lhs, class Rhs, class S, class N>
                auto compose_neighbor_tables(Lhs &&lhs, Rhs &&rhs, S s, N n) {
                    std::vector<std::vector<int>> tbl(s);
                    for (auto &&neighbors : tbl)
                        neighbors.resize(n, -1);
                    return storage::builder<Traits>.template type<int>().dimensions(s, n).initializer(
                        [&](auto h, auto n) { return tbl[h][n]; })();
                }

                // TODO: build missing tables
                template <class Traits, class From, class To, class... Tos>
                constexpr auto get_neighbors_table =
                    [](auto const &mesh) -> decltype(mesh_get_neighbors_table(mesh, Traits(), From(), To(), Tos()...)) {
                    return mesh_get_neighbors_table(mesh, Traits(), From(), To(), Tos()...);
                };
            } // namespace concept_impl_
            using concept_impl_::get_location_number;
            using concept_impl_::get_max_neighbors_number;
            using concept_impl_::get_neighbors_table;
            using concept_impl_::has_skip_values;
        } // namespace mesh
    }     // namespace next
} // namespace gridtools
