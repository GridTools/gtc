#pragma once

#include <gridtools/common/generic_metafunctions/utility.hpp>
#include <gridtools/common/host_device.hpp>

namespace gridtools {
    namespace next {
        namespace mesh {
            template <class... Locations>
            struct neighbor_table_f {
                static_assert(sizeof...(Locations) > 1);
                // TODO: verify that the loactions within every link are different
                template <class Mesh>
                auto operator()(Mesh const &mesh) const {
                    return mesh_get_neighbor_table(mesh, Locations()...);
                }
            };

            template <class Location, class Mesh>
            auto get_location_size(Mesh const &mesh) {
                return mesh_get_location_size(mesh, Location());
            }

            template <class... Keys>
            constexpr neighbor_table_f<Keys...> get_neighbor_table = {};

            template <class Neighbors, class Fun>
            GT_FUNCTION void for_each_neighbor(Neighbors &&neighbors, Fun &&fun) {
                mesh_for_each_neighbor(wstd::forward<Neighbors>(neighbors), wstd::forward<Fun>(fun));
            }

            GT_FUNCTION int mesh_get_neighbor_index(int i) { return i; }
            GT_FUNCTION int mesh_get_neighbor_sign(int) { return 1; }

            template <class NeighborInfo>
            GT_FUNCTION auto get_neighbor_index(NeighborInfo const &info) {
                return mesh_get_neighbor_index(info);
            }

            template <class NeighborInfo>
            GT_FUNCTION auto get_neighbor_sign(NeighborInfo const &info) {
                return mesh_get_neighbor_sign(info);
            }
        } // namespace mesh
    }     // namespace next
} // namespace gridtools
