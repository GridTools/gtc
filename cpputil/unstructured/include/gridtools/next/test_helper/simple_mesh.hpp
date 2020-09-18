/**
 * Simple 3x3 Cartesian, periodic, hand-made mesh.
 *
 *      0e    1e    2e
 *   | ---- | ---- | ---- |
 * 9e|0v 10e|1v 11e|2v  9e|0v
 *   |  0c  |  1c  |  2c  |
 *   |  3e  |  4e  |  5e  |
 *   | ---- | ---- | ---- |
 *12e|3v 13e|4v 14e|5v 12e|3v
 *   |  3c  |  4c  |  5c  |
 *   |  6e  |  7e  |  8e  |
 *   | ---- | ---- | ---- |
 *15e|6v 16e|7v 17e|8v 15e| 6v
 *   |  6c  |  7c  |  8c  |
 *   |  0e  |  1e  |  2e  |
 *   | ---- | ---- | ---- |
 *    0v     1v     2v     0v
 *
 */

#include <type_traits>

#include <gridtools/common/generic_metafunctions/utility.hpp>
#include <gridtools/common/hymap.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/sid/concept.hpp>

#include "../unstructured.hpp"

namespace gridtools {
    namespace next {
        namespace test_helper {
            namespace simple_mesh_impl_ {
                template <auto TablePtr>
                struct neighbors {
                    int m_index;

                    neighbors operator*() const { return *this; }
                    GT_FUNCTION neighbors operator()() const { return *this; }
                };

                template <auto TablePtr>
                GT_FUNCTION neighbors<TablePtr> operator+(neighbors<TablePtr> obj, int diff) {
                    return {obj.m_index};
                }

                struct neighbors_stride {};

                template <auto TablePtr, class Offset>
                GT_FUNCTION void sid_shift(neighbors<TablePtr> &obj, neighbors_stride, Offset offset) {
                    obj.m_index += offset;
                }

                template <class Offset>
                GT_FUNCTION void sid_shift(int &obj, neighbors_stride, Offset offset) {
                    obj += offset;
                }

                template <auto TablePtr, class Fun>
                GT_FUNCTION void mesh_for_each_neighbor(neighbors<TablePtr> neighbors, Fun &&fun) {
                    for (auto &&item : (*TablePtr)[neighbors.m_index])
                        wstd::forward<Fun>(fun)(item);
                }

                template <class Dim, auto TablePtr>
                struct neighbors_table {
                    friend neighbors<TablePtr> sid_get_origin(neighbors_table) { return {0}; }
                    friend typename hymap::keys<Dim>::template values<neighbors_stride> sid_get_strides(
                        neighbors_table) {
                        return {};
                    }
                    friend int sid_get_ptr_diff(neighbors_table) { return {}; }
                    friend typename hymap::keys<Dim>::template values<integral_constant<int, 0>> sid_get_lower_bounds(
                        neighbors_table) {
                        return {};
                    }
                    friend typename hymap::keys<Dim>::template values<
                        integral_constant<int, std::extent_v<std::remove_pointer<decltype(TablePtr)>>>>
                    sid_get_upper_bounds(neighbors_table) {
                        return {};
                    }
                };

                struct simple_mesh {};

                inline integral_constant<int, 9> mesh_get_location_size(simple_mesh, vertex) { return {}; }
                inline integral_constant<int, 18> mesh_get_location_size(simple_mesh, edge) { return {}; }
                inline integral_constant<int, 9> mesh_get_location_size(simple_mesh, cell) { return {}; }

                auto mesh_get_neighbor_table(simple_mesh, cell, cell) {
                    static constexpr int table[][4] = {
                        //
                        {6, 1, 3, 2},
                        {7, 2, 4, 0},
                        {8, 0, 5, 1},
                        {0, 4, 6, 5},
                        {1, 5, 7, 3},
                        {2, 3, 8, 4},
                        {3, 7, 0, 8},
                        {4, 8, 1, 6},
                        {5, 6, 2, 7} //
                    };
                    return neighbors_table<cell, &table>{};
                }
                static_assert(is_sid<decltype(mesh_get_neighbor_table(simple_mesh(), cell(), cell()))>());

                auto mesh_get_neighbor_table(simple_mesh, edge, vertex) {
                    static constexpr int table[][2] = {
                        //
                        {0, 1},
                        {1, 2},
                        {2, 0},
                        {3, 4},
                        {4, 5},
                        {5, 3},
                        {6, 7},
                        {7, 8},
                        {8, 6},
                        {0, 3},
                        {1, 4},
                        {2, 5},
                        {3, 6},
                        {4, 7},
                        {5, 8},
                        {6, 0},
                        {7, 1},
                        {8, 2} //
                    };
                    return neighbors_table<cell, &table>{};
                }
                static_assert(is_sid<decltype(mesh_get_neighbor_table(simple_mesh(), edge(), vertex()))>());
            } // namespace simple_mesh_impl_
            using simple_mesh_impl_::simple_mesh;
        } // namespace test_helper
    }     // namespace next
} // namespace gridtools

/**
 * TODO maybe later: Simple 2x2 Cartesian hand-made mesh, non periodic, one halo line.
 *
 *     0e    1e    2e    3e
 *   | --- | --- | --- | --- |
 *   |0v   |1v   |2v   |3v   |4v
 *   |  0c |  1c |  2c |  3c |
 *   |     |     |     |     |
 *   | --- | --- | --- | --- |
 *   |     |     |     |     |
 *   |  4c |  5c |  6c |  7c |
 *   |     |     |     |     |
 *   | --- | --- | --- | --- |
 *   |     |     |     |     |
 *   |  8c |  9c | 10c | 11c |
 *   |     |     |     |     |
 *   | --- | --- | --- | --- |
 *   |     |     |     |     |
 *   | 12c | 13c | 14c | 15c |
 *   |     |     |     |     |
 *   | --- | --- | --- | --- |
 *
 *
 */
