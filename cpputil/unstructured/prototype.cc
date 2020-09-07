#include <iostream>
#include <tuple>
#include <vector>

#include <gridtools/common/hymap.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/next/mesh.hpp>
#include <gridtools/next/unstructured.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/sid/synthetic.hpp>

using namespace gridtools;
using namespace next;

namespace tu = tuple_util;

using vertex2edge = meta::list<vertex, edge>;
using edge2vertex = meta::list<edge, vertex>;
using cell2vertex = meta::list<cell, vertex>;

namespace my_personal_connectivity {
    template <std::size_t MaxNeighbors, class LocationType>
    struct myCon {
        using location = LocationType;
        static constexpr std::size_t neighs = MaxNeighbors;
        std::vector<std::array<int, MaxNeighbors>> neighborTable;
    };

    template <std::size_t MaxNeighbors, class LocationType>
    auto connectivity_max_neighbors(myCon<MaxNeighbors, LocationType> const &) {
        return std::integral_constant<std::size_t, MaxNeighbors>{};
    }

    template <std::size_t MaxNeighbors, class LocationType>
    std::size_t connectivity_size(myCon<MaxNeighbors, LocationType> const &conn) {
        return conn.neighborTable.size();
    }

    template <std::size_t MaxNeighbors, class LocationType>
    auto connectivity_neighbor_table(myCon<MaxNeighbors, LocationType> const &conn) {

        using strides_t = typename hymap::keys<LocationType,
            neighbor>::template values<std::integral_constant<size_t, MaxNeighbors>, std::integral_constant<size_t, 1>>;

        return sid::synthetic()
            .set<sid::property::origin>(
                sid::make_simple_ptr_holder(reinterpret_cast<const int *>(conn.neighborTable.data())))
            .template set<sid::property::strides>(
                strides_t(std::integral_constant<size_t, MaxNeighbors>{}, std::integral_constant<size_t, 1>{}))
            .template set<sid::property::strides_kind, strides_t>();
    }
} // namespace my_personal_connectivity

struct in_tag;
struct out_tag;
struct connectivity_tag;

template <class SID, class NeighPtrT>
auto indirect_access(SID &&field, NeighPtrT const &neighptr) {
    auto ptr = sid::get_origin(field)();
    sid::shift(ptr, sid::get_stride<vertex>(sid::get_strides(field)), *neighptr);
    return ptr;
}

template <class Mesh, class In, class Out>
void sum_vertex_to_cell(Mesh const &mesh, In &&in, Out &&out) {
    auto n_cells = connectivity::size(at_key<cell2vertex>(mesh));

    auto cell2vertex_conn = mesh::connectivity<cell, vertex>(mesh);
    auto cell_to_vertex = connectivity::neighbor_table(cell2vertex_conn);

    static_assert(sid::concept_impl_::is_sid<decltype(cell_to_vertex)>{});

    auto cells = tu::make<sid::composite::keys<out_tag, connectivity_tag>::values>(out, cell_to_vertex);

    static_assert(sid::concept_impl_::is_sid<decltype(cells)>{});

    auto ptr = sid::get_origin(cells)();
    for (std::size_t i = 0; i < n_cells; ++i) {
        std::cout << "cell: " << *at_key<out_tag>(ptr) << std::endl;
        int sum = 0;
        auto neigh_ptr = at_key<connectivity_tag>(ptr);
        for (std::size_t neigh_vertex = 0; neigh_vertex < next::connectivity::max_neighbors(at_key<cell2vertex>(mesh));
             ++neigh_vertex) {
            auto absolute_neigh_index = *neigh_ptr;
            std::cout << absolute_neigh_index << " ";

            auto in_ptr = indirect_access(in, neigh_ptr);

            std::cout << *in_ptr << std::endl;

            sum += *in_ptr;

            // last thing in the loop: shift
            sid::shift(neigh_ptr, sid::get_stride<neighbor>(sid::get_strides(cell_to_vertex)), 1);
        }
        *at_key<out_tag>(ptr) = sum;
        sid::shift(ptr, sid::get_stride<cell>(sid::get_strides(cells)), 1); // TODO sid::loop
    }
}

auto make_cell_to_vertex_connectivity() {
    std::vector<std::array<int, 4>> res;
    res.emplace_back(std::array{0, 1, 3, 4});
    res.emplace_back(std::array{1, 2, 4, 5});
    res.emplace_back(std::array{3, 4, 6, 7});
    res.emplace_back(std::array{4, 5, 7, 8});
    return res;
}

int main() {
    auto mesh = tu::make<hymap::keys<cell2vertex>::values>(
        my_personal_connectivity::myCon<4, cell>{make_cell_to_vertex_connectivity()});

    using namespace literals;

    constexpr int size = 3;
    int in_array[size][size]; // on vertices
    auto in = sid::synthetic()
                  .set<sid::property::origin>(sid::make_simple_ptr_holder(&in_array[0][0]))
                  .set<sid::property::strides>(tu::make<hymap::keys<vertex>::values>(1_c))
                  .set<sid::property::upper_bounds>(tu::make<hymap::keys<vertex>::values>(size * size));

    static_assert(is_sid<decltype(in)>{});

    int out_array[size - 1][size - 1]; // on cells
    auto out = sid::synthetic()
                   .set<sid::property::origin>(sid::make_simple_ptr_holder(&out_array[0][0]))
                   .set<sid::property::strides>(tu::make<hymap::keys<cell>::values>(1_c))
                   .set<sid::property::upper_bounds>(tu::make<hymap::keys<cell>::values>((size - 1) * (size - 1)));

    for (int i = 0; auto &&row : in_array)
        for (auto &&val : row) {
            val = i++;
        }
    for (int i = 0; auto &&row : out_array)
        for (auto &&val : row) {
            val = i++;
        }

    sum_vertex_to_cell(mesh, in, out);

    std::cout << "=== result ===" << std::endl;
    for (auto &&row : out_array)
        for (auto &&val : row) {
            std::cout << val << std::endl;
        }
}
