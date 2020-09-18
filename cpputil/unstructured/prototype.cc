#include <iostream>

#include <gridtools/common/tuple_util.hpp>
#include <gridtools/next/mesh.hpp>
#include <gridtools/next/unstructured.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/sid/loop.hpp>
#include <gridtools/sid/rename_dimensions.hpp>
#include <gridtools/sid/synthetic.hpp>

using namespace gridtools;
using namespace next;
namespace tu = tuple_util;

namespace my_personal_connectivity {
    struct myNeighbors {
        int m_vals[4];
    };

    template <class Fun>
    GT_FUNCTION void mesh_for_each_neighbor(myNeighbors obj, Fun fun) {
        for (int i : obj.m_vals)
            fun(i);
    }

    struct myMesh {};

    int mesh_get_location_size(myMesh, cell) { return 4; }

    auto mesh_get_neighbor_table(myMesh, cell, vertex) {
        static const myNeighbors table[4] = {{{0, 1, 3, 4}}, {{1, 2, 4, 5}}, {{3, 4, 6, 7}}, {{4, 5, 7, 8}}};
        return sid::rename_numbered_dimensions<cell>(table);
    }
} // namespace my_personal_connectivity
using my_personal_connectivity::myMesh;

struct in_tag;
struct out_tag;
struct connectivity_tag;

template <class SID, class Offset>
auto indirect_access(SID &&field, Offset offset) {
    return sid::shifted(sid::get_origin(field)(), sid::get_stride<vertex>(sid::get_strides(field)), offset);
}

template <class Mesh, class In, class Out>
void sum_vertex_to_cell(Mesh const &mesh, In &&in, Out &&out) {
    auto n_cells = mesh::get_location_size<cell>(mesh);
    auto cell_to_vertex = mesh::get_neighbor_table<cell, vertex>(mesh);
    static_assert(is_sid<decltype(cell_to_vertex)>());
    auto cells = tu::make<sid::composite::keys<out_tag, connectivity_tag>::values>(out, cell_to_vertex);
    static_assert(is_sid<decltype(cells)>());
    sid::make_loop<cell>(n_cells)([&](auto ptr, auto &&) {
        std::cout << "cell: " << *at_key<out_tag>(ptr) << std::endl;
        int sum = 0;
        mesh::for_each_neighbor(*at_key<connectivity_tag>(ptr), [&](auto neighbor) {
            auto absolute_neigh_index = mesh::get_neighbor_index(neighbor);
            std::cout << absolute_neigh_index << " ";
            auto in_ptr = indirect_access(in, absolute_neigh_index);
            std::cout << *in_ptr << std::endl;
            sum += *in_ptr;
        });
        *at_key<out_tag>(ptr) = sum;
    })(sid::get_origin(cells)(), sid::get_strides(cells));
}

int main() {
    myMesh mesh;

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
        for (auto &&val : row)
            val = i++;

    for (int i = 0; auto &&row : out_array)
        for (auto &&val : row)
            val = i++;

    sum_vertex_to_cell(mesh, in, out);

    std::cout << "=== result ===" << std::endl;
    for (auto &&row : out_array)
        for (auto &&val : row)
            std::cout << val << std::endl;
}
