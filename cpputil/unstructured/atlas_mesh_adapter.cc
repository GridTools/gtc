#include <iostream>
#include <type_traits>

#include <atlas/grid.h>
#include <atlas/grid/StructuredGrid.h>
#include <atlas/mesh.h>
#include <atlas/mesh/actions/BuildCellCentres.h>
#include <atlas/mesh/actions/BuildDualMesh.h>
#include <atlas/mesh/actions/BuildEdges.h>
#include <atlas/meshgenerator.h>

#include <gridtools/common/tuple_util.hpp>
#include <gridtools/sid/composite.hpp>

#include <gridtools/next/atlas_adapter.hpp>
#include <gridtools/next/atlas_array_view_adapter.hpp>
#include <gridtools/next/mesh.hpp>

#include "tests/include/util/atlas_util.hpp"

using namespace gridtools;
using namespace next;
namespace tu = tuple_util;

int main() {
    auto mesh = ::atlas_util::make_mesh();
    atlas::mesh::actions::build_edges(mesh);
    atlas::mesh::actions::build_node_to_edge_connectivity(mesh);

    auto const &v2e = mesh::connectivity<vertex, edge>(mesh);

    auto n_vertices = connectivity::size(v2e);
    std::cout << n_vertices << std::endl;
    std::cout << connectivity::max_neighbors(v2e) << std::endl;
    std::cout << connectivity::skip_value(v2e) << std::endl;
    auto v2e_tbl = connectivity::neighbor_table(v2e);

    static_assert(is_sid<decltype(v2e_tbl)>{});
    auto composite = tu::make<sid::composite::keys<edge // TODO just a dummy
        >::values>(v2e_tbl);
    static_assert(is_sid<decltype(composite)>{});

    auto strides = sid::get_strides(v2e_tbl);

    std::cout << at_key<neighbor>(strides) << std::endl;
    std::cout << at_key<vertex>(strides) << std::endl;

    for (std::size_t v = 0; v < connectivity::size(v2e); ++v) {
        auto ptr = sid::get_origin(v2e_tbl)();
        sid::shift(ptr, at_key<vertex>(strides), v);
        for (std::size_t i = 0; i < next::connectivity::max_neighbors(v2e); ++i) {
            sid::shift(ptr, at_key<neighbor>(strides), 1);
        }
    }
}
