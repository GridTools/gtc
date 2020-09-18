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
#include <gridtools/next/atlas_adapter.hpp>
#include <gridtools/next/atlas_array_view_adapter.hpp>
#include <gridtools/next/mesh.hpp>
#include <gridtools/next/unstructured.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/loop.hpp>

#include "tests/include/util/atlas_util.hpp"

using namespace gridtools;
using namespace next;

int main() {
    auto mesh = ::atlas_util::make_mesh();
    atlas::mesh::actions::build_edges(mesh);
    atlas::mesh::actions::build_node_to_edge_connectivity(mesh);

    auto &&v2e = mesh::get_neighbor_table<vertex, edge>(mesh);

    auto &&n_vertices = mesh::get_location_size<vertex>(mesh);
    std::cout << n_vertices << std::endl;

    static_assert(is_sid<decltype(v2e)>());
    auto composite = tuple_util::make<sid::composite::keys<vertex>::values>(v2e);
    static_assert(is_sid<decltype(composite)>());

    auto strides = sid::get_strides(v2e);

    std::cout << sid::get_stride<vertex>(strides) << std::endl;

    sid::make_loop<vertex>(n_vertices)([](auto ptr, auto &&) {
        mesh::for_each_neighbor(*ptr, [](auto neighbor) { mesh::get_neighbor_index(neighbor); });
    })(sid::get_origin(v2e)(), sid::get_strides(v2e));
}
