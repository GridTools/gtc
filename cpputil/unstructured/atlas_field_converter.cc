#include <type_traits>

#include <array_fwd.h>
#include <atlas/array.h>
#include <atlas/grid.h>
#include <atlas/grid/StructuredGrid.h>
#include <atlas/mesh.h>
#include <atlas/mesh/actions/BuildCellCentres.h>
#include <atlas/mesh/actions/BuildDualMesh.h>
#include <atlas/mesh/actions/BuildEdges.h>
#include <atlas/meshgenerator.h>
#include <atlas/option.h>
#include <field/Field.h>
#include <functionspace/EdgeColumns.h>

#include <gridtools/common/integral_constant.hpp>
#include <gridtools/next/atlas_adapter.hpp>
#include <gridtools/next/atlas_array_view_adapter.hpp>
#include <gridtools/next/atlas_field_util.hpp>
#include <gridtools/next/mesh.hpp>
#include <gridtools/sid/rename_dimensions.hpp>
#include <gridtools/sid/synthetic.hpp>

#include "tests/include/util/atlas_util.hpp"

using namespace gridtools;
using namespace next;

int main() {
    auto mesh = ::atlas_util::make_mesh();
    atlas::mesh::actions::build_edges(mesh);

    int nb_levels = 5;
    atlas::functionspace::EdgeColumns fs_edges(mesh, atlas::option::levels(nb_levels) | atlas::option::halo(1));

    atlas::Field f;
    auto my_field = fs_edges.createField<double>(atlas::option::name("my_field"));

    auto view = atlas::array::make_view<double, 2>(my_field);
    for (int i = 0; i < fs_edges.size(); ++i)
        for (int k = 0; k < nb_levels; ++k)
            view(i, k) = i * 10 + k;

    auto my_field_sidified = next::atlas_util::as<edge, dim::k>::with_type<double>{}(my_field);
    static_assert(is_sid<decltype(my_field_sidified)>{});

    auto my_field_as_data_store = next::atlas_util::as_data_store<edge, dim::k>::with_type<double>{}(my_field);
    static_assert(is_sid<decltype(my_field_as_data_store)>{});

    auto strides = sid::get_strides(my_field_sidified);
    auto strides_ds = sid::get_strides(my_field_as_data_store);
    for (int i = 0; i < fs_edges.size(); ++i)
        for (int k = 0; k < nb_levels; ++k) {
            auto ptr = sid::get_origin(my_field_sidified)();
            sid::shift(ptr, at_key<edge>(strides), i);
            sid::shift(ptr, at_key<dim::k>(strides), k);
            auto ptr2 = sid::get_origin(my_field_as_data_store)();
            sid::shift(ptr2, at_key<edge>(strides_ds), i);
            sid::shift(ptr2, at_key<dim::k>(strides_ds), k);
            std::cout << view(i, k) << "/" << *ptr << "/" << *ptr2 << std::endl;
        }
}
