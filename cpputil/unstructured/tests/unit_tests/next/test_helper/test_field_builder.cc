#include <array>
#include <utility>

#include <gridtools/next/mesh.hpp>
#include <gridtools/next/test_helper/field_builder.hpp>
#include <gridtools/next/test_helper/simple_mesh.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/sid/loop.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {
    using namespace gridtools;
    using namespace next;

    TEST(field_builder, cell_field) {
        test_helper::simple_mesh mesh;

        auto field = test_helper::make_field<double, cell>(mesh);

        static_assert(std::is_same<double *, sid::ptr_type<decltype(field)>>{});
        EXPECT_EQ(mesh::get_location_size<cell>(mesh), at_key<cell>(sid::get_upper_bounds(field)));
    }

    struct in_tag;
    struct out_tag;
    struct connectivity_tag;

    template <class Dim, class Mesh, class Sid, class Fun>
    void make_full_loop(Mesh const &mesh, Sid &&sid, Fun &&fun) {
        sid::make_loop<Dim>(mesh::get_location_size<Dim>(mesh))(std::forward<Fun>(fun))(
            sid::get_origin(sid)(), sid::get_strides(sid));
    }

    TEST(simple_mesh_regression, cell2cell_reduction) {
        test_helper::simple_mesh mesh;
        auto in = test_helper::make_field<double, cell>(mesh);
        auto out = test_helper::make_field<double, cell>(mesh);
        make_full_loop<cell>(mesh, in, [](auto ptr, auto &&) { *ptr = 1; });
        make_full_loop<cell>(mesh,
            tuple_util::make<sid::composite::keys<out_tag, connectivity_tag>::values>(
                out, mesh::get_neighbor_table<cell, cell>(mesh)),
            [&, in_ptr = sid::get_origin(in)(), in_stride = sid::get_stride<cell>(sid::get_strides(in))](
                auto ptrs, auto &&) {
                double acc = 0;
                mesh::for_each_neighbor(*at_key<connectivity_tag>(ptrs), [&](auto neighbor) {
                    acc += *sid::shifted(in_ptr, in_stride, mesh::get_neighbor_index(neighbor));
                });
                *at_key<out_tag>(ptrs) = acc;
            });
        make_full_loop<cell>(mesh, out, [](auto ptr, auto &&) { EXPECT_EQ(4, *ptr); });
    }
} // namespace
