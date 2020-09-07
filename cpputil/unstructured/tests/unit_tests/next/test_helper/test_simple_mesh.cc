#include "gridtools/meta/debug.hpp"
#include <array>

#include <gridtools/meta/at.hpp>
#include <gridtools/next/mesh.hpp>
#include <gridtools/next/test_helper/simple_mesh.hpp>
#include <gridtools/sid/concept.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace gridtools;
using namespace next;

template <class Mesh, class Key, class... Keys>
auto get_neighbors(Mesh const &mesh, int i, Key, Keys...) {
    auto conn = mesh::connectivity<Key, Keys...>(mesh);
    auto tbl = connectivity::neighbor_table(conn);
    auto ptr = sid::get_origin(tbl)();
    auto strides = sid::get_strides(tbl);
    sid::shift(ptr, at_key<Key>(strides), i);
    std::array<int, decltype(connectivity::max_neighbors(conn))::value> result;
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = *ptr;
        sid::shift(ptr, at_key<neighbor>(strides), 1);
    }
    return result;
}

TEST(simple_mesh, cell2cell) {
    test_helper::simple_mesh mesh;

    auto c2c = mesh::connectivity<cell, cell>(mesh);

    EXPECT_EQ(9, connectivity::size(c2c));
    EXPECT_EQ(4, connectivity::max_neighbors(c2c));
    EXPECT_EQ(-1, connectivity::skip_value(c2c));

    EXPECT_THAT(get_neighbors(mesh, 0, cell(), cell()), testing::UnorderedElementsAre(1, 3, 2, 6));
    EXPECT_THAT(get_neighbors(mesh, 1, cell(), cell()), testing::UnorderedElementsAre(0, 7, 2, 4));
    // etc
}

TEST(simple_mesh, edge2vertex) {
    test_helper::simple_mesh mesh;

    auto e2v = mesh::connectivity<edge, vertex>(mesh);

    EXPECT_EQ(18, connectivity::size(e2v));
    EXPECT_EQ(2, connectivity::max_neighbors(e2v));
    EXPECT_EQ(-1, connectivity::skip_value(e2v));

    EXPECT_THAT(get_neighbors(mesh, 0, edge(), vertex()), testing::UnorderedElementsAre(0, 1));
    // etc
}
