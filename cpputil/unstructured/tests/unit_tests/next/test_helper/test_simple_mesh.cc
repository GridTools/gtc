#include <gridtools/next/test_helper/simple_mesh.hpp>

#include <vector>

#include <gridtools/meta/at.hpp>
#include <gridtools/next/mesh.hpp>
#include <gridtools/next/unstructured.hpp>
#include <gridtools/sid/concept.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace gridtools;
using namespace next;

template <class Mesh, class Key, class... Keys>
std::vector<int> get_neighbors(Mesh const &mesh, int i, Key, Keys...) {
    auto tbl = mesh::get_neighbor_table<Key, Keys...>(mesh);
    std::vector<int> res;
    mesh::for_each_neighbor(*sid::shifted(sid::get_origin(tbl)(), sid::get_stride<Key>(sid::get_strides(tbl)), i),
        [&](auto neighbor) { res.push_back(mesh::get_neighbor_index(neighbor)); });
    return res;
}

TEST(simple_mesh, cell2cell) {
    test_helper::simple_mesh testee;

    EXPECT_EQ(9, mesh::get_location_size<cell>(testee));
    EXPECT_THAT(get_neighbors(testee, 0, cell(), cell()), testing::UnorderedElementsAre(1, 3, 2, 6));
    EXPECT_THAT(get_neighbors(testee, 1, cell(), cell()), testing::UnorderedElementsAre(0, 7, 2, 4));
    // etc
}

TEST(simple_mesh, edge2vertex) {
    test_helper::simple_mesh testee;

    EXPECT_EQ(18, mesh::get_location_size<edge>(testee));
    EXPECT_THAT(get_neighbors(testee, 0, edge(), vertex()), testing::UnorderedElementsAre(0, 1));
    // etc
}
