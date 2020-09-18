#pragma once

#include <gridtools/common/hymap.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/next/mesh.hpp>
#include <gridtools/next/unstructured.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/sid/loop.hpp>

namespace nabla_impl_ {
    using namespace gridtools;
    using namespace next;

    struct connectivity_tag;
    struct S_MXX_tag;
    struct S_MYY_tag;
    struct zavgS_MXX_tag;
    struct zavgS_MYY_tag;
    struct pnabla_MXX_tag;
    struct pnabla_MYY_tag;
    struct vol_tag;

    template <class Mesh,
        class S_MXX_t,
        class S_MYY_t,
        class zavgS_MXX_t,
        class zavgS_MYY_t,
        class pp_t,
        class pnabla_MXX_t,
        class pnabla_MYY_t,
        class vol_t>
    void nabla(Mesh &&mesh,
        S_MXX_t &&S_MXX,
        S_MYY_t &&S_MYY,
        zavgS_MXX_t &&zavgS_MXX,
        zavgS_MYY_t &&zavgS_MYY,
        pp_t &&pp,
        pnabla_MXX_t &&pnabla_MXX,
        pnabla_MYY_t &&pnabla_MYY,
        vol_t &&vol) {
        static_assert(is_sid<S_MXX_t>());
        static_assert(is_sid<S_MYY_t>());
        static_assert(is_sid<zavgS_MXX_t>());
        static_assert(is_sid<zavgS_MYY_t>());
        static_assert(is_sid<pp_t>());
        static_assert(is_sid<pnabla_MXX_t>());
        static_assert(is_sid<pnabla_MYY_t>());
        static_assert(is_sid<vol_t>());

        auto edge_fields = tuple_util::make<
            sid::composite::keys<connectivity_tag, S_MXX_tag, S_MYY_tag, zavgS_MXX_tag, zavgS_MYY_tag>::values>(
            mesh::get_neighbor_table<edge, vertex>(mesh), S_MXX, S_MYY, zavgS_MXX, zavgS_MYY);

        sid::make_loop<edge>(mesh::get_location_size<edge>(mesh))(
            [pp_ptr = sid::get_origin(pp)(), pp_stride = sid::get_stride<vertex>(sid::get_strides(pp))](
                auto &ptrs, auto &&) {
                double acc = 0;
                mesh::for_each_neighbor(*at_key<connectivity_tag>(ptrs), [&](auto neighbor) {
                    acc += *sid::shifted(pp_ptr, pp_stride, mesh::get_neighbor_index(neighbor));
                });
                double zavg = .5 * acc;
                *at_key<zavgS_MXX_tag>(ptrs) = *at_key<S_MXX_tag>(ptrs) * zavg;
                *at_key<zavgS_MYY_tag>(ptrs) = *at_key<S_MYY_tag>(ptrs) * zavg;
            })(sid::get_origin(edge_fields)(), sid::get_strides(edge_fields));

        auto vertex_fields =
            tuple_util::make<sid::composite::keys<connectivity_tag, pnabla_MXX_tag, pnabla_MYY_tag, vol_tag>::values>(
                mesh::get_neighbor_table<vertex, edge>(mesh), pnabla_MXX, pnabla_MYY, vol);

        sid::make_loop<vertex>(mesh::get_location_size<vertex>(mesh))(
            [zavgS_MXX_ptr = sid::get_origin(zavgS_MXX)(),
                zavgS_MXX_stride = sid::get_stride<edge>(sid::get_strides(zavgS_MXX)),
                zavgS_MYY_ptr = sid::get_origin(zavgS_MYY)(),
                zavgS_MYY_stride = sid::get_stride<edge>(sid::get_strides(zavgS_MYY))](auto &ptrs, auto &&) {
                double accXX = 0;
                double accYY = 0;
                mesh::for_each_neighbor(*at_key<connectivity_tag>(ptrs), [&](auto neighbor) {
                    auto index = mesh::get_neighbor_index(neighbor);
                    double sign = mesh::get_neighbor_sign(neighbor);
                    accXX += *sid::shifted(zavgS_MXX_ptr, zavgS_MXX_stride, index) * sign;
                    accYY += *sid::shifted(zavgS_MYY_ptr, zavgS_MYY_stride, index) * sign;
                });
                auto vol = *at_key<vol_tag>(ptrs);
                *at_key<pnabla_MXX_tag>(ptrs) = accXX / vol;
                *at_key<pnabla_MYY_tag>(ptrs) = accYY / vol;
            })(sid::get_origin(vertex_fields)(), sid::get_strides(vertex_fields));
    }
} // namespace nabla_impl_
using nabla_impl_::nabla;
