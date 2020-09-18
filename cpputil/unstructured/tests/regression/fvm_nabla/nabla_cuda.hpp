#pragma once

#ifndef __CUDACC__
#error expected cuda compilation context
#endif

#include <gridtools/common/cuda_util.hpp>
#include <gridtools/common/hymap.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/next/cuda_util.hpp>
#include <gridtools/next/mesh.hpp>
#include <gridtools/next/unstructured.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/concept.hpp>

namespace nabla_cuda_impl_ {
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
    struct sign_tag;

    template <class Size, class EdgePtrs, class EdgeStrides, class PpPtr, class PpStrides>
    __global__ void nabla_edge(
        Size size, EdgePtrs edge_ptr_holders, EdgeStrides edge_strides, PpPtr pp_ptr_holder, PpStrides pp_strides) {
        {
            auto idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size)
                return;

            auto edge_ptrs = edge_ptr_holders();
            auto pp_ptr = pp_ptr_holder();
            auto pp_stride = sid::get_stride<vertex>(pp_strides);

            sid::shift(edge_ptrs, sid::get_stride<edge>(edge_strides), idx);

            double acc = 0;
            mesh::for_each_neighbor(*device::at_key<connectivity_tag>(edge_ptrs),
                [&](auto neighbor) { acc += *sid::shifted(pp_ptr, pp_stride, mesh::get_neighbor_index(neighbor)); });
            double zavg = .5 * acc;
            *device::at_key<zavgS_MXX_tag>(edge_ptrs) = *device::at_key<S_MXX_tag>(edge_ptrs) * zavg;
            *device::at_key<zavgS_MYY_tag>(edge_ptrs) = *device::at_key<S_MYY_tag>(edge_ptrs) * zavg;
        }
    }

    template <class Size,
        class VertexOrigins,
        class VertexStrides,
        class EdgeNeighborOrigins,
        class EdgeNeighborStrides>
    __global__ void nabla_vertex(Size size,
        VertexOrigins vertex_origins,
        VertexStrides vertex_strides,
        EdgeNeighborOrigins edge_neighbor_origins,
        EdgeNeighborStrides edge_neighbor_strides) {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= size)
            return;

        auto vertex_ptrs = vertex_origins();
        auto edge_ptrs = edge_neighbor_origins();

        sid::shift(vertex_ptrs, sid::get_stride<vertex>(vertex_strides), idx);

        double accXX = 0;
        double accYY = 0;
        mesh::for_each_neighbor(*device::at_key<connectivity_tag>(vertex_ptrs), [&](auto neighbor) {
            auto neighbor_ptrs = sid::shifted(
                edge_ptrs, sid::get_stride<edge>(edge_neighbor_strides), mesh::get_neighbor_index(neighbor));
            auto sign = mesh::get_neighbor_sign(neighbor);
            accXX += *device::at_key<zavgS_MXX_tag>(neighbor_ptrs) * sign;
            accYY += *device::at_key<zavgS_MYY_tag>(neighbor_ptrs) * sign;
        });
        auto vol = *device::at_key<vol_tag>(vertex_ptrs);
        *device::at_key<pnabla_MXX_tag>(vertex_ptrs) = accXX / vol;
        *device::at_key<pnabla_MYY_tag>(vertex_ptrs) = accYY / vol;
    }

    template <class Size, class T, class U>
    auto run(Size size, T t, U u) {
        return
            [&](void (*fun)(
                Size, sid::ptr_holder_type<T>, sid::strides_type<T>, sid::ptr_holder_type<U>, sid::strides_type<U>)) {
                auto [blocks, threads_per_block] = cuda_setup(size);
                fun<<<blocks, threads_per_block>>>(
                    size, sid::get_origin(t), sid::get_strides(t), sid::get_origin(u), sid::get_strides(u));
                GT_CUDA_CHECK(cudaDeviceSynchronize());
            };
    }

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
        run(mesh::get_location_size<edge>(mesh),
            tuple_util::make<
                sid::composite::keys<connectivity_tag, S_MXX_tag, S_MYY_tag, zavgS_MXX_tag, zavgS_MYY_tag>::values>(
                mesh::get_neighbor_table<edge, vertex>(mesh), S_MXX, S_MYY, zavgS_MXX, zavgS_MYY),
            std::forward<pp_t>(pp))(nabla_edge);
        run(mesh::get_location_size<vertex>(mesh),
            tuple_util::make<sid::composite::keys<connectivity_tag, pnabla_MXX_tag, pnabla_MYY_tag, vol_tag>::values>(
                mesh::get_neighbor_table<vertex, edge>(mesh), pnabla_MXX, pnabla_MYY, vol),
            tuple_util::make<sid::composite::keys<zavgS_MXX_tag, zavgS_MYY_tag>::values>(zavgS_MXX, zavgS_MYY))(
            nabla_vertex);
    }
} // namespace nabla_cuda_impl_
using nabla_cuda_impl_::nabla;
