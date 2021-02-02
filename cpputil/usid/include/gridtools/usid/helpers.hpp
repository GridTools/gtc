#pragma once

#include <limits>
#include <type_traits>
#include <utility>

#include <gridtools/common/defs.hpp>
#include <gridtools/common/host_device.hpp>
#include <gridtools/common/hymap.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/utility.hpp>
#include <gridtools/meta/id.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/concept.hpp>
#include <gridtools/sid/contiguous.hpp>
#include <gridtools/sid/loop.hpp>
#include <gridtools/sid/rename_dimensions.hpp>

#include "dim.hpp"
#include "domain.hpp"

namespace gridtools::usid {
template <class T, class Alloc, class HSize, class KSize>
auto make_simple_tmp_storage(HSize h_size, KSize k_size, Alloc &alloc) {
  return sid::make_contiguous<T>(
      alloc, hymap::keys<dim::h, dim::k>::values<HSize, KSize>(h_size, k_size));
}

template <class T, class Tag, int_t N, bool HasSkipValues, class F, class Init,
          class G, class Ptr, class Strides, class Neighbors>
GT_FUNCTION T fold_neighbors(F f, Init init, G g, Ptr &&ptr, Strides &&strides,
                             Neighbors &&neighbors) {
  T acc = init(meta::lazy::id<T>());
  sid::make_loop<Tag>(
      integral_constant<int_t, N>())([&](auto const &ptr, auto &&) {
    auto i = *host_device::at_key<Tag>(ptr);
    if constexpr (HasSkipValues)
      if (i < 0)
        return;
    acc = f(acc, g(ptr, sid::shifted(neighbors.first, neighbors.second, i)));
  })(wstd::forward<decltype(ptr)>(ptr), strides);
  return acc;
}

template <class T, class Tag, int_t N, bool HasSkip = true, class F,
          class... Args>
GT_FUNCTION T sum_neighbors(F f, Args &&...args) {
  return fold_neighbors<T, Tag, N, HasSkip>(
      [](auto x, auto y) { return x + y; },
      [](auto z) -> typename decltype(z)::type { return 0; }, f,
      wstd::forward<Args>(args)...);
}

template <class T, class Tag, int_t N, bool HasSkip = true, class F,
          class... Args>
GT_FUNCTION T mul_neighbors(F f, Args &&...args) {
  return fold_neighbors<T, Tag, N, HasSkip>(
      [](auto x, auto y) { return x * y; },
      [](auto z) -> typename decltype(z)::type { return 1; }, f,
      wstd::forward<Args>(args)...);
}

template <class T, class Tag, int_t N, bool HasSkip = true, class F,
          class... Args>
GT_FUNCTION T min_neighbors(F f, Args &&...args) {
  return fold_neighbors<T, Tag, N, HasSkip>(
      [](auto x, auto y) { return x < y ? x : y; },
      [](auto z) -> typename decltype(z)::type {
        constexpr auto res =
            std::numeric_limits<typename decltype(z)::type>::max();
        return res;
      },
      f, wstd::forward<Args>(args)...);
}

template <class T, class Tag, int_t N, bool HasSkip = true, class F,
          class... Args>
GT_FUNCTION T max_neighbors(F f, Args &&...args) {
  return fold_neighbors<T, Tag, N, HasSkip>(
      [](auto x, auto y) { return x > y ? x : y; },
      [](auto z) -> typename decltype(z)::type {
        constexpr auto res =
            std::numeric_limits<typename decltype(z)::type>::min();
        return res;
      },
      f, wstd::forward<Args>(args)...);
}
} // namespace gridtools::usid
