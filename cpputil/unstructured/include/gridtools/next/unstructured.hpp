#pragma once

#include <gridtools/common/integral_constant.hpp>

namespace gridtools {
    namespace next {
        struct vertex {};
        struct edge {};
        struct cell {};

        template <class...>
        struct neighbor {};

        namespace dim {
            using horizontal = integral_constant<int, 0>;
            using vertical = integral_constant<int, 1>;
            using h = horizontal;
            using v = vertical;
            using k = vertical;
        } // namespace dim
    }     // namespace next
} // namespace gridtools
