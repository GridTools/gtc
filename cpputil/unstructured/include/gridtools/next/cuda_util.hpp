#pragma once

#include <tuple>

namespace gridtools::next {
    inline std::tuple<int, int> cuda_setup(int N) {
        int threads_per_block = 32;
        int blocks = (N + threads_per_block - 1) / threads_per_block;
        return {blocks, threads_per_block};
    }
} // namespace gridtools::next
