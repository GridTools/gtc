#include "unstructured.hpp"
#include <gridtools/common/hymap.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/sid/contiguous.hpp>

namespace gridtools {
    namespace next {
        template <class Location, class Data, class Allocator, class USize, class KSize>
        auto make_simple_tmp_storage(USize u_size, KSize k_size, Allocator &alloc) {
            return sid::make_contiguous<Data, int>(
                alloc, tuple_util::make<hymap::keys<Location, dim::k>::template values>(u_size, k_size));
        }
    } // namespace next
} // namespace gridtools
