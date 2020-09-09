#include <type_traits>

#include <gridtools/next/mesh.hpp>
#include <gridtools/sid/rename_dimensions.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>

#ifdef __CUDACC__ // TODO proper handling
#include <gridtools/storage/gpu.hpp>
using storage_trait = gridtools::storage::gpu;
#else
#include <gridtools/storage/cpu_ifirst.hpp>
using storage_trait = gridtools::storage::cpu_ifirst;
#endif

namespace gridtools::next::test_helper {
    template <class T, class Location, class Mesh>
    auto make_field(Mesh const &mesh) {
        return sid::rename_numbered_dimensions<Location>(storage::builder<storage_trait>.template type<T>().dimensions(
            connectivity::size(mesh::connectivity<Location>(mesh))).build());
    }
} // namespace gridtools::next::test_helper
