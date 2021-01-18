cmake_minimum_required(VERSION 3.14.5)
project(usid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-Wall)

include(FetchContent)
FetchContent_Declare(GridTools
    GIT_REPOSITORY https://github.com/GridTools/gridtools.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(GridTools)

add_library(usid_naive_helper INTERFACE)
add_library(usid::usid_naive_helper ALIAS usid_naive_helper)
target_include_directories(usid_naive_helper INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(usid_naive_helper INTERFACE GridTools::gridtools)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    add_library(usid_cuda_helper INTERFACE)
    add_library(usid::usid_cuda_helper ALIAS usid_cuda_helper)
    target_include_directories(usid_cuda_helper INTERFACE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(usid_cuda_helper INTERFACE GridTools::gridtools GridTools::stencil_gpu)
endif()

set(BUILD_SHARED_LIBS OFF)

FetchContent_Declare(GoogleTest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.10.0
)
FetchContent_MakeAvailable(GoogleTest)

find_package(eckit REQUIRED)
find_package(Atlas REQUIRED)

add_library(fvm_nabla_driver INTERFACE)
add_library(usid::fvm_nabla_driver ALIAS fvm_nabla_driver)
target_include_directories(fvm_nabla_driver INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(fvm_nabla_driver INTERFACE atlas eckit GridTools::gridtools gtest)
if(CMAKE_CUDA_COMPILER)
    target_link_libraries(fvm_nabla_driver INTERFACE GridTools::stencil_gpu)
endif()


if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()
