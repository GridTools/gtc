cmake_minimum_required(VERSION 3.14.5)
project(gtc_regression2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(usid
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../cpputil/usid)
FetchContent_MakeAvailable(usid)

include(CTest)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# TODO(anstaf): factor out the stuff into cmake function 
add_custom_command(
    OUTPUT fvm_nabla_naive.hpp
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fvm_nabla.py naive > fvm_nabla_naive.hpp)
add_custom_target(fvm_nabla_naive_header DEPENDS fvm_nabla_naive.hpp)
add_executable(fvm_nabla_naive fvm_nabla_naive.cpp)
target_include_directories(fvm_nabla_naive PRIVATE ${CMAKE_BINARY_DIR})
add_dependencies(fvm_nabla_naive fvm_nabla_naive_header)
target_link_libraries(fvm_nabla_naive PRIVATE usid::usid_naive_helper usid::fvm_nabla_driver gtest_main)
add_test(NAME fvm_nabla_naive COMMAND $<TARGET_FILE:fvm_nabla_naive>)

# TODO(anstaf): add gpu executable
