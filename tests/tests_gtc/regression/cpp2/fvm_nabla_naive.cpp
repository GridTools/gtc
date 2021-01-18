#include "fvm_nabla_naive.hpp"
#include <gtest/gtest.h>
#include <tests/fvm_nabla_driver.hpp>

TEST(fvm, nabla_naive) { fvm_nabla_driver(nabla); }
