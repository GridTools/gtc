#include "nabla_cuda.hpp"
#include <gtest/gtest.h>
#include <tests/fvm_nabla_driver.hpp>

TEST(fvm, nabla_cuda) { fvm_nabla_driver(nabla); }
