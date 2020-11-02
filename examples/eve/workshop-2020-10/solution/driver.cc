#include <iomanip>
#include <iostream>

#include "defs.hpp"
#include "generated.hpp"

int main() {
  Field in{
      {1, 1, 1, 1, 1}, {1, 2, 2, 2, 1}, {1, 2, 3, 2, 1},
      {1, 2, 2, 2, 1}, {1, 1, 1, 1, 1},
  };
  Field out{};

  Domain domain{5, 5};

  lap(domain, out, in);

  for (std::size_t i = 0; i < domain[0]; ++i) {
    for (std::size_t j = 0; j < domain[1]; ++j) {
      std::cout << std::setw(3) << out[i][j] << " ";
    }
    std::cout << std::endl;
  }
}