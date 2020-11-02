void lap(Domain domain, Field &out, Field &in) {
  for (std::size_t i = 1; i < domain[0] - 1; ++i) {
    for (std::size_t j = 1; j < domain[1] - 1; ++j) {
      out[i + 0][j + 0] =
          ((-4 * in[i + 0][j + 0]) + ((in[i + -1][j + 0] + in[i + 1][j + 0]) +
                                      (in[i + 0][j + -1] + in[i + 0][j + 1])));
    }
  }
}