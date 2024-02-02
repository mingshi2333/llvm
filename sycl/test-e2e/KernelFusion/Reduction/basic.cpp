// RUN: %{build} -fsycl-embed-ir -o %t.out
// RUN: %{run} %t.out

#include "./reduction.hpp"

#include <array>

int main() {
  constexpr std::array<std::size_t, 3> localSizes{
      globalSize /*Test single work-group*/,
      globalSize / 32 /*Test middle-sized work-group*/,
      1 /*Test single item work-groups*/};
  for (size_t localSize : localSizes) {
    test<detail::reduction::strategy::basic>(
        nd_range<1>{globalSize, localSize});
  }
}
