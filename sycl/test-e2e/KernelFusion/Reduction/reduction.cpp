// RUN: %{build} -fsycl-embed-ir -o %t.out
// RUN: %{run} %t.out

// Test fusion works with reductions. Some algorithms will lead to fusion being
// cancelled in some devices. These should work properly anyway.

#include <sycl/sycl.hpp>
#include <utility>

#include "../helpers.hpp"
#include "sycl/detail/reduction_forward.hpp"

using namespace sycl;

constexpr inline size_t globalSize = 512;

template <typename BinaryOperation> class ReductionTest;

template <detail::reduction::strategy Strategy> void test(nd_range<1> ndr) {
  std::array<int, globalSize> data;
  int sumRes = 0;
  int maxRes = 0;

  {
    queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};

    buffer<int> dataBuf{data};
    buffer<int> sumBuf{&sumRes, 1};
    buffer<int> maxBuf{&maxRes, 1};

    ext::codeplay::experimental::fusion_wrapper fw{q};

    fw.start_fusion();
    iota(q, dataBuf, 0);

    q.submit([&](handler &cgh) {
      accessor in(dataBuf, cgh, read_only);
      auto sumRed = reduction(sumBuf, cgh, plus<>{},
                              property::reduction::initialize_to_identity{});
      detail::reduction_parallel_for<detail::auto_name, Strategy>(
          cgh, ndr, ext::oneapi::experimental::empty_properties_t{}, sumRed,
          [=](nd_item<1> Item, auto &Red) {
            Red.combine(in[Item.get_global_id()]);
          });
    });

    q.submit([&](handler &cgh) {
      accessor in(dataBuf, cgh, read_only);
      auto maxRed = reduction(maxBuf, cgh, maximum<>{},
                              property::reduction::initialize_to_identity{});
      detail::reduction_parallel_for<detail::auto_name, Strategy>(
          cgh, ndr, ext::oneapi::experimental::empty_properties_t{}, maxRed,
          [=](nd_item<1> Item, auto &Red) {
            Red.combine(in[Item.get_global_id()]);
          });
    });

    fw.complete_fusion(ext::codeplay::experimental::property::no_barriers{});
  }

  constexpr int expectedMax = globalSize - 1;
  constexpr int expectedSum = globalSize * expectedMax / 2;

  assert(sumRes == expectedSum);
  assert(maxRes == expectedMax);
}

template <detail::reduction::strategy... strategies>
void test_strategies(
    std::integer_sequence<detail::reduction::strategy, strategies...>,
    size_t localSize) {
  ((test<strategies>({globalSize, localSize})), ...);
}

int main() {
  constexpr std::array<std::size_t, 3> localSizes{
      globalSize /*Test single work-group*/,
      globalSize / 32 /*Test middle-sized work-group*/,
      1 /*Test single item work-groups*/};
  for (size_t localSize : localSizes) {
    test_strategies(
        std::integer_sequence<
            detail::reduction::strategy,
            detail::reduction::strategy::group_reduce_and_last_wg_detection,
            detail::reduction::strategy::local_atomic_and_atomic_cross_wg,
            detail::reduction::strategy::range_basic,
            detail::reduction::strategy::group_reduce_and_atomic_cross_wg,
            detail::reduction::strategy::local_mem_tree_and_atomic_cross_wg,
            detail::reduction::strategy::group_reduce_and_multiple_kernels,
            detail::reduction::strategy::basic,
            detail::reduction::strategy::multi>{},
        localSize);
  }
}
