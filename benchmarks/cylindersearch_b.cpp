#include "testsetup.hpp"

#include <py4dgeo/compute.hpp>
#include <py4dgeo/epoch.hpp>

#include <benchmark/benchmark.h>

using namespace py4dgeo;

static void
cylindersearch_benchmark(benchmark::State& state)
{
  auto [cloud, corepoints] = ahk_benchcloud();
  Epoch epoch(*cloud);
  epoch.kdtree.build_tree(10);
  epoch.kdtree.precompute(*corepoints, 2.0, MemoryPolicy::MINIMAL);

  EigenNormalSet directions(1, 3);
  directions << 0, 0, 1;

  for (auto _ : state) {
    auto points = cylinder_workingset_finder(
      epoch, 1.0, corepoints->row(0), directions, 3.0, 0);
  }
}

BENCHMARK(cylindersearch_benchmark)->Unit(benchmark::kMicrosecond);
BENCHMARK_MAIN();
