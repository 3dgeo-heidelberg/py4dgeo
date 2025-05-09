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
  Epoch::set_default_radius_search_tree(SearchTree::KDTree);
  epoch.kdtree.build_tree(10);

  EigenNormalSet directions(1, 3);
  directions << 0, 0, 1;

  WorkingSetFinderParameters params{ epoch,
                                     1.0,
                                     corepoints->row(0),
                                     directions,
                                     static_cast<double>(state.range(0)) };

  for (auto _ : state) {
    auto points = cylinder_workingset_finder(params);
  }
}

BENCHMARK(cylindersearch_benchmark)
  ->Unit(benchmark::kMicrosecond)
  ->DenseRange(2.0, 8.0, 1.0);
BENCHMARK_MAIN();
