#include "testsetup.hpp"

#include <py4dgeo/compute.hpp>
#include <py4dgeo/epoch.hpp>

#include <benchmark/benchmark.h>

using namespace py4dgeo;

static void
multiscale_normal_benchmark(benchmark::State& state)
{
  auto [cloud, corepoints] = ahk_benchcloud();
  Epoch epoch(*cloud);
  epoch.kdtree.build_tree(10);
  std::vector<double> normal_radii{ 0.1, 0.5, 1.0, 2.0, 5.0 };
  std::vector<double> used_radii;
  EigenNormalSet directions(corepoints->rows(), 3);
  EigenNormalSet orientation(1, 3);
  orientation << 0, 0, 1;

  for (auto _ : state) {
    // Precompute the multiscale directions
    compute_multiscale_directions(
      epoch, *corepoints, normal_radii, orientation, directions, used_radii);
  }
}

BENCHMARK(multiscale_normal_benchmark)->Unit(benchmark::kMicrosecond);
BENCHMARK_MAIN();
