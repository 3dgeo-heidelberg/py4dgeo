#include "testsetup.hpp"

#include <py4dgeo/compute.hpp>
#include <py4dgeo/epoch.hpp>
#include <py4dgeo/searchtree.hpp>

#include <benchmark/benchmark.h>

using namespace py4dgeo;

static void
multiscale_normal_benchmark_kdtree(benchmark::State& state)
{
  auto [cloud, corepoints] = ahk_benchcloud();
  Epoch epoch(*cloud);
  Epoch::set_default_radius_search_tree(SearchTree::KDTree);
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

static void
multiscale_normal_benchmark_octree(benchmark::State& state)
{
  auto [cloud, corepoints] = ahk_benchcloud();
  Epoch epoch(*cloud);
  Epoch::set_default_radius_search_tree(SearchTree::Octree);
  epoch.octree.build_tree();
  std::vector<double> radii{ 0.1, 0.5, 1.0, 2.0, 5.0 };
  std::vector<double> used_radii;
  EigenNormalSet directions(corepoints->rows(), 3);
  EigenNormalSet orientation(1, 3);
  orientation << 0, 0, 1;

  for (auto _ : state) {
    compute_multiscale_directions(
      epoch, *corepoints, radii, orientation, directions, used_radii);
  }
}

BENCHMARK(multiscale_normal_benchmark_kdtree)->Unit(benchmark::kMicrosecond);
BENCHMARK(multiscale_normal_benchmark_octree)->Unit(benchmark::kMicrosecond);
BENCHMARK_MAIN();
