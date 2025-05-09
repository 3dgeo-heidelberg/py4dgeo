#include "testsetup.hpp"

#include <py4dgeo/compute.hpp>
#include <py4dgeo/epoch.hpp>

#include <benchmark/benchmark.h>

using namespace py4dgeo;

static void
distances_benchmark(benchmark::State& state)
{
  auto [cloud, corepoints] = ahk_benchcloud();
  Epoch epoch(*cloud);
  Epoch::set_default_radius_search_tree(SearchTree::KDTree);
  epoch.kdtree.build_tree(10);
  std::vector<double> normal_radii{ 1.0 };
  std::vector<double> used_radii;
  EigenNormalSet directions(corepoints->rows(), 3);
  EigenNormalSet orientation(1, 3);
  orientation << 0, 0, 1;

  // Precompute the multiscale directions
  compute_multiscale_directions(
    epoch, *corepoints, normal_radii, orientation, directions, used_radii);

  // We try to test all callback combinations
  auto wsfinder = radius_workingset_finder;
  auto distancecalc = mean_stddev_distance;

  for (auto _ : state) {
    // Calculate the distances
    DistanceVector distances;
    UncertaintyVector uncertainties;

    compute_distances(*corepoints,
                      2.0,
                      epoch,
                      epoch,
                      directions,
                      0.0,
                      0.0,
                      distances,
                      uncertainties,
                      wsfinder,
                      distancecalc);
  }
}

BENCHMARK(distances_benchmark)->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();
