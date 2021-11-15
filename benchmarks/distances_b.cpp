#include "bench.hpp"
#include <py4dgeo/compute.hpp>
#include <py4dgeo/epoch.hpp>

static void
distances_benchmark(benchmark::State& state)
{
  auto [cloud, corepoints] = ahk_benchcloud();
  std::cout << cloud->rows() << std::endl;
  Epoch epoch(*cloud);
  epoch.kdtree.build_tree(10);
  epoch.kdtree.precompute(*corepoints, 10.0, MemoryPolicy::MINIMAL);

  std::vector<double> scales{ 1.0 };
  EigenPointCloud directions(corepoints->rows(), 3);
  EigenPointCloud orientation(1, 3);
  orientation << 0, 0, 1;

  // Precompute the multiscale directions
  compute_multiscale_directions(
    epoch, *corepoints, scales, orientation, directions);

  // We try to test all callback combinations
  auto wsfinder = radius_workingset_finder;
  auto uncertaintymeasure = standard_deviation_uncertainty;

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
                      distances,
                      uncertainties,
                      wsfinder,
                      uncertaintymeasure);
  }
}

BENCHMARK(distances_benchmark);
