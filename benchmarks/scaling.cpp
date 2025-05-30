#include <benchmark/benchmark.h>

#ifdef PY4DGEO_WITH_OPENMP

#include "testsetup.hpp"

#include <omp.h>

#include <py4dgeo/compute.hpp>
#include <py4dgeo/epoch.hpp>

using namespace py4dgeo;

static void
scalability_benchmark(benchmark::State& state)
{
  auto [cloud, corepoints] = ahk_benchcloud();
  Epoch epoch(*cloud);
  Epoch::set_default_radius_search_tree(SearchTree::KDTree);
  epoch.kdtree.build_tree(10);

  for (auto _ : state) {
    // Set the number of threads according to benchmark state
    omp_set_num_threads(state.range(0));

    std::vector<double> normal_radii{ 1.0 };
    EigenNormalSet directions(corepoints->rows(), 3);
    std::vector<double> used_radii;
    EigenNormalSet orientation(1, 3);
    orientation << 0, 0, 1;

    // Precompute the multiscale directions
    compute_multiscale_directions(
      epoch, *corepoints, normal_radii, orientation, directions, used_radii);

    // We try to test all callback combinations
    auto wsfinder = radius_workingset_finder;
    auto distancecalc = mean_stddev_distance;

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
  state.SetComplexityN(state.range(0));
}

BENCHMARK(scalability_benchmark)
  ->Unit(benchmark::kMillisecond)
  ->DenseRange(1, omp_get_max_threads(), 1)
  ->Complexity();

#endif // PY4DGEO_WITH_OPENMP

BENCHMARK_MAIN();
