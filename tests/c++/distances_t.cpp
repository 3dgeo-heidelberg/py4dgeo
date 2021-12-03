#include "catch2/catch.hpp"
#include "py4dgeo/compute.hpp"
#include "py4dgeo/kdtree.hpp"
#include "py4dgeo/py4dgeo.hpp"
#include "testsetup.hpp"

#include <vector>

using namespace py4dgeo;

TEST_CASE("M3C2 distance calculation", "[compute]")
{
  // Get a test epoch
  auto [cloud, corepoints] = testcloud();
  Epoch epoch(*cloud);
  epoch.kdtree.build_tree(10);
  epoch.kdtree.precompute(epoch.cloud, 10.0, MemoryPolicy::COREPOINTS);

  std::vector<double> scales{ 3.0 };
  EigenNormalSet directions(epoch.cloud.rows(), 3);
  EigenNormalSet orientation(1, 3);
  orientation << 0, 0, 1;

  // Precompute the multiscale directions
  compute_multiscale_directions(
    epoch, *corepoints, scales, orientation, directions);

  // Calculate the distances
  DistanceVector distances;
  UncertaintyVector uncertainties;

  // We try to test all callback combinations
  auto wsfinder =
    GENERATE(radius_workingset_finder, cylinder_workingset_finder);
  auto uncertaintymeasure =
    GENERATE(no_uncertainty, standard_deviation_uncertainty);

  compute_distances(epoch.cloud,
                    2.0,
                    epoch,
                    epoch,
                    directions,
                    0.0,
                    distances,
                    uncertainties,
                    wsfinder,
                    uncertaintymeasure);

  REQUIRE(distances.size() == epoch.cloud.rows());
  REQUIRE(uncertainties.size() == epoch.cloud.rows());

  for (std::size_t i = 0; i < distances.size(); ++i)
    REQUIRE(std::abs(distances[i]) < 1e-8);
}

TEST_CASE("Single-direction M3C2 distance calculation", "[compute]")
{
  // Get a test epoch
  auto [cloud, corepoints] = testcloud();
  Epoch epoch(*cloud);
  epoch.kdtree.build_tree(10);
  epoch.kdtree.precompute(epoch.cloud, 10.0, MemoryPolicy::COREPOINTS);

  // Single distance vector
  EigenNormalSet directions(1, 3);
  directions << 0, 0, 1;

  // Calculate the distances
  DistanceVector distances;
  UncertaintyVector uncertainties;

  // We try to test all callback combinations
  auto wsfinder =
    GENERATE(radius_workingset_finder, cylinder_workingset_finder);
  auto uncertaintymeasure =
    GENERATE(no_uncertainty, standard_deviation_uncertainty);

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

  for (std::size_t i = 0; i < distances.size(); ++i)
    REQUIRE(std::abs(distances[i]) < 1e-8);
}

TEST_CASE("Cylinder Search Correctness", "[compute]")
{
  auto [cloud, corepoints] = testcloud();
  Epoch epoch(*cloud);
  epoch.kdtree.build_tree(10);
  epoch.kdtree.precompute(epoch.cloud, 10.0, MemoryPolicy::MINIMAL);

  EigenPointCloud corepoint(1, 3);
  corepoint << 10, 10, 0;

  EigenNormalSet normal(1, 3);
  normal << 0.70710678, 0.70710678, 0.0;

  auto cyl = cylinder_workingset_finder(
    epoch, 1.0, corepoint.row(0), normal.row(0), 5.0, 0);

  REQUIRE(cyl.rows() == 23);
}