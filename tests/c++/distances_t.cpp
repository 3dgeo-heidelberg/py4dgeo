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

  std::vector<double> normal_radii{ 3.0 };
  std::vector<double> used_radii;
  EigenNormalSet directions(epoch.cloud.rows(), 3);
  EigenNormalSet orientation(1, 3);
  orientation << 0, 0, 1;

  // Precompute the multiscale directions
  compute_multiscale_directions(
    epoch, *corepoints, normal_radii, orientation, directions, used_radii);

  // Calculate the distances
  DistanceVector distances;
  UncertaintyVector uncertainties;

  // We try to test all callback combinations
  auto wsfinder =
    GENERATE(radius_workingset_finder, cylinder_workingset_finder);
  auto distancecalc = GENERATE(mean_stddev_distance, median_iqr_distance);

  compute_distances(epoch.cloud,
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
  Epoch::set_default_radius_search_tree(SearchTree::KDTree);
  epoch.kdtree.build_tree(10);

  // Single distance vector
  EigenNormalSet directions(1, 3);
  directions << 0, 0, 1;

  // Calculate the distances
  DistanceVector distances;
  UncertaintyVector uncertainties;

  // We try to test all callback combinations
  auto wsfinder =
    GENERATE(radius_workingset_finder, cylinder_workingset_finder);
  auto distancecalc = GENERATE(mean_stddev_distance, median_iqr_distance);

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

  for (std::size_t i = 0; i < distances.size(); ++i)
    REQUIRE(std::abs(distances[i]) < 1e-8);
}

TEST_CASE("Cylinder Search Correctness", "[compute]")
{
  auto [cloud, corepoints] = testcloud();
  Epoch epoch(*cloud);
  Epoch::set_default_radius_search_tree(SearchTree::KDTree);
  epoch.kdtree.build_tree(10);

  EigenPointCloud corepoint(1, 3);
  corepoint << 10, 10, 0;

  EigenNormalSet normal(1, 3);
  normal << 0.70710678, 0.70710678, 0.0;

  WorkingSetFinderParameters params{
    epoch, 1.0, corepoint.row(0), normal.row(0), 5.0
  };
  auto cyl = cylinder_workingset_finder(params);

  REQUIRE(cyl.rows() == 23);

  for (IndexType i = 0; i < cyl.rows(); ++i) {
    auto to_midpoint =
      cyl.cast<double>().row(i) - corepoint.cast<double>().row(0);
    auto to_midpoint_plane = (to_midpoint * normal.row(0).transpose()).eval();
    auto to_axis2 =
      (to_midpoint - to_midpoint_plane * normal).rowwise().squaredNorm().eval();

    REQUIRE(to_axis2(0, 0) <= 1.0);
    REQUIRE(std::abs(to_midpoint_plane(0, 0)) <= 5.0);
  }
}
