#include "testsetup.hpp"
#include <py4dgeo/compute.hpp>
#include <py4dgeo/epoch.hpp>
#include <py4dgeo/kdtree.hpp>
#include <py4dgeo/py4dgeo.hpp>
#include <py4dgeo/searchtree.hpp>

#include <Eigen/Eigen>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

using namespace py4dgeo;

TEST_CASE("M3C2 Multiscale direction calculation", "[compute]")
{
  // Get a test epoch
  auto [cloud, corepoints] = testcloud();
  Epoch epoch(*cloud);
  Epoch::set_default_radius_search_tree(SearchTree::KDTree);
  epoch.kdtree.build_tree(10);

  std::vector<double> normal_radii{ 1.0, 2.0, 3.0 };
  EigenNormalSet result(epoch.cloud.rows(), 3);
  std::vector<double> used_radii;
  EigenNormalSet orientation(1, 3);
  orientation << 0, 0, 1;

  REQUIRE(result.rows() == 441);

  // Do the calculation
  compute_multiscale_directions(
    epoch, *corepoints, normal_radii, orientation, result, used_radii);

  for (IndexType i = 0; i < result.rows(); ++i)
    REQUIRE(std::abs(result.row(i).norm() - 1.0) < 1e-8);
}
