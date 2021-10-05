#include "Eigen/Eigen"
#include "catch2/catch.hpp"
#include "py4dgeo/py4dgeo.hpp"
#include "testsetup.hpp"

#include <vector>

using namespace py4dgeo;

TEST_CASE("M3C2 Multiscale direction calculation", "[compute]")
{
  // Get a test cloud
  auto cloud = testcloud();

  // Get a KDTree
  auto kdtree = KDTree::create(cloud);
  kdtree.build_tree(10);
  kdtree.precompute(cloud, 10.0);

  std::vector<double> scales{ 1.0, 2.0, 3.0 };
  EigenPointCloud result(cloud.rows(), 3);
  REQUIRE(result.rows() == 441);

  // Do the calculation
  compute_multiscale_directions(cloud, cloud, scales, kdtree, result);

  for (IndexType i = 0; i < result.rows(); ++i)
    REQUIRE(std::abs(result.row(i).norm() - 1.0) < 1e-8);
}