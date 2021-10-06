#include "catch2/catch.hpp"
#include "py4dgeo/py4dgeo.hpp"
#include "testsetup.hpp"

#include <vector>

using namespace py4dgeo;

TEST_CASE("M3C2 distance calculation", "[compute]")
{
  // Get a test cloud
  auto cloud = testcloud();

  // Get a KDTree
  auto kdtree = KDTree::create(cloud);
  kdtree.build_tree(10);
  kdtree.precompute(cloud, 10.0);

  std::vector<double> scales{ 1.0 };
  EigenPointCloud directions(cloud.rows(), 3);

  // Precompute the multiscale directions
  compute_multiscale_directions(cloud, cloud, scales, kdtree, directions);

  SECTION("Distance calculation with standard radius search")
  {
    // Calculate the distances
    EigenVector distances(cloud.rows(), 1);
    compute_distances(
      cloud, 1.0, cloud, kdtree, cloud, kdtree, directions, 0.0, distances);

    for (IndexType i = 0; i < distances.rows(); ++i)
      REQUIRE(std::abs(distances.row(i).norm()) < 1e-8);
  }

  SECTION("Distance calculation with cylinder search")
  {
    // Calculate the distances
    EigenVector distances(cloud.rows(), 1);
    compute_distances(
      cloud, 1.0, cloud, kdtree, cloud, kdtree, directions, 2.0, distances);

    for (IndexType i = 0; i < distances.rows(); ++i)
      REQUIRE(std::abs(distances.row(i).norm()) < 1e-8);
  }
}