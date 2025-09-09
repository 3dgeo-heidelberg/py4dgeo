#include "testsetup.hpp"
#include <py4dgeo/epoch.hpp>
#include <py4dgeo/octree.hpp>
#include <py4dgeo/searchtree.hpp>

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>

using namespace py4dgeo;

TEST_CASE("Octree is correctly build", "[octree]")
{
  // Get a test epoch
  auto [cloud, corepoints] = testcloud();
  Epoch epoch(*cloud);

  constexpr double tol = 1.e-6;

  SECTION("Build cubic, no corner")
  {
    // Construct the Octree
    auto tree = Octree::create(epoch.cloud);
    tree.build_tree(true);

    Eigen::Vector3d extent = tree.get_max_point() - tree.get_min_point();

    REQUIRE(std::abs(extent.x() - extent.y()) < tol);
    REQUIRE(std::abs(extent.x() - extent.z()) < tol);
    REQUIRE(std::abs(extent.y() - extent.z()) < tol);
  }
  SECTION("Build cubic, min corner")
  {
    // Construct the Octree
    auto tree = Octree::create(epoch.cloud);
    tree.build_tree(true, cloud->colwise().minCoeff());

    Eigen::Vector3d extent = tree.get_max_point() - tree.get_min_point();

    REQUIRE(std::abs(extent.x() - extent.y()) < tol);
    REQUIRE(std::abs(extent.x() - extent.z()) < tol);
    REQUIRE(std::abs(extent.y() - extent.z()) < tol);
  }
  SECTION("Build cubic, max corner")
  {
    // Construct the Octree
    auto tree = Octree::create(epoch.cloud);
    tree.build_tree(true, std::nullopt, cloud->colwise().maxCoeff());

    Eigen::Vector3d extent = tree.get_max_point() - tree.get_min_point();

    REQUIRE(std::abs(extent.x() - extent.y()) < tol);
    REQUIRE(std::abs(extent.x() - extent.z()) < tol);
    REQUIRE(std::abs(extent.y() - extent.z()) < tol);
  }

  // Construct the Octree
  auto tree = Octree::create(epoch.cloud);
  tree.build_tree();

  // Do radius search with radius wide enough to cover the entire cloud
  constexpr double radius = 100.;

  SECTION("Perform radius search")
  {
    // Find all nodes with a radius search
    Eigen::Vector3d query_point{ 0.0, 0.0, 0.0 };
    RadiusSearchResult result;

    for (unsigned int level = 0; level < 7; ++level) {
      auto num = tree.radius_search(query_point, radius, level, result);
      REQUIRE(num == epoch.cloud.rows());
      REQUIRE(result.size() == epoch.cloud.rows());
    }
  }

  SECTION("Perform radius search with distances")
  {
    // Find all nodes with a radius search
    Eigen::Vector3d query_point{ 0.0, 0.0, 0.0 };
    RadiusSearchDistanceResult result;

    unsigned int level =
      epoch.octree.find_appropriate_level_for_radius_search(radius);
    auto num =
      tree.radius_search_with_distances(query_point, radius, level, result);
    REQUIRE(num == epoch.cloud.rows());
    REQUIRE(result.size() == epoch.cloud.rows());
    REQUIRE(std::is_sorted(result.begin(), result.end(), [](auto a, auto b) {
      return a.second < b.second;
    }));
  }
}
