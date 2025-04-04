#include "catch2/catch.hpp"
#include "py4dgeo/epoch.hpp"
#include "py4dgeo/octree.hpp"
#include "py4dgeo/py4dgeo.hpp"
#include "testsetup.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace py4dgeo;

TEST_CASE("Octree is correctly build", "[octree]")
{
  // Get a test epoch
  auto [cloud, corepoints] = testcloud();
  Epoch epoch(*cloud);

  // Construct the Octree
  auto tree = Octree::create(epoch.cloud);
  tree.build_tree();

  SECTION("Perform radius search")
  {
    // Find all nodes with a radius search
    Eigen::Vector3d query_point{ 0.0, 0.0, 0.0 };
    Octree::RadiusSearchResult result;

    // Do radius search with radius wide enough to cover the entire cloud
    double radius = 100.;
    unsigned int level =
      epoch.octree.find_appropriate_level_for_radius_search(radius);
    auto num = tree.radius_search(query_point, radius, level, result);
    REQUIRE(num == epoch.cloud.rows());
    REQUIRE(result.size() == epoch.cloud.rows());
  }

  SECTION("Perform radius search with distances")
  {
    // Find all nodes with a radius search
    Eigen::Vector3d query_point{ 0.0, 0.0, 0.0 };
    Octree::RadiusSearchDistanceResult result;

    // Do radius search with radius wide enough to cover the entire cloud
    double radius = 100.;
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
