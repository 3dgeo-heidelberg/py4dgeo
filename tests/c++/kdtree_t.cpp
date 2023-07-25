#include "catch2/catch.hpp"
#include "py4dgeo/epoch.hpp"
#include "py4dgeo/kdtree.hpp"
#include "py4dgeo/py4dgeo.hpp"
#include "testsetup.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace py4dgeo;

TEST_CASE("KDTree is correctly build", "[kdtree]")
{
  // Get a test epoch
  auto [cloud, corepoints] = testcloud();
  Epoch epoch(*cloud);

  // Construct the KDTree
  auto tree = KDTree::create(epoch.cloud);
  tree.build_tree(10);

  SECTION("Perform radius search")
  {
    // Find all nodes with a radius search
    std::array<double, 3> o{ 0.0, 0.0, 0.0 };
    KDTree::RadiusSearchResult result;

    // Do radius search with radius wide enough to cover the entire cloud
    auto num = tree.radius_search(o.data(), 100.0, result);
    REQUIRE(num == epoch.cloud.rows());
    REQUIRE(result.size() == epoch.cloud.rows());
  }

  SECTION("Perform radius search with distances")
  {
    // Find all nodes with a radius search
    std::array<double, 3> o{ 0.0, 0.0, 0.0 };
    KDTree::RadiusSearchDistanceResult result;

    // Do radius search with radius wide enough to cover the entire cloud
    auto num = tree.radius_search_with_distances(o.data(), 100.0, result);
    REQUIRE(num == epoch.cloud.rows());
    REQUIRE(result.size() == epoch.cloud.rows());
    REQUIRE(std::is_sorted(result.begin(), result.end(), [](auto a, auto b) {
      return a.second < b.second;
    }));
  }

  SECTION("Nearest neighbor search with distances")
  {
    std::pair<std::vector<IndexType>, std::vector<double>> result;
    tree.nearest_neighbors_with_distances(epoch.cloud, result);
  }
}
