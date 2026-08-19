#include "testsetup.hpp"
#include <py4dgeo/epoch.hpp>
#include <py4dgeo/kdtree.hpp>
#include <py4dgeo/py4dgeo.hpp>
#include <py4dgeo/searchtree.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>

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
    RadiusSearchResult result;

    // Do radius search with radius wide enough to cover the entire cloud
    auto num = tree.radius_search(o.data(), 100.0, result);
    REQUIRE(num == epoch.cloud.rows());
    REQUIRE(result.size() == epoch.cloud.rows());
  }

  SECTION("Perform radius search with distances")
  {
    // Find all nodes with a radius search
    std::array<double, 3> o{ 0.0, 0.0, 0.0 };
    RadiusSearchDistanceResult result;

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
    NearestNeighborsDistanceResult result;
    int k = 5;
    tree.nearest_neighbors_with_distances(epoch.cloud, result, k);
    REQUIRE(result.size() == epoch.cloud.rows());
    REQUIRE(result[0].first.size() == k);
    REQUIRE(result[0].first[k - 1] > 0);
  }

  SECTION("Nearest neighbor search:")
  {
    NearestNeighborsResult result;
    int k = 5;
    tree.nearest_neighbors(epoch.cloud, result, k);
    REQUIRE(result.size() == epoch.cloud.rows());
    REQUIRE(result[0].size() == k);
    REQUIRE(result[0][k - 1] > 0);
  }
}
