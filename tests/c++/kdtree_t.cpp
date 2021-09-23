#include "catch2/catch.hpp"
#include "py4dgeo/py4dgeo.hpp"
#include "testsetup.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace py4dgeo;

TEST_CASE("KDTree is correctly build", "[kdtree]")
{
  // Get the test cloud
  auto cloud = testcloud();

  // Construct the KDTree
  auto tree = KDTree::create(cloud);
  tree.build_tree(10);

  SECTION("Perform radius search")
  {
    // Find all nodes with a radius search
    std::array<double, 3> o{ 0.0, 0.0, 0.0 };
    KDTree::RadiusSearchResult result;

    // Do radius search with radius wide enough to cover the entire cloud
    auto num = tree.radius_search(o.data(), 100.0, result);
    REQUIRE(num == cloud.rows());
    REQUIRE(result.size() == cloud.rows());
  }

  SECTION("Precomputation and radius search")
  {
    EigenPointCloud qp = cloud(Eigen::seq(0, 20), Eigen::all);
    tree.precompute(qp, 20.0);

    for (std::size_t i = 0; i < 20; ++i) {
      KDTree::RadiusSearchResult result1, result2;
      tree.radius_search(&qp(i, 0), 5.0, result1);
      tree.precomputed_radius_search(i, 5.0, result2);
      REQUIRE(result1.size() == result2.size());
    }
  }

  SECTION("Serialize + deserialize and do radius search")
  {
    // Serialize and deserialize the KDTree
    std::stringstream ss;
    tree.to_stream(ss);
    auto deserialized = KDTree::from_stream(ss);

    // Find all nodes with a radius search
    std::array<double, 3> o{ 0.0, 0.0, 0.0 };
    KDTree::RadiusSearchResult result;

    // Do radius search with radius wide enough to cover the entire cloud
    auto num = deserialized->radius_search(o.data(), 100.0, result);
    REQUIRE(num == cloud.rows());
    REQUIRE(result.size() == cloud.rows());
  }
}
