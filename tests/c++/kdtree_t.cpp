#include "catch2/catch.hpp"
#include "py4dgeo/py4dgeo.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace py4dgeo;

TEST_CASE("KDTree is correctly build", "[kdtree]")
{
  // Load a test case
  std::vector<double> data;
  std::ifstream stream("../data/plane_horizontal_t1.xyz");
  std::string line;
  std::size_t points{ 0 };

  while (std::getline(stream, line)) {
    std::istringstream s(line);
    double x;
    for (int i = 0; i < 3; ++i) {
      s >> x;
      data.push_back(x);
    }
    ++points;
  }

  // Interpret the given data as an Eigen matrix
  Eigen::Map<EigenPointCloud> cloud(data.data(), points, 3);

  // Construct the KDTree
  auto tree = KDTree::create(cloud);
  tree.build_tree(10);

  // Find all nodes with a radius search
  std::array<double, 3> o{ 0.0, 0.0, 0.0 };
  KDTree::RadiusSearchResult result;

  // Do radius search with radius wide enough to cover the entire cloud
  auto num = tree.radius_search(o.data(), 100.0, result);
  REQUIRE(num == points);
  REQUIRE(result.size() == points);

  Eigen::Map<EigenPointCloud> qp(data.data(), 20, 3);
  tree.precompute(qp, 20.0);

  for (std::size_t i = 0; i < 20; ++i) {
    KDTree::RadiusSearchResult result1, result2;
    tree.radius_search(&qp(i, 0), 5.0, result1);
    tree.precomputed_radius_search(i, 5.0, result2);
    REQUIRE(result1.size() == result2.size());
  }
}
