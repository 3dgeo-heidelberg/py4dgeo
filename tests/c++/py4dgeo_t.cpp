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

  // Construct the KDTree
  KDTree tree(data.data(), points);
  tree.build_tree(10);

  // Find all nodes with a radius search
  std::array<double, 3> o{ 0.0, 0.0, 0.0 };
  std::vector<std::pair<std::size_t, double>> result;

  auto num = tree.radius_search(o.data(), 100.0, result);

  REQUIRE(num == points);
  REQUIRE(result.size() == points);
}
