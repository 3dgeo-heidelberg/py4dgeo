#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "py4dgeo/py4dgeo.hpp"

#include "tests.hpp"

using namespace py4dgeo;

EigenPointCloud
testcloud()
{
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
  EigenPointCloud cloud(points, 3);
  std::copy(data.data(), data.data() + points * 3, &cloud(0, 0));
  return cloud;
}