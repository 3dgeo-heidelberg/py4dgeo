#include "testsetup.hpp"

#include <fstream>
#include <sstream>
#include <vector>

using namespace py4dgeo;

EigenPointCloud
testcloud()
{
  std::vector<float> data;
  std::ifstream stream("../data/plane_horizontal_t1.xyz");
  std::string line;
  std::size_t points{ 0 };

  while (std::getline(stream, line)) {
    std::istringstream s(line);
    float x;
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
