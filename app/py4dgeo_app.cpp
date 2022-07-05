#include "py4dgeo/py4dgeo.hpp"

#include <fstream>
#include <vector>

using namespace py4dgeo;

int
main()
{
  // Load a test case
  std::vector<double> data;
  std::ifstream stream("../tests/data/plane_horizontal_t1.xyz");
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
}
