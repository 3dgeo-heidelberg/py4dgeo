#include "py4dgeo/segmentation.hpp"

#include <vector>

namespace py4dgeo {

inline double
distance(double x, double y)
{
  return std::fabs(x - y);
}

double
dtw_distance(EigenTimeSeriesConstRef ts1, EigenTimeSeriesConstRef ts2)
{
  const auto n = ts1.size();
  std::vector<std::vector<double>> d(n, std::vector<double>(n));

  // Upper left corner
  d[0][0] = distance(ts1[0], ts2[0]);

  // Upper row and left-most column
  for (std::size_t i = 1; i < n; ++i) {
    d[i][0] = distance(ts1[i], ts2[0]) + d[i - 1][0];
    d[0][i] = distance(ts1[0], ts2[i]) + d[0][i - 1];
  }

  // Rest of the distance matrix
  for (std::size_t i = 1; i < n; ++i)
    for (std::size_t j = 1; j < n; ++j)
      d[i][j] = distance(ts1[i], ts2[j]) +
                std::fmin(std::fmin(d[i - 1][j], d[i][j - 1]), d[i - 1][j - 1]);

  return d[n - 1][n - 1];
}

}