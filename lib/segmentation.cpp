#include "py4dgeo/segmentation.hpp"

#include <cmath>
#include <limits>
#include <unordered_set>
#include <vector>

namespace py4dgeo {

ObjectByChange
region_growing(const RegionGrowingAlgorithmData& data)
{
  // Instantiate a set of candidates to check
  std::unordered_set<IndexType> candidates;
  candidates.insert(data.seed.index);

  // Create the return object
  ObjectByChange obj;
  obj.start_epoch = data.seed.start_epoch;
  obj.end_epoch = data.seed.end_epoch;

  // Grow while we have candidates
  while (!candidates.empty()) {
    // Get one element to process and remove it from the candidates
    auto candidate = *candidates.begin();
    candidates.erase(candidates.begin());

    // Calculate distance
    TimeseriesDistanceFunctionData distance_data{
      data.distances.row(data.seed.index), data.distances.row(candidate)
    };
    double distance = dtw_distance(distance_data);

    // Apply thresholding
    if (distance < data.thresholds[0]) {
      // Use this point in the grown region
      obj.indices.insert(candidate);

      // Add neighboring corepoints to list of candidates
      KDTree::RadiusSearchResult neighbors;
      data.corepoints.kdtree.radius_search(
        &data.corepoints.cloud(candidate, 0), data.radius, neighbors);
      for (auto n : neighbors) {
        // Check whether the corepoint is already present among candidates or
        // result. If not this is a new candidate
        if ((candidates.find(n) == candidates.end()) &&
            (obj.indices.find(n) == obj.indices.end()))
          candidates.insert(n);
      }
    }
  }

  return obj;
}

inline double
distance(double x, double y)
{
  return std::fabs(x - y);
}

double
dtw_distance(const TimeseriesDistanceFunctionData& data)
{
  // Create an index vector of non-NaN values
  std::vector<IndexType> indices;
  indices.reserve(data.ts1.size());
  for (IndexType i = 0; i < data.ts1.size(); ++i)
    if (!(std::isnan(data.ts1[i]) || (std::isnan(data.ts2[i]))))
      indices.push_back(i);

  // If all values were NaN, our distance is NaN
  if (indices.empty())
    return std::numeric_limits<double>::quiet_NaN();

  const auto n = indices.size();
  std::vector<std::vector<double>> d(n, std::vector<double>(n));

  // Upper left corner
  d[0][0] = distance(data.ts1[indices[0]], data.ts2[indices[0]]);

  // Upper row and left-most column
  for (std::size_t i = 1; i < n; ++i) {
    d[i][0] =
      distance(data.ts1[indices[i]], data.ts2[indices[0]]) + d[i - 1][0];
    d[0][i] =
      distance(data.ts1[indices[0]], data.ts2[indices[i]]) + d[0][i - 1];
  }

  // Rest of the distance matrix
  for (std::size_t i = 1; i < n; ++i)
    for (std::size_t j = 1; j < n; ++j)
      d[i][j] = distance(data.ts1[indices[i]], data.ts2[indices[j]]) +
                std::fmin(std::fmin(d[i - 1][j], d[i][j - 1]), d[i - 1][j - 1]);

  return d[n - 1][n - 1];
}

double
normalized_dtw_distance(const TimeseriesDistanceFunctionData& data)
{
  // Calculate Dmax from the first timeseries
  double max_dist = 0.0;
  for (auto entry : data.ts1)
    if (!std::isnan(entry))
      max_dist += std::abs(entry);

  return std::fmin(1.0, 1.0 - (max_dist - dtw_distance(data)) / max_dist);
}

}