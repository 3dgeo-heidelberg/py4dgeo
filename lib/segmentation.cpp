#include "py4dgeo/segmentation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py4dgeo {

ObjectByChange
region_growing(const RegionGrowingAlgorithmData& data,
               const TimeseriesDistanceFunction& distance_function)
{
  // Ensure that we have a sorted list of thresholds
  std::vector<double> sorted_thresholds(data.thresholds);
  std::sort(sorted_thresholds.begin(), sorted_thresholds.end());

  // Instantiate a set of candidates to check
  std::unordered_set<IndexType> candidates;

  // Keep a list of calculated distances that we did not need so far
  // as we might need them for later threshold levels
  std::unordered_map<IndexType, double> calculated_distances;

  // We throw the initial seed into the calculated distances
  // to kick off the calculation in below loop
  calculated_distances[data.seed.index] = 0.0;

  // Create the return object
  ObjectByChange obj;
  obj.start_epoch = data.seed.start_epoch;
  obj.end_epoch = data.seed.end_epoch;
  obj.threshold = sorted_thresholds[0];

  // Store a ratio value to compare against for premature termination
  double last_ratio = 0.5;

  for (auto threshold : sorted_thresholds) {
    // The additional points found at this threshold level. These will
    // be added to return object after deciding whether the adaptive
    // procedure should continue.
    std::unordered_set<IndexType> additional_points;

    // Make all already calculated distances below our threshold
    // candidates and start the candidate loop
    auto distit = calculated_distances.begin();
    while (distit != calculated_distances.end()) {
      if (distit->second < threshold) {
        candidates.insert(distit->first);
        distit = calculated_distances.erase(distit);
      } else
        ++distit;
    }

    // Grow while we have candidates
    while (!candidates.empty()) {
      // Get one element to process and remove it from the candidates
      auto candidate = *candidates.begin();
      candidates.erase(candidates.begin());

      // Calculate distance - may already be calculated
      auto distit = calculated_distances.find(candidate);
      double distance;
      if (distit == calculated_distances.end()) {
        TimeseriesDistanceFunctionData distance_data{
          data.distances.row(data.seed.index), data.distances.row(candidate)
        };
        distance = distance_function(distance_data);
      } else {
        distance = distit->second;
        calculated_distances.erase(distit);
      }

      // Apply thresholding
      if (distance < threshold) {
        // Use this point in the grown region
        additional_points.insert(candidate);

        // Add neighboring corepoints to list of candidates
        KDTree::RadiusSearchResult neighbors;
        data.corepoints.kdtree.radius_search(
          &data.corepoints.cloud(candidate, 0), data.radius, neighbors);
        for (auto n : neighbors) {
          // Check whether the corepoint is already present among candidates,
          // the final result or the points added at this threshold level.
          // If none of that match, this is a new candidate
          if ((candidates.find(n) == candidates.end()) &&
              (additional_points.find(n) == additional_points.end()) &&
              (obj.indices.find(n) == obj.indices.end()))
            candidates.insert(n);
        }
      } else {
        // Store the calculated distance in case we need it again
        calculated_distances[candidate] = distance;
      }
    }

    // Determine whether this is the final threshold level or we need
    // to continue to the next threshold level
    double new_ratio =
      static_cast<double>(obj.indices.size()) /
      static_cast<double>(obj.indices.size() + additional_points.size());
    if (new_ratio < last_ratio) {
      // If this is using the strictest of all thresholds, we need to
      // add the points here.
      if (obj.indices.empty())
        obj.indices.merge(additional_points);

      return obj;
    }

    // If not, we now need to move all additional points into obj
    last_ratio = new_ratio;
    obj.threshold = threshold;
    obj.indices.merge(additional_points);
  }

  // If we came up to here, a local maximum was not produced.
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
