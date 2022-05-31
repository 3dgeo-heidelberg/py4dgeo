#include "py4dgeo/segmentation.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <set>
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
  std::multimap<double, IndexType> calculated_distances;
  std::set<double, std::greater<double>> used_distances;

  // We add the initial seed to the candidates in order to kick off
  // the calculation in the loop below
  candidates.insert(data.seed.index);
  calculated_distances.insert({ 0.0, data.seed.index });

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
    std::set<double, std::greater<double>> with_additional_distances(
      used_distances);

    // Grow while we have candidates
    while ((!candidates.empty()) &&
           (calculated_distances.begin()->first < threshold)) {
      // Get one element to process and remove it from the candidates
      auto [distance, candidate] = *calculated_distances.begin();
      calculated_distances.erase(calculated_distances.begin());
      candidates.erase(candidate);

      // Use this point in the grown region
      additional_points.insert(candidate);
      with_additional_distances.insert(distance);

      // Maybe adapt the threshold to a percentile criterium
      if (with_additional_distances.size() >= data.min_segments) {
        auto it = with_additional_distances.begin();
        std::advance(
          it,
          static_cast<std::size_t>(0.05 * with_additional_distances.size()));
        threshold = *it;
      }

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
            (obj.indices.find(n) == obj.indices.end())) {
          TimeseriesDistanceFunctionData distance_data{
            data.distances.row(data.seed.index), data.distances.row(candidate)
          };
          calculated_distances.insert({ distance_function(distance_data), n });
          candidates.insert(n);
        }
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

      // Apply minimum segment threshold
      if (obj.indices.size() < data.min_segments)
        return ObjectByChange();

      return obj;
    }

    // If not, we now need to move all additional points into obj
    last_ratio = new_ratio;
    obj.threshold = threshold;
    obj.indices.merge(additional_points);
    std::swap(used_distances, with_additional_distances);

    // If the object is too large, we return it immediately
    // TODO: This does not actually cut the return object to max_segments,
    //       but the interplay with adaptive thresholding is non-trivial.
    if (obj.indices.size() >= data.max_segments)
      return obj;
  }

  // Apply minimum segment threshold
  if (obj.indices.size() < data.min_segments)
    return ObjectByChange();

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
