#include "py4dgeo/segmentation.hpp"
#include "py4dgeo/openmp.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <unordered_set>
#include <vector>

namespace py4dgeo {

void
fixed_threshold_region_growing(
  const RegionGrowingAlgorithmData& data,
  const TimeseriesDistanceFunction& distance_function,
  double threshold,
  ObjectByChange& obj)
{
  // Instantiate a set of candidates to check
  std::unordered_set<IndexType> rejected;
  std::multimap<double, IndexType> candidates_distances;
  std::set<double, std::greater<double>> used_distances;

  // We add the initial seed to the candidates in order to kick off
  // the calculation in the loop below
  candidates_distances.insert({ 0.0, data.seed.index });

  // Create the return object
  obj.start_epoch = data.seed.start_epoch;
  obj.end_epoch = data.seed.end_epoch;
  obj.threshold = threshold;

  // The seed is included in the final object for sure
  obj.indices_distances[data.seed.index] = 0.0;
  used_distances.insert(0.0);

  double residual = threshold;

  // Grow while we have candidates
  while (!candidates_distances.empty()) {
    // Get one element to process and remove it from the candidates
    auto [distance, candidate] = *candidates_distances.begin();
    candidates_distances.erase(candidates_distances.begin());

    // Add neighboring corepoints to list of candidates
    KDTree::RadiusSearchResult neighbors;
    data.corepoints.kdtree.radius_search(
      &data.corepoints.cloud(candidate, 0), data.radius, neighbors);
    for (auto n : neighbors) {
      // Check whether the corepoint is already present among candidates,
      // the final result or the points added at this threshold level.
      // If none of that match, this is a new candidate
      if ((rejected.find(n) == rejected.end()) &&
          (obj.indices_distances.find(n) == obj.indices_distances.end())) {
        // Calculate the distance for this neighbor
        TimeseriesDistanceFunctionData distance_data{
          data.distances.row(data.seed.index),
          data.distances.row(candidate),
          data.distances(data.seed.index, 0),
          data.distances(candidate, 0)
        };
        auto d = distance_function(distance_data);

        // If it is smaller than the threshold, add it to the object (or
        // rather: maybe do so if adaptive thresholding wants you to)
        if (d < threshold) {
          obj.indices_distances[n] = d;
          used_distances.insert(d);
        } else {
          rejected.insert(n);
        }

        // Decide whether this neighbor should also be used as a candidate
        // for further neighbor selection. We do not do this with *all*
        // neighbors added to the grown region, but only with those under the
        // 95th percentile.
        if (d < residual) {
          candidates_distances.insert({ d, n });
        }

        // Update the residual parameter for above criterion
        if (obj.indices_distances.size() >= data.min_segments) {
          auto it = used_distances.begin();
          std::advance(it,
                       static_cast<std::size_t>(0.05 * used_distances.size()));
          residual = *it;
        }
      }
    }
  }
}

ObjectByChange
region_growing(const RegionGrowingAlgorithmData& data,
               const TimeseriesDistanceFunction& distance_function)
{
  // Ensure that we have a sorted list of thresholds
  std::vector<double> sorted_thresholds(data.thresholds);
  std::sort(sorted_thresholds.begin(), sorted_thresholds.end());

  // Run for all threshold levels
  std::vector<ObjectByChange> objects(sorted_thresholds.size());

  // Calculate region growing for all threshold levels. When finished,
  // decide which one to pick.
  CallbackExceptionVault vault;
#ifdef PY4DGEO_WITH_OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < sorted_thresholds.size(); ++i)
    vault.run([&]() {
      fixed_threshold_region_growing(
        data, distance_function, sorted_thresholds[i], objects[i]);
    });

  // Potentially rethrow an exception that occurred in above parallel region
  vault.rethrow();

  // Decide which threshold to use
  double last_ratio = 0.5;
  for (std::size_t i = 0; i < objects.size() - 1; ++i) {
    // Apply the maximum threshold
    if (objects[i].indices_distances.size() >= data.max_segments)
      return objects[i];

    double new_ratio =
      static_cast<double>(objects[i].indices_distances.size()) /
      static_cast<double>(objects[i + 1].indices_distances.size());
    if (new_ratio <= last_ratio) {
      // Apply minimum segment threshold
      if (objects[i].indices_distances.size() < data.min_segments)
        return ObjectByChange();

      return objects[i];
    } else {
      last_ratio = new_ratio;
    }
  }

  return objects[objects.size() - 1];
}

inline double
distance(double x, double y, double norm1, double norm2)
{
  return std::fabs(x - norm1 - (y - norm2));
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
  d[0][0] = distance(
    data.ts1[indices[0]], data.ts2[indices[0]], data.norm1, data.norm2);

  // Upper row and left-most column
  for (std::size_t i = 1; i < n; ++i) {
    d[i][0] =
      distance(
        data.ts1[indices[i]], data.ts2[indices[0]], data.norm1, data.norm2) +
      d[i - 1][0];
    d[0][i] =
      distance(
        data.ts1[indices[0]], data.ts2[indices[i]], data.norm1, data.norm2) +
      d[0][i - 1];
  }

  // Rest of the distance matrix
  for (std::size_t i = 1; i < n; ++i)
    for (std::size_t j = 1; j < n; ++j)
      d[i][j] =
        distance(
          data.ts1[indices[i]], data.ts2[indices[j]], data.norm1, data.norm2) +
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

std::vector<double>
change_point_detection(const ChangePointDetectionData& data)
{
  std::vector<double> changepoints;

  // TODO: Implement detection here

  return changepoints;
}

}
