#include <py4dgeo/segmentation.hpp>

#include <py4dgeo/openmp.hpp>
#include <py4dgeo/searchtree.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
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
    RadiusSearchResult neighbors;
    auto radius_search =
      get_radius_search_function(data.corepoints, data.radius);
    radius_search(data.corepoints.cloud.row(candidate), neighbors);
    for (auto n : neighbors) {
      // Check whether the corepoint is already present among candidates,
      // the final result or the points added at this threshold level.
      // If none of that match, this is a new candidate
      if ((rejected.find(n) == rejected.end()) &&
          (obj.indices_distances.find(n) == obj.indices_distances.end())) {
        // Calculate the distance for this neighbor
        TimeseriesDistanceFunctionData distance_data{
          data.distances(
            data.seed.index,
            Eigen::seq(data.seed.start_epoch, data.seed.end_epoch)),
          data.distances(
            n, Eigen::seq(data.seed.start_epoch, data.seed.end_epoch)),
          data.distances(data.seed.index, 0),
          data.distances(n, 0)
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

double
median_calculation(std::vector<double>& subsignal)
{                          // function calculate median of the vector
                           // the function change the vector
  if (subsignal.empty()) { // exeption
    throw std::runtime_error{ "Empty signal passed to median calculation" };
  }
  auto n = subsignal.size() / 2;
  std::nth_element(subsignal.begin(), subsignal.begin() + n, subsignal.end());
  double med = subsignal[n];
  if (subsignal.size() % 2 == 0) { // If the set size is even
    auto max_it = std::max_element(subsignal.begin(), subsignal.begin() + n);
    med = (*max_it + med) / 2.0;
  }
  return med;
}

double
median_calculation_simp(std::vector<double>& subsignal)
{
  if (subsignal.empty()) {
    throw std::runtime_error{ "Empty signal passed to median calculation" };
  }

  // Copy elements to a separate container
  std::vector<double> values = subsignal;

  auto n = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + n, values.end());

  return values[values.size() / 2];
}

std::vector<IndexType>
local_maxima_calculation(std::vector<double>& score, IndexType order)
{
  std::vector<IndexType> result;

  if (score.empty()) { // exeption
    throw std::runtime_error{ "The score is empty" };
  }
  if (order < 1) { // exeption
    throw std::runtime_error{
      "Order in local_maxima_calculation function < 1"
    };
  }
  auto n = score.size();
  if (n == 1) { // exeption
    throw std::runtime_error{ "The score contains only one element" };
  }

  std::vector<double>::const_iterator current = score.begin();

  std::vector<double>::const_iterator right_array_left_index =
    score.begin() + 1;
  std::vector<double>::const_iterator right_array_right_index =
    score.begin() + order + 1;

  std::vector<double>::const_iterator left_array_left_index = score.begin();
  std::vector<double>::const_iterator left_array_right_index = score.begin();

  std::vector<double>::const_iterator max_right_array =
    std::max_element(right_array_left_index, right_array_right_index);

  std::vector<double>::const_iterator max_left_array =
    std::max_element(left_array_left_index, left_array_right_index);

  auto distance_range = 0;
  auto max_right_array_num = 0.0;
  auto max_left_array_num = 0.0;
  auto second_max = 0.0;
  IndexType current_distance = std::distance(score.cbegin(), current);

  while (current_distance < n) // check main part of score
  {
    if (left_array_left_index == left_array_right_index) {
      max_left_array_num = 0;
    } else {
      max_left_array =
        std::max_element(left_array_left_index, left_array_right_index);
      max_left_array_num = *max_left_array;
    }

    if (right_array_left_index == right_array_right_index) {
      max_right_array_num = 0;
    } else {
      max_right_array =
        std::max_element(right_array_left_index, right_array_right_index);
      max_right_array_num = *max_right_array;
    }

    if (current_distance < order) {
      distance_range =
        order - std::distance(left_array_left_index, left_array_right_index);
      if (distance_range == order)
        max_left_array_num =
          *(std::max_element((score.cend() - distance_range), score.cend()));
      else
        max_left_array_num = std::max(
          *max_left_array,
          *(std::max_element((score.cend() - distance_range), score.cend())));
    }

    else if (current_distance > n - order - 1) {
      distance_range =
        order - std::distance(right_array_left_index, right_array_right_index);
      if (distance_range == order)
        max_right_array_num =
          *(std::max_element(score.cbegin(), score.cbegin() + distance_range));
      else
        max_right_array_num = std::max(
          *max_right_array,
          *(std::max_element(score.cbegin(), score.cbegin() + distance_range)));
    }

    if (*current > max_left_array_num && *current > max_right_array_num) {
      auto item_index = current - score.cbegin();
      result.push_back(item_index);
      if (current_distance + order + 1 < n) {
        current = current + order + 1;
        current_distance = current_distance + order + 1;
      } else
        break;
    }

    else {
      if (current_distance + 1 < n) {
        current++;
        current_distance++;
      } else
        break;
    }

    if (current == score.end() - 1) {
      right_array_left_index = score.cend();
    } else {
      right_array_left_index = current + 1;
    }

    if (current_distance >= n - order - 1) {
      right_array_right_index = score.cend();
    } else {
      right_array_right_index = current + order + 1;
    }

    left_array_right_index = current;
    if (current_distance < order)
      left_array_left_index = score.begin();
    else
      left_array_left_index = current - order;
  }
  return result;
}

double
cost_L1_error(EigenTimeSeriesConstRef signal,
              IndexType start,
              IndexType end,
              IndexType min_size)
{ // the function calculate error with cost function "l1"

  if (end < start) { // exeption
    throw std::runtime_error{ "End < Start in cost_L1_error function" };
  }

  if (start == end) {
    return 0.0;
  }

  std::vector<double> signal_subvector(signal.begin() + start,
                                       signal.begin() + end);

  double median = median_calculation(signal_subvector);

  double sum_result = std::accumulate(
    signal_subvector.begin(),
    signal_subvector.end(),
    0.0,
    [median](double a, double b) { return a + std::abs(b - median); });

  return sum_result;
}

std::vector<double>
fit_change_point_detection(EigenTimeSeriesConstRef signal,
                           IndexType width,
                           IndexType jump,
                           IndexType min_size)
{
  std::vector<double> score;
  score.reserve(signal.size());
  double gain;
  IndexType half_of_width = width / 2;

  for (int i = 0; i < signal.size(); i += jump) {
    if ((i < half_of_width) || (i >= (signal.size() - half_of_width))) {
      continue;
    }
    IndexType start = i - half_of_width;
    IndexType end = i + half_of_width;
    gain = cost_L1_error(signal, start, end, min_size);
    if (gain < 0) {
      score.push_back(0);
    }
    gain -= cost_L1_error(signal, start, i, min_size) +
            cost_L1_error(signal, i, end, min_size);
    score.push_back(gain);
  }
  return score;
}

double
sum_of_costs(EigenTimeSeriesConstRef signal,
             std::vector<IndexType>& bkps,
             IndexType min_size)
{ // input bkps array should be sorted!!!
  double result = 0.0;
  int start = 0;
  int end = 0;
  for (auto i : bkps) {
    end = i;
    result += cost_L1_error(signal, start, end, min_size);
    start = end;
  }
  return result;
}

std::vector<IndexType>
predict_change_point_detection(EigenTimeSeriesConstRef signal,
                               std::vector<double>& score,
                               IndexType width,
                               IndexType jump,
                               IndexType min_size,
                               double pen)
{
  int n_samples = signal.size();
  std::vector<IndexType> bkps;
  bkps.reserve(n_samples / width);
  int bkp;
  double gain;
  bkps.push_back(n_samples);
  bool stop = false;
  double error = sum_of_costs(signal, bkps, min_size);

  // forcing order to be above one in case jump is too large
  int preorder = std::max(width, 2 * min_size) / (2 * jump);
  int order = std::max(preorder, 1);

  std::vector<int> inds;
  inds.reserve(signal.size()); // todo: I can make it faster

  int half_of_width = width / 2;
  for (int i = 0; i < signal.size(); i += jump) {
    if ((i < half_of_width) || (i >= (signal.size() - half_of_width))) {
      continue;
    } else
      inds.push_back(i);
  }

  std::vector<IndexType> peak_inds_shifted_indx;

  peak_inds_shifted_indx = local_maxima_calculation(score, order);

  std::vector<double> gains;
  std::vector<IndexType> peak_inds_arr;
  std::vector<IndexType> peak_inds;

  for (auto i : peak_inds_shifted_indx) {
    gains.push_back(score[i]);
    peak_inds_arr.push_back(inds[i]);
  }

  std::vector<int> index_vec(peak_inds_arr.size());
  std::iota(index_vec.begin(), index_vec.end(), 0);
  std::sort(index_vec.begin(), index_vec.end(), [&](int a, int b) {
    return gains[a] < gains[b];
  });

  for (auto i : index_vec) {
    peak_inds.push_back(peak_inds_arr[i]);
  }

  while (!stop) {
    stop = true;
    if (peak_inds.size() != 0) {
      bkp = peak_inds.back();
      peak_inds.pop_back();
    } else {
      break;
    }

    if (pen > 0) {
      std::vector<IndexType> temp_bkps = bkps;
      temp_bkps.push_back(bkp);
      sort(temp_bkps.begin(), temp_bkps.end());
      gain = error - sum_of_costs(signal, temp_bkps, min_size);
      if (gain > pen) {
        stop = false;
      }
    }

    if (!stop) {
      bkps.push_back(bkp);
      sort(bkps.begin(), bkps.end());
      error = sum_of_costs(signal, bkps, min_size);
    }
  }
  return bkps;
}

std::vector<IndexType>
change_point_detection(const ChangePointDetectionData& data)
{
  std::vector<IndexType> changepoints;
  std::vector<double> score;
  score.reserve(data.ts.size());
  score = fit_change_point_detection(
    data.ts, data.window_width, data.jump, data.min_size);
  changepoints = predict_change_point_detection(
    data.ts, score, data.window_width, data.jump, data.min_size, data.penalty);

  return changepoints;
}

}
