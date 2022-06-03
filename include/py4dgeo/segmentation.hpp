#pragma once

#include "py4dgeo/epoch.hpp"
#include "py4dgeo/py4dgeo.hpp"

#include <functional>
#include <unordered_map>
#include <vector>

namespace py4dgeo {

/** @brief The type to use for the spatiotemporal distance array. */
using EigenSpatiotemporalArray =
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using EigenSpatiotemporalArrayRef = Eigen::Ref<EigenSpatiotemporalArray>;
using EigenSpatiotemporalArrayConstRef =
  Eigen::Ref<const EigenSpatiotemporalArray>;

/** @brief The type to use for a TimeSeries */
using EigenTimeSeries = Eigen::Vector<double, Eigen::Dynamic>;
using EigenTimeSeriesRef = Eigen::Ref<EigenTimeSeries>;
using EigenTimeSeriesConstRef = Eigen::Ref<const EigenTimeSeries>;

/** @brief The data object passed to time series distance functions */
struct TimeseriesDistanceFunctionData
{
  EigenTimeSeriesConstRef ts1;
  EigenTimeSeriesConstRef ts2;
  double norm1;
  double norm2;
};

/** @brief The signature to use for a distance function */
using TimeseriesDistanceFunction =
  std::function<double(const TimeseriesDistanceFunctionData&)>;

/** @brief Basic data structure for 4D change object */
struct ObjectByChange
{
  std::unordered_map<IndexType, double> indices_distances;
  IndexType start_epoch;
  IndexType end_epoch;
  double threshold;
};

struct RegionGrowingSeed
{
  IndexType index;
  IndexType start_epoch;
  IndexType end_epoch;
};

struct RegionGrowingAlgorithmData
{
  EigenSpatiotemporalArrayConstRef distances;
  const Epoch& corepoints;
  double radius;
  RegionGrowingSeed seed;
  std::vector<double> thresholds;
  std::size_t min_segments;
  std::size_t max_segments;
};

/** @brief The main region growing algorithm */
ObjectByChange
region_growing(const RegionGrowingAlgorithmData&,
               const TimeseriesDistanceFunction&);

/** @brief The DTW distance measure implementation used in 4DOBC */
double
dtw_distance(const TimeseriesDistanceFunctionData&);

/** @brief Normalized DTW distance measure for 4DOBC */
double
normalized_dtw_distance(const TimeseriesDistanceFunctionData&);

} // namespace py4dgeo
