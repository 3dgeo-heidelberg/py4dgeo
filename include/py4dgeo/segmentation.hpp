#pragma once

#include "py4dgeo/py4dgeo.hpp"

#include <functional>
#include <vector>

namespace py4dgeo {

/** @brief The type to use for the spatiotemporal distance array. */
using EigenSpatiotemporalArray =
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using EigenSpatiotemporalArrayRef = Eigen::Ref<EigenSpatiotemporalArray>;
using EigenSpatiotemporalArrayConstRef =
  const Eigen::Ref<const EigenSpatiotemporalArray>&;

/** @brief The type to use for a TimeSeries */
using EigenTimeSeries = Eigen::Vector<double, Eigen::Dynamic>;
using EigenTimeSeriesRef = Eigen::Ref<EigenTimeSeries>;
using EigenTimeSeriesConstRef = const Eigen::Ref<const EigenTimeSeries>&;

/** @brief The signature to use for a distance function */
using TimeseriesDistanceFunction =
  std::function<double(EigenTimeSeriesConstRef, EigenTimeSeriesConstRef)>;

/** @brief Basic data structure for 4D change object */
struct ObjectByChange
{
  std::vector<IndexType> corepoint_indices;
  unsigned int start_epoch;
  unsigned int end_epoch;
};

/** The DTW distance measure implementation used in 4DOBC */
double dtw_distance(EigenTimeSeriesConstRef, EigenTimeSeriesConstRef);

} // namespace py4dgeo
