#pragma once

#include <py4dgeo/epoch.hpp>
#include <py4dgeo/kdtree.hpp>
#include <py4dgeo/py4dgeo.hpp>

#include <functional>
#include <tuple>
#include <vector>

namespace py4dgeo {

/* Signatures for the callbacks used in the M3C2 algorithm implementation */

/** @brief The parameter struct for @ref WorkingSetFinderCallback */
struct WorkingSetFinderParameters
{
  /** @brief The epoch that we are operating on */
  const Epoch& epoch;
  /** @brief The search radius*/
  double radius;
  /** @brief The (single) core point that we are dealing with */
  EigenPointCloudConstRef corepoint;
  /** @brief The cylinder axis direction */
  EigenNormalSetConstRef cylinder_axis;
  /** @brief The maximum cylinder (half) length*/
  double max_distance;
};

/** @brief The callback type that determines the point cloud working subset in
 * the vicinity of a core point */
using WorkingSetFinderCallback =
  std::function<EigenPointCloud(const WorkingSetFinderParameters&)>;

/** @brief The parameter struct for @ref DistanceUncertaintyCalculationCallback
 */
struct DistanceUncertaintyCalculationParameters
{
  /** @brief The point cloud in the first epoch to operate on */
  EigenPointCloudConstRef workingset1;
  /** @brief The point cloud in the second epoch to operate on */
  EigenPointCloudConstRef workingset2;
  /** @brief The (single) core point that we are dealing with */
  EigenPointCloudConstRef corepoint;
  /** @brief The surface normal at the current core point */
  EigenNormalSetConstRef normal;
  /** @brief The registration error */
  double registration_error;
};

/** @brief The callback type for calculating the distance between two point
 * clouds */
using DistanceUncertaintyCalculationCallback =
  std::function<std::tuple<double, DistanceUncertainty>(
    const DistanceUncertaintyCalculationParameters&)>;

/* Variety of callback declarations usable in M3C2 algorithms */

/** @brief Implementation of working set finder that performs a regular radius
 * search */
EigenPointCloud
radius_workingset_finder(const WorkingSetFinderParameters&);

/** @brief Implementation of a working set finder that performs a cylinder
 * search */
EigenPointCloud
cylinder_workingset_finder(const WorkingSetFinderParameters&);

/** @brief Mean-based implementation of point cloud distance
 *
 * This is the default implementation of point cloud distance that takes
 * the mean of both point clouds (center of mass), projects it onto the
 * cylinder axis and calculates the distance.
 */
std::tuple<double, DistanceUncertainty>
mean_stddev_distance(const DistanceUncertaintyCalculationParameters&);

/** @brief Median-based implementation of point cloud distance
 *
 * Use median of distances in pointcloud instead of mean. This
 * results in a more expensive but more robust distance measure.
 */
std::tuple<double, DistanceUncertainty>
median_iqr_distance(const DistanceUncertaintyCalculationParameters&);

/* Compute interfaces used in the M3C2 main algorithm */

/** @brief Compute M3C2 multi scale directions */
void
compute_multiscale_directions(const Epoch&,
                              EigenPointCloudConstRef,
                              const std::vector<double>&,
                              EigenNormalSetConstRef,
                              EigenNormalSetRef,
                              std::vector<double>&);

/** @brief Compute M3C2 distances */
void
compute_distances(EigenPointCloudConstRef,
                  double,
                  const Epoch&,
                  const Epoch&,
                  EigenNormalSetConstRef,
                  double,
                  double,
                  DistanceVector&,
                  UncertaintyVector&,
                  const WorkingSetFinderCallback&,
                  const DistanceUncertaintyCalculationCallback&);

/** @brief Compute correspondence distances */
std::vector<double>
compute_correspondence_distances(const Epoch&,
                                 EigenPointCloudConstRef,
                                 std::vector<EigenPointCloud>,
                                 unsigned int);

}
