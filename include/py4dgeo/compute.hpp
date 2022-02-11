#pragma once

#include <functional>

#include "epoch.hpp"
#include "kdtree.hpp"
#include "py4dgeo.hpp"

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

/** @brief The parameter struct for @ref UncertaintyMeasureCallback */
struct UncertaintyMeasureParameters
{
  /** @brief The point cloud in the first epoch to operate on */
  EigenPointCloudConstRef workingset1;
  /** @brief The point cloud in the second epoch to operate on */
  EigenPointCloudConstRef workingset2;
  /** @brief The surface normal at the current core point */
  EigenNormalSetConstRef normal;
  /** @brief The registration error */
  double registration_error;
};

/** @brief The callback type for calculating uncertainty measures */
using UncertaintyMeasureCallback =
  std::function<DistanceUncertainty(const UncertaintyMeasureParameters&)>;

/* Variety of callback declarations usable in M3C2 algorithms */

/** @brief Implementation of working set finder that performs a regular radius
 * search */
EigenPointCloud
radius_workingset_finder(const WorkingSetFinderParameters&);

/** @brief Implementation of a working set finder that performs a cylinder
 * search */
EigenPointCloud
cylinder_workingset_finder(const WorkingSetFinderParameters&);

/** @brief No-op implementation of uncertainty calculation
 *
 * This can be used if the calculation of uncertainties should be skipped
 * to save computation time.
 */
inline DistanceUncertainty
no_uncertainty(const UncertaintyMeasureParameters&)
{
  return DistanceUncertainty{ 0.0, 0.0, 0, 0.0, 0 };
}

/** @brief Standard deviation implementation of uncertainty calculation */
DistanceUncertainty
standard_deviation_uncertainty(const UncertaintyMeasureParameters&);

/* Compute interfaces used in the M3C2 main algorithm */

/** @brief Compute M3C2 multi scale directions */
void
compute_multiscale_directions(const Epoch&,
                              EigenPointCloudConstRef,
                              const std::vector<double>&,
                              EigenNormalSetConstRef,
                              EigenNormalSetRef);

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
                  const UncertaintyMeasureCallback&);

}
