#pragma once

#include <functional>

#include "epoch.hpp"
#include "kdtree.hpp"
#include "py4dgeo.hpp"

namespace py4dgeo {

/* Signatures for the callbacks used in the M3C2 algorithm implementation */

/** @brief The callback type that determines the point cloud working subset in
 * the vicinity of a core point
 *
 *  For a reference of the signature, see the implementations provided by @c
 * py4dgeo:
 *
 *  * @ref radius_workingset_finder
 *  * @ref cylinder_workingset_finder
 */
using WorkingSetFinderCallback =
  std::function<EigenPointCloud(const Epoch&,
                                double,
                                EigenPointCloudConstRef,
                                EigenPointCloudConstRef,
                                double,
                                IndexType)>;

/** @brief The callback type for calculating uncertainty measures
 *
 * For a reference of the signature, see the implementation provided by @c
 * py4dgeo:
 *
 * * @ref no_uncertainty
 * * @ref standard_deviation_uncertainty
 */
using UncertaintyMeasureCallback =
  std::function<double(EigenPointCloudConstRef,
                       EigenPointCloudConstRef,
                       EigenPointCloudConstRef)>;

/* Variety of callback declarations usable in M3C2 algorithms */

/** @brief Implementation of working set finder that performs a regular radius
 * search
 *
 * @param epoch The epoch that we are operating on
 * @param radius Search radius
 * @param corepoint The (single) core point that we are dealing with
 * @param direction The search direction
 * @param max_cylinder_length The maximum cylinder length
 * @param core_idx The index of the core point in the core point set
 *
 * @return A point cloud data structure representing the working set
 */
EigenPointCloud
radius_workingset_finder(const Epoch& epoch,
                         double radius,
                         EigenPointCloudConstRef corepoint,
                         EigenPointCloudConstRef direction,
                         double max_cylinder_length,
                         IndexType core_idx);

/** @brief Implementation of a working set finder that performs a cylinder
 * search.
 *
 * Selects all points within the cylinder defined by:
 * * the middle axis through @c corepoint in direction @c direction
 * * the radius given by @c radius
 * * a maximum cylinder length parameter needed to save resources
 *
 * @copydoc radius_workingset_finder
 */
EigenPointCloud
cylinder_workingset_finder(const Epoch& epoch,
                           double radius,
                           EigenPointCloudConstRef corepoint,
                           EigenPointCloudConstRef direction,
                           double max_cylinder_length,
                           IndexType core_idx);

/** @brief No-op implementation of uncertainty calculation
 *
 * This can be used if the calculation of uncertainties should be skipped
 * to save computation time.
 */
inline double
no_uncertainty(EigenPointCloudConstRef,
               EigenPointCloudConstRef,
               EigenPointCloudConstRef)
{
  return 0.0;
}

/** @brief Standard deviation implementation of uncertainty calculation
 *
 * Calculates the standard deviation of the distance measure.
 *
 * @param set1 The working set of points in the first epoch
 * @param set2 The working set of points in the second epoch
 * @param direction The normal direction
 * @returns uncertainty The storage for the computed uncertainty values
 */
double
standard_deviation_uncertainty(EigenPointCloudConstRef set1,
                               EigenPointCloudConstRef set2,
                               EigenPointCloudConstRef direction);

/* Compute interfaces used in the M3C2 main algorithm */

/** @brief Compute M3C2 multi scale directions */
void
compute_multiscale_directions(const Epoch&,
                              EigenPointCloudConstRef,
                              const std::vector<double>&,
                              EigenPointCloudRef);

/** @brief Compute M3C2 distances */
void
compute_distances(EigenPointCloudConstRef,
                  double,
                  const Epoch&,
                  const Epoch&,
                  EigenPointCloudConstRef,
                  double,
                  EigenVectorRef,
                  EigenVectorRef,
                  const WorkingSetFinderCallback&,
                  const UncertaintyMeasureCallback&);

}
