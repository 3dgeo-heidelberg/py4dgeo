#pragma once

#include <functional>

#include "kdtree.hpp"
#include "py4dgeo.hpp"

namespace py4dgeo {

/** Signatures for the callbacks used in the M3C2 algorithm implementation */

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
  std::function<EigenPointCloud(EigenPointCloudRef,
                                const KDTree&,
                                double,
                                EigenPointCloudRef,
                                EigenPointCloudRef,
                                double,
                                IndexType)>;

/** @brief Implementation of working set finder that performs a regular radius
 * search
 *
 * @param cloud
 * @param kdtree
 * @param radius
 * @param corepoint
 * @param direction
 * @param max_cylinder_length
 * @param core_idx
 *
 * @return A point cloud data structure representing the working set
 */
EigenPointCloud
radius_workingset_finder(EigenPointCloudRef cloud,
                         const KDTree& kdtree,
                         double radius,
                         EigenPointCloudRef corepoint,
                         EigenPointCloudRef direction,
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
cylinder_workingset_finder(EigenPointCloudRef cloud,
                           const KDTree& kdtree,
                           double radius,
                           EigenPointCloudRef corepoint,
                           EigenPointCloudRef direction,
                           double max_cylinder_length,
                           IndexType core_idx);

/** @brief Compute M3C2 multi scale directions */
void
compute_multiscale_directions(EigenPointCloudRef,
                              EigenPointCloudRef,
                              const std::vector<double>&,
                              const KDTree&,
                              EigenPointCloudRef);

/** @brief Compute M3C2 distances */
void
compute_distances(EigenPointCloudRef,
                  double,
                  EigenPointCloudRef,
                  const KDTree&,
                  EigenPointCloudRef,
                  const KDTree&,
                  EigenPointCloudRef,
                  double,
                  EigenVectorRef,
                  WorkingSetFinderCallback wsfinder = radius_workingset_finder);

}
