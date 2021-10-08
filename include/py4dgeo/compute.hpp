#pragma once

#include <functional>

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
  std::function<EigenPointCloud(EigenPointCloudConstRef,
                                const KDTree&,
                                double,
                                EigenPointCloudConstRef,
                                EigenPointCloudConstRef,
                                double,
                                IndexType)>;

/* Variety of callback declarations usable in M3C2 algorithms */

/** @brief Implementation of working set finder that performs a regular radius
 * search
 *
 * @param cloud The full point cloud that we operate on
 * @param kdtree The search tree for the point cloud
 * @param radius Search radius
 * @param corepoint The (single) core point that we are dealing with
 * @param direction The search direction
 * @param max_cylinder_length The maximum cylinder length
 * @param core_idx The index of the core point in the core point set
 *
 * @return A point cloud data structure representing the working set
 */
EigenPointCloud
radius_workingset_finder(EigenPointCloudConstRef cloud,
                         const KDTree& kdtree,
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
cylinder_workingset_finder(EigenPointCloudConstRef cloud,
                           const KDTree& kdtree,
                           double radius,
                           EigenPointCloudConstRef corepoint,
                           EigenPointCloudConstRef direction,
                           double max_cylinder_length,
                           IndexType core_idx);

/* Compute interfaces used in the M3C2 main algorithm */

/** @brief Compute M3C2 multi scale directions */
void
compute_multiscale_directions(EigenPointCloudConstRef,
                              EigenPointCloudConstRef,
                              const std::vector<double>&,
                              const KDTree&,
                              EigenPointCloudRef);

/** @brief Compute M3C2 distances */
void
compute_distances(EigenPointCloudConstRef,
                  double,
                  EigenPointCloudConstRef,
                  const KDTree&,
                  EigenPointCloudConstRef,
                  const KDTree&,
                  EigenPointCloudConstRef,
                  double,
                  EigenVectorRef,
                  const WorkingSetFinderCallback&);

}
