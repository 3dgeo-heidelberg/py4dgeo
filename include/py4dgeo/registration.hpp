#pragma once

#include "compute.hpp"
#include "epoch.hpp"
#include "kdtree.hpp"
#include "segmentation.hpp"
#include <py4dgeo/py4dgeo.hpp>

#include <Eigen/Geometry>

namespace py4dgeo {

/** @brief The type that used for affine transformations on point clouds */
using Transformation = Eigen::Transform<double, 3, Eigen::Affine>;

/** @brief Apply an affine transformation to a point cloud (in-place) */
void
transform_pointcloud_inplace(EigenPointCloudRef cloud,
                             const Transformation& trafo,
                             EigenPointCloudConstRef reduction_point);

/** Union/Find data structure */
class DisjointSet
{
public:
  /** @brief Construct the data structure for a given size */
  DisjointSet(IndexType size);

  /** @brief Find the subset identifier that the i-th element currently belongs
   * to */
  IndexType Find(IndexType i) const;

  /** @brief Merge two subsets into one
   *
   * @param i First subset identifier to merge
   * @param j Second subset identifier to merge
   * @param balance_sizes If true, the large subset is merged into the smaller.
   *
   * @return The subset identifier of the merged subset
   */
  IndexType Union(IndexType i, IndexType j, bool balance_sizes);

private:
  /** @brief The number of points in the data structure */
  IndexType size_;

  /** @brief The subset sizes */
  std::vector<IndexType> numbers_;

  /** @brief The subset index for all our points */
  mutable std::vector<IndexType> subsets_;
};
/** @brief Calculate the amount of supervoxel, based on seed_resolution */
std::size_t
estimate_supervoxel_count(EigenPointCloudConstRef cloud,
                          double seed_resolution);

/** @brief Distribution of points into a certain number of supervoxels
 *
 * @param epoch The epoch to segment
 * @param kdtree The KDTree for the epoch
 * @param seed_resolution The value for calculation of supervoxel count
 * @param k The number of neighbors for each point
 *
 */
std::vector<std::vector<int>>
supervoxel_segmentation(Epoch& epoch,
                        const KDTree& kdtree,
                        double seed_resolution,
                        int k);

} // namespace py4dgeo
