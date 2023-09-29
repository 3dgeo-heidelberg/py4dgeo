#pragma once

#include "kdtree.hpp"
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

std::vector<std::vector<int>>
supervoxel_segmentation(EigenPointCloudConstRef cloud,
                        const KDTree& kdtree,
                        double seed_resolution);

} // namespace py4dgeo
