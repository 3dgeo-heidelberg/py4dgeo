#pragma once

#include <py4dgeo/compute.hpp>
#include <py4dgeo/epoch.hpp>
#include <py4dgeo/kdtree.hpp>
#include <py4dgeo/py4dgeo.hpp>
#include <py4dgeo/searchtree.hpp>
#include <py4dgeo/segmentation.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <vector>

namespace py4dgeo {

/** @brief The type that used for affine transformations on point clouds */
using Transformation = Eigen::Transform<double, 3, Eigen::Affine>;

/** @brief Apply an affine transformation to a point cloud (in-place) */
void
transform_pointcloud_inplace(EigenPointCloudRef cloud,
                             const Transformation& trafo,
                             EigenPointCloudConstRef reduction_point,
                             EigenNormalSetRef normals);

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

/** @brief Perform supervoxel segmentation
 *
 * @param epoch The epoch to be segmented.
 * @param kdtree The KDTree corresponding to the epoch's points.
 * @param seed_resolution The seed resolution used in supervoxel count
 * calculation.
 * @param k The number of neighbors considered for each point during
 * segmentation.
 * @param normals The normal vectors of the epoch's points.
 *
 */
std::vector<std::vector<int>>
supervoxel_segmentation(
  Epoch& epoch,
  const KDTree& kdtree,
  double seed_resolution,
  int k,
  EigenNormalSet normals = EigenNormalSet::Zero(
    1,
    3)); // it will be changed to EigenNormalSetRef afterwards);

/** @brief Perform supervoxel segmentation algorithm to distribute points within
 * an epoch.
 * @param epoch The epoch to be segmented.
 * @param kdtree The KDTree corresponding to the epoch's points.
 * @param normals The normal vectors of the epoch's points.
 * @param resolution The resolution of the supervoxels.
 * @param k The number of neighbors considered for each point during
 * segmentation.
 * @param minSVPvalue The minimum number of points in a supervoxel.
 *  */

std::vector<Supervoxel>
segment_pc(Epoch& epoch,
           const KDTree& kdtree,
           EigenNormalSet normals,
           double resolution,
           int k,
           int minSVPvalue);

/** @brief Perform searching a transformation that fits two point clouds onto
 * each other using Gauss-Newton method
 * @param epoch The epoch to be segmented.
 * @param kdtree The KDTree corresponding to the epoch's points.
 * @param normals The normal vectors of the epoch's points.
 * @param resolution The resolution of the supervoxels.
 * @param k The number of neighbors considered for each point during
 * segmentation.
 * @param minSVPvalue The mutable parameter.
 *  */

Eigen::Matrix4d
fit_transform_GN(EigenPointCloudConstRef trans_cloud,
                 EigenPointCloudConstRef reference_cloud,
                 EigenNormalSetConstRef reference_normals);

} // namespace py4dgeo
