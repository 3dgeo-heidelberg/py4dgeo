#pragma once

#include <Eigen/Core>

namespace py4dgeo {

/* Declare the most important types used in py4dgeo */

/** @brief The C++ type for a point cloud
 *
 * Point clouds are represented as (nx3) matrices from the Eigen library.
 *
 * The choice of this type allows us both very efficient implementation
 * of numeric algorithms using the Eigen library, as well as no-copy
 * interoperability with numpy's multidimensional arrays.
 */
using EigenPointCloud =
  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

/** @brief A non-const reference type for passing around @ref EigenPointCloud
 *
 * You should use this in function signatures that accept a point cloud
 * as a parameter and need to modify the point cloud.
 */
using EigenPointCloudRef = Eigen::Ref<EigenPointCloud>;

/** @brief A const reference type for passing around @ref EigenPointCloud
 *
 * You should use this in function signatures that accept a read-only
 * point cloud.
 */
using EigenPointCloudConstRef = Eigen::Ref<const EigenPointCloud>;

/** @brief The C++ type to represent a set of normal vectors on a point cloud */
using EigenNormalSet =
  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

/** @brief A mutable reference to a set of normal vectors on a point cloud */
using EigenNormalSetRef = Eigen::Ref<EigenNormalSet>;

/** @brief An immutable reference to a set of normal vectors on a point cloud */
using EigenNormalSetConstRef = Eigen::Ref<const EigenNormalSet>;

/** @brief The type used for point cloud indices */
using IndexType = Eigen::Index;

/** @brief Return structure for the uncertainty of the distance computation
 *
 * This structured type is used to describe the uncertainty of point cloud
 * distance at a single corepoint. It contains the level of detection,
 * the spread within both point clouds (e.g. the standard deviation of the
 * distance measure) and the number of sampled points.
 */
struct DistanceUncertainty
{
  double lodetection;
  double spread1;
  IndexType num_samples1;
  double spread2;
  IndexType num_samples2;
};

/** @brief The variable-sized vector type used for distances */
using DistanceVector = std::vector<double>;

/** @brief The variable-sized vector type used for uncertainties */
using UncertaintyVector = std::vector<DistanceUncertainty>;

/** @brief An enumerator for py4dgeo's memory policy
 *
 * This is used and documented through its Python binding equivalent.
 */
enum class MemoryPolicy
{
  STRICT = 0,
  MINIMAL = 1,
  COREPOINTS = 2,
  RELAXED = 3
};

struct Supervoxel
{
  EigenPointCloud cloud;
  EigenNormalSet normals;
  Eigen::Vector3d centroid;
  EigenPointCloud boundary_points;

  // Default constructor
  Supervoxel()
    : cloud()
    , normals()
    , centroid()
    , boundary_points()
  {
  }

  // constructor
  Supervoxel(EigenPointCloudConstRef c,
             EigenNormalSetConstRef n,
             const Eigen::Vector3d& center,
             EigenPointCloudConstRef boundary)
    : cloud(c)
    , normals(n)
    , centroid(center)
    , boundary_points(boundary)
  {
  }
};

}
