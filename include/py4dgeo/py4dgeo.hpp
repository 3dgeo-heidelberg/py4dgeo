#pragma once

#include <Eigen/Eigen>

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
using EigenPointCloudConstRef = const Eigen::Ref<const EigenPointCloud>&;

/** @brief A vector of dynamic size
 *
 * The choice of this type allows us both very efficient implementation
 * of numeric algorithms using the Eigen library, as well as no-copy
 * interoperability with numpy's arrays.
 */
using EigenVector = Eigen::Matrix<double, Eigen::Dynamic, 1>;

/** @brief A non-const reference type for passing around @ref EigenVector
 *
 * You should use this in function signatures that accept and modify a vector.
 */
using EigenVectorRef = Eigen::Ref<EigenVector>;

/** @brief A const reference type for passing around @ref EigenVector
 *
 * You should use this in function signatures that accept a read-only vector.
 */
using EigenVectorConstRef = const Eigen::Ref<const EigenVector>&;

/** @brief The type used for point cloud indices */
using IndexType = Eigen::Index;

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

}