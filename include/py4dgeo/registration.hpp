#pragma once

#include <py4dgeo/py4dgeo.hpp>

#include <Eigen/Geometry>

namespace py4dgeo {

/** @brief The type that used for affine transformations on point clouds */
using Transformation = Eigen::Transform<double, 3, Eigen::Affine>;

/** @brief Apply an affine transformation to a point cloud (in-place) */
void
transform_pointcloud_inplace(EigenPointCloudRef cloud,
                             const Transformation& trafo);

} // namespace py4dgeo
