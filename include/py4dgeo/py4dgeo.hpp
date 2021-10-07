#pragma once

#include <Eigen/Eigen>

namespace py4dgeo {

// The types we use for Point Clouds on the C++ side
using EigenPointCloud =
  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using EigenPointCloudRef = Eigen::Ref<EigenPointCloud>;
using EigenPointCloudConstRef = const Eigen::Ref<const EigenPointCloud>&;
using EigenVector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using EigenVectorRef = Eigen::Ref<EigenVector>;
using IndexType = Eigen::Index;

}