#pragma once

#include "py4dgeo/kdtree.hpp"
#include "py4dgeo/py4dgeo.hpp"

namespace py4dgeo {

/** @brief A data structure representing an epoch
 *
 * It stores the point cloud itself (without taking ownership of it) and
 * the KDTree (with ownership). In the future, relevant metadata fields can
 * be easily added to this data structure without changing any signatures
 * that depend on Epoch.
 */
struct Epoch
{
  Epoch(EigenPointCloudRef cloud_)
    : cloud(cloud_)
    , kdtree(cloud_)
  {}

  // The relevant data members
  EigenPointCloudRef cloud;
  KDTree kdtree;

  // We can add a collection of metadata here
};

} // namespace py4dgeo
