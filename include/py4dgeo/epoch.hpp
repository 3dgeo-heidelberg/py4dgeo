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
class Epoch
{
public:
  // Constructors
  Epoch(const EigenPointCloudRef&);
  Epoch(std::shared_ptr<EigenPointCloud>);

  // Methods for (de)serialization
  static std::unique_ptr<Epoch> from_stream(std::istream&);
  std::ostream& to_stream(std::ostream&) const;

private:
  // If this epoch is unserialized, it owns the point cloud
  std::shared_ptr<EigenPointCloud> owned_cloud;

public:
  // The data members are accessible from the outside. This could be
  // realized through getter methods.
  EigenPointCloudRef cloud;
  KDTree kdtree;

  // We can add a collection of metadata here
};

} // namespace py4dgeo
