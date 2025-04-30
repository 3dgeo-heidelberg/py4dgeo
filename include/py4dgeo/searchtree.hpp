#pragma once

#include <py4dgeo/py4dgeo.hpp>

#include <Eigen/Core>

#include <functional>
#include <vector>

namespace py4dgeo {

//! Return type used for radius searches
using RadiusSearchResult = std::vector<IndexType>;

//! Return type used for radius searches that export calculated **squared**
//! distances
using RadiusSearchDistanceResult = std::vector<std::pair<IndexType, double>>;

//! Return type used for nearest neighbor with Euclidian distances searches
using NearestNeighborsDistanceResult =
  std::vector<std::pair<std::vector<IndexType>, std::vector<double>>>;

//! Return type used for nearest neighbor searches
using NearestNeighborsResult = std::vector<std::vector<IndexType>>;

enum class SearchTree
{
  KDTree,
  Octree,
};

class Epoch;

// Helper types
using RadiusSearchFuncSingle =
  std::function<void(const Eigen::Vector3d&, std::vector<IndexType>&)>;

using RadiusSearchFunc = std::function<
  void(const Eigen::Vector3d&, std::size_t, std::vector<IndexType>&)>;

// For a single radius
RadiusSearchFuncSingle
get_radius_search_function(const Epoch& epoch, double radius);

// For a vector of radii
RadiusSearchFunc
get_radius_search_function(const Epoch& epoch,
                           const std::vector<double>& radii);

} // namespace py4dgeo
