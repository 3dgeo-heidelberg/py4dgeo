#pragma once

#include <py4dgeo/py4dgeo.hpp>

#include <Eigen/Core>

#include <functional>
#include <vector>

namespace py4dgeo {

//! Return type used for radius searches
using RadiusSearchResult = std::vector<IndexType>;

//! Return type used for radius searches that export calculated distances
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

// Helper type
using RadiusSearchFunc =
  std::function<void(const Eigen::Vector3d&, size_t, std::vector<IndexType>&)>;

// Declaration
RadiusSearchFunc
get_radius_search_function(const Epoch& epoch,
                           const std::vector<double>& radii,
                           SearchTree tree);

} // namespace py4dgeo
