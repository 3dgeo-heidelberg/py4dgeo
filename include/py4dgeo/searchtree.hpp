#pragma once

#include <py4dgeo/py4dgeo.hpp>

#include <Eigen/Core>

#include <functional>
#include <vector>

namespace py4dgeo {

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
