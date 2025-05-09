#include "py4dgeo/searchtree.hpp"

#include <py4dgeo/epoch.hpp>

#include <Eigen/Core>

#include <vector>

namespace py4dgeo {

// For a single radius
RadiusSearchFuncSingle
get_radius_search_function(const Epoch& epoch, double radius)
{
  if (Epoch::get_default_radius_search_tree() == SearchTree::Octree) {
    unsigned int level =
      epoch.octree.find_appropriate_level_for_radius_search(radius);

    return [&, radius, level](const Eigen::Vector3d& point,
                              RadiusSearchResult& out) {
      epoch.octree.radius_search(point, radius, level, out);
    };
  } else {
    return [&, radius](const Eigen::Vector3d& point, RadiusSearchResult& out) {
      epoch.kdtree.radius_search(point.data(), radius, out);
    };
  }
}

// For a vector of radii
RadiusSearchFunc
get_radius_search_function(const Epoch& epoch, const std::vector<double>& radii)
{
  if (Epoch::get_default_radius_search_tree() == SearchTree::Octree) {
    std::vector<unsigned int> levels(radii.size());
    for (size_t i = 0; i < radii.size(); ++i) {
      levels[i] =
        epoch.octree.find_appropriate_level_for_radius_search(radii[i]);
    }

    return [&, radii, levels = std::move(levels)](
             const Eigen::Vector3d& point, size_t r, RadiusSearchResult& out) {
      epoch.octree.radius_search(point, radii[r], levels[r], out);
    };
  } else {
    return [&, radii](
             const Eigen::Vector3d& point, size_t r, RadiusSearchResult& out) {
      epoch.kdtree.radius_search(point.data(), radii[r], out);
    };
  }
}

} // namespace py4dgeo
