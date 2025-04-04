#include "py4dgeo/searchtree.hpp"
#include <py4dgeo/epoch.hpp>

namespace py4dgeo {

RadiusSearchFunc
get_radius_search_function(const Epoch& epoch,
                           const std::vector<double>& radii,
                           SearchTree tree)
{
  if (tree == SearchTree::Octree) {
    std::vector<unsigned int> levels(radii.size());
    for (size_t i = 0; i < radii.size(); ++i) {
      levels[i] =
        epoch.octree.find_appropriate_level_for_radius_search(radii[i]);
    }

    return [&, levels = std::move(levels)](const Eigen::Vector3d& point,
                                           size_t r,
                                           std::vector<IndexType>& out) {
      out.clear();
      epoch.octree.radius_search(point, radii[r], levels[r], out);
    };
  } else {
    return
      [&](const Eigen::Vector3d& point, size_t r, std::vector<IndexType>& out) {
        out.clear();
        epoch.kdtree.radius_search(point.data(), radii[r], out);
      };
  }
}

} // namespace py4dgeo
