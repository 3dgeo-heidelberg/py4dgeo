#pragma once

#include <py4dgeo/kdtree.hpp>
#include <py4dgeo/octree.hpp>
#include <py4dgeo/py4dgeo.hpp>
#include <py4dgeo/searchtree.hpp>

#include <iostream>
#include <memory>

namespace py4dgeo {

/** @brief A data structure representing an epoch
 *
 * Stores the point cloud itself (without taking ownership of it) and
 * provides two search trees: a KDTree and an Octree. This structure allows
 * efficient spatial queries without duplicating data.
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

  static void set_default_radius_search_tree(SearchTree tree)
  {
    default_radius_search_tree = tree;
  }

  static void set_default_nearest_neighbor_tree(SearchTree tree)
  {
    if (tree == SearchTree::Octree) {
      std::cerr << "[Warning] Octree is not yet implemented for nearest "
                   "neighbor queries. Use KDTree instead.\n";
      return;
    }
    default_nearest_neighbor_tree = tree;
  }

  static SearchTree get_default_radius_search_tree()
  {
    return default_radius_search_tree;
  }

  static SearchTree get_default_nearest_neighbor_tree()
  {
    return default_nearest_neighbor_tree;
  }

private:
  // If this epoch is unserialized, it owns the point cloud
  std::shared_ptr<EigenPointCloud> owned_cloud;

  // Default for search operations
  static SearchTree default_radius_search_tree;
  static SearchTree default_nearest_neighbor_tree;

public:
  // The data members are accessible from the outside. This could be
  // realized through getter methods.
  EigenPointCloudRef cloud;
  KDTree kdtree;
  Octree octree;

  // We can add a collection of metadata here
};

} // namespace py4dgeo
