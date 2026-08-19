#pragma once

#include <py4dgeo/py4dgeo.hpp>

#include <Eigen/Core>

#include <cstddef>
#include <functional>
#include <utility>
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

/**
 * @brief Function type for performing a single-radius search.
 *
 * This function takes a 3D query point (as an Eigen vector) and outputs a
 * vector of point indices that lie within a sphere around the query point with
 * specified radius. The specific search algorithm (KDTree or Octree) is
 * determined at runtime.
 *
 * @param query_point The 3D coordinates of the point to search around.
 * @param result A vector of indices of points within the search radius.
 */
using RadiusSearchFuncSingle =
  std::function<void(const Eigen::Vector3d&, RadiusSearchResult&)>;

/**
 * @brief Function type for performing a multi-radius search.
 *
 * This function takes a 3D point and an index representing the radius to use
 * from a precomputed list of radii. It outputs a vector of point indices that
 * lie within the selected radius.
 *
 * @param query_point The 3D coordinates of the point to search around.
 * @param radius_index The index to select the radius from a list of radii.
 * @param result A vector of indices of points within the search radius.
 */
using RadiusSearchFunc =
  std::function<void(const Eigen::Vector3d&, std::size_t, RadiusSearchResult&)>;

/**
 * @brief Returns a function for performing a single-radius search.
 *
 * Depending on the default search tree type set in the Epoch class (KDTree or
 * Octree), this function returns a callable object that efficiently performs
 * radius searches around a given point.
 *
 * In the case of the Octree, the function also computes the appropriate level
 * of the tree based on the given radius.
 *
 * @param epoch The Epoch object containing the search tree.
 * @param radius The radius within which to search for neighboring points.
 *
 * @return A callable function that performs the radius search.
 */
RadiusSearchFuncSingle
get_radius_search_function(const Epoch& epoch, double radius);

/**
 * @brief Returns a function for performing multi-radius searches.
 *
 * Depending on the default search tree type set in the Epoch class (KDTree or
 * Octree), this function returns a callable object that efficiently performs
 * radius searches for a given list of radii.
 *
 * In the case of the Octree, the function precomputes the appropriate level
 * for each radius to optimize search performance.
 *
 * @param epoch The Epoch object containing the search tree.
 * @param radii A vector of radii for which search functions are needed.
 *
 * @return A callable function that performs the radius search for each radius.
 */
RadiusSearchFunc
get_radius_search_function(const Epoch& epoch,
                           const std::vector<double>& radii);

} // namespace py4dgeo
