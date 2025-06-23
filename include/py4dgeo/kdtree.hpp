#pragma once

#include <py4dgeo/py4dgeo.hpp>
#include <py4dgeo/searchtree.hpp>

#include "nanoflann.hpp"
#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <istream>
#include <memory>
#include <ostream>
#include <vector>

namespace py4dgeo {

// Forward declaration of Epoch
class Epoch;

/** @brief Efficient KDTree data structure for nearest neighbor/radius searches
 *
 * This data structure allows efficient radius searches in 3D point cloud data.
 * It is based on NanoFLANN: https://github.com/jlblancoc/nanoflann
 */
class KDTree
{

private:
  /** @brief An adaptor between our Eigen data structures and NanoFLANN */
  struct Adaptor
  {
    EigenPointCloudRef cloud;

    inline std::size_t kdtree_get_point_count() const { return cloud.rows(); }

    double kdtree_get_pt(const IndexType idx, const IndexType dim) const
    {
      return cloud(idx, dim);
    }

    template<class BBOX>
    bool kdtree_get_bbox(BBOX&) const
    {
      return false;
    }
  };

  /** @brief A structure to perform efficient radius searches with NanoFLANN
   *
   * The built-in return set of NanoFLANN does automatically export the
   * distances as well, which we want to omit if we already know that we do not
   * need the distance information.
   */
  struct NoDistancesReturnSet
  {
    using DistanceType = double;

    double radius;
    RadiusSearchResult& indices;

    inline std::size_t size() const { return indices.size(); }

    inline bool full() const { return true; }

    inline bool addPoint(double dist, IndexType idx)
    {
      if (dist < radius)
        indices.push_back(idx);
      return true;
    }

    inline double worstDist() const { return radius; }

    inline void sort() {}
  };

  /**
   * @brief A custom result set structure for collecting radius search results
   *        with squared distances using nanoflann.
   *
   * This structure is used with nanoflann's radiusSearchCustomCallback()
   * to collect search results in a user-defined format. It stores pairs of
   * point indices and their corresponding squared distances to the query point
   * in a std::vector<std::pair<IndexType, double>>
   * (RadiusSearchDistanceResult).
   *
   * By using this structure instead of nanoflann::ResultItem, we avoid copying
   * between incompatible result types and maintain compatibility with Octree
   * which uses the same result type.
   *
   * This class satisfies the interface expected by nanoflann's custom result
   * set:
   * - addPoint(distance, index): stores the result if within radius
   * - size(), full(), worstDist(): for compatibility
   * - sort(): optional post-processing
   *
   */
  struct WithDistancesReturnSet
  {
    using DistanceType = double;

    double radius;
    RadiusSearchDistanceResult& result;

    inline std::size_t size() const { return result.size(); }
    inline bool full() const { return true; }

    inline bool addPoint(double dist_sq, IndexType idx)
    {
      if (dist_sq < radius)
        result.emplace_back(idx, dist_sq);
      return true;
    }

    inline double worstDist() const { return radius; }

    inline void sort()
    {
      std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
      });
    }
  };

  //! The NanoFLANN index implementation that we use
  using KDTreeImpl = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, Adaptor, double>,
    Adaptor,
    3,
    IndexType>;

  // We allow the Epoch class to directly call below constructor
  friend Epoch;

  //! Private constructor from pointcloud - use through @ref KDTree::create
  KDTree(const EigenPointCloudRef&);

public:
  /** @brief Construct instance of KDTree from a given point cloud
   *
   * This is implemented as a static function instead of a public constructor
   * to ease the implementation of Python bindings.
   *
   * @param cloud The point cloud to construct the search tree for
   */
  static KDTree create(const EigenPointCloudRef& cloud);

  /** @brief Save the index to a (file) stream */
  std::ostream& saveIndex(std::ostream& stream) const;

  /** @brief Load the index from a (file) stream */
  std::istream& loadIndex(std::istream& stream);

  /** @brief Build the KDTree index
   *
   * This initializes the KDTree search index. Calling this method is required
   * before performing any nearest neighbors or radius searches.
   *
   * @param leaf The threshold parameter definining at what size the search
   *             tree is cutoff. Below the cutoff, a brute force search is
   * performed. This parameter controls a trade off decision between search tree
   *             build time and query time.
   */
  void build_tree(int leaf);

  /** @brief Invalidate the KDTree index */
  void invalidate();

  /** @brief Peform radius search around given query point
   *
   * This method determines all the points from the point cloud within the given
   * radius of the query point. It returns only the indices and the result is
   * not sorted according to distance.
   *
   * @param[in] querypoint A pointer to the 3D coordinate of the query point
   * @param[in] radius The radius to search within
   * @param[out] result A data structure to hold the result. It will be cleared
   * during application.
   *
   * @return The amount of points in the return set
   */
  std::size_t radius_search(const double* querypoint,
                            double radius,
                            RadiusSearchResult& result) const;

  /** @brief Perform radius search around given query point exporting distance
   * information
   *
   * This method determines all the points from the point cloud within the given
   * radius of the query point. It returns their indices and their distances
   * from the query point. The result is sorted by ascending distance from the
   * query point.
   *
   * @param[in] querypoint A pointer to the 3D coordinate of the query point
   * @param[in] radius The radius to search within
   * @param[out] result A data structure to hold the result. It will be cleared
   * during application.
   *
   * @return The amount of points in the return set
   */
  std::size_t radius_search_with_distances(
    const double* querypoint,
    double radius,
    RadiusSearchDistanceResult& result) const;

  /** @brief Calculate the nearest neighbors with Euclidian distance for an
   * entire point cloud
   *
   * @param[in] cloud The point cloud to use as query points
   * @param[out] result The indexes and distances of k nearest neighbors for
   * each point
   * @param[in] k The amount of nearest neighbors to calculate
   *
   */
  void nearest_neighbors_with_distances(EigenPointCloudConstRef cloud,
                                        NearestNeighborsDistanceResult& result,
                                        int k) const;

  /** @brief Calculate the nearest neighbors for an entire point cloud
   *
   * @param[in] cloud The point cloud to use as query points
   * @param[out] result The indexes of k nearest neighbors for each point
   * @param[in] k The amount of nearest neighbors to calculate
   *
   */
  void nearest_neighbors(EigenPointCloudConstRef cloud,
                         NearestNeighborsResult& result,
                         int k) const;

  /** @brief Return the leaf parameter this KDTree has been built with */
  int get_leaf_parameter() const;

private:
  Adaptor adaptor;
  std::shared_ptr<KDTreeImpl> search;
  int leaf_parameter = 0;
};

} // namespace py4dgeo
