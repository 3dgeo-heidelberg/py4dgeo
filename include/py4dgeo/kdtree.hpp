#pragma once

#include <Eigen/Eigen>

#include <istream>
#include <memory>
#include <ostream>
#include <utility>
#include <vector>

#include "nanoflann.hpp"
#include "py4dgeo.hpp"

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
public:
  //! Return type used for radius searches
  using RadiusSearchResult = std::vector<IndexType>;

  //! Return type used for radius searches that export calculated distances
  using RadiusSearchDistanceResult = std::vector<std::pair<IndexType, double>>;

  //! Return type used for nearest neighbor searches
  using NearestNeighborsDistanceResult =
    std::pair<std::vector<IndexType>, std::vector<double>>;

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

  /** @brief Calculate the nearest neighbors for an entire point cloud
   *
   * @param[in] cloud The point cloud to use as query points
   * @param[out] result The distan
   */
  void nearest_neighbors_with_distances(
    EigenPointCloudConstRef cloud,
    NearestNeighborsDistanceResult& result) const;

  /** @brief Return the leaf parameter this KDTree has been built with */
  int get_leaf_parameter() const;

private:
  Adaptor adaptor;
  std::shared_ptr<KDTreeImpl> search;
  int leaf_parameter = 0;
};

} // namespace py4dgeo
