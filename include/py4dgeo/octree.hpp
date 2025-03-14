#pragma once

#include <Eigen/Eigen>

#include <istream>
#include <memory>
#include <ostream>
#include <utility>
#include <vector>

#include "py4dgeo.hpp"

namespace py4dgeo {

// Forward declaration of Epoch
class Epoch;

/** @brief Efficient Octree data structure for nearest neighbor/radius searches
 *
 * This data structure allows efficient radius searches in 3D point cloud data.
 * Unlike KDTree, it recursively subdivides space into eight octants.
 */
class Octree
{
public:
  //! Return type used for radius searches
  using RadiusSearchResult = std::vector<IndexType>;

  //! Return type used for radius searches that export calculated distances
  using RadiusSearchDistanceResult = std::vector<std::pair<IndexType, double>>;

  //! Return type used for nearest neighbor with Euclidian distances searches
  using NearestNeighborsDistanceResult =
    std::vector<std::pair<std::vector<IndexType>, std::vector<double>>>;

  //! Return type used for nearest neighbor searches
  using NearestNeighborsResult = std::vector<std::vector<IndexType>>;

  //! Alias for the spatial key type used for Z-order value encoding
  using SpatialKey = uint64_t; // 16-bit allows 5 depth levels, 32-bit allows 10
                               // levels, 64-bit allows 21 levels

  //! Struct combining Z-order value and original point index
  struct IndexAndKey
  {
    SpatialKey key;  //!< Z-order value
    IndexType index; //!< Index of the corresponding point in cloud

    //! Sorting operator (sort by Z-vale)
    bool operator<(const IndexAndKey& other) const { return key < other.key; }
  };

private:
  //! Reference to the point cloud
  EigenPointCloudRef cloud;
  //! Number of points in the cloud
  unsigned int number_of_points;

  //! Pairs of spatial key (Z-order values) and corresponding index, sorted by
  //! z-order value
  std::vector<IndexAndKey> indexed_keys;

  //! Min point of the bounding cube
  Eigen::Vector3d min_point;
  //! Max point of the bounding cube
  Eigen::Vector3d max_point;
  //! Size of the bounding cube
  double cube_size;

  //! Leaf parameter, i.e., maximum number of points a cell can hold before
  //! subdivision
  unsigned int leaf_parameter = 0;
  //! Max depth level of the octree, depends solely on spatial key integer
  //! representation
  static constexpr unsigned int max_depth = (sizeof(SpatialKey) * 8) / 3;

  //! Cell size as a function of depth level
  std::array<double, max_depth + 1> cell_size;

  //! Allow the Epoch class to directly call the private constructor
  friend Epoch;

  //! Private constructor from point cloud - use through @ref Octree::create
  Octree(const EigenPointCloudRef&);

private:
  /** @brief Return the number of points in the associated cloud */
  inline const unsigned int get_number_of_points() const
  {
    return number_of_points;
  }

  /** @brief Return the side length of the bounding cube */
  double const get_cube_size() const { return cube_size; }

  /** @brief Return the minimum point of the bounding cube */
  Eigen::Vector3d const get_min_point() const { return min_point; }

  /** @brief Return the maximum point of the bounding cube */
  Eigen::Vector3d const get_max_point() const { return max_point; }

  /** @brief Return the size of cells at a level of depth */
  inline const double get_cell_size(unsigned int level) const
  {
    return cell_size[level];
  }

  /**
   * @brief Computes the smallest power-of-two bounding cube that contains all
   * points.
   *
   * This function calculates the minimum and maximum coordinates of the point
   * cloud and determines a cube that fully encloses the data. The cube size is
   * rounded up to the nearest power of two to ensure a well-balanced Octree
   * structure.
   *
   * The computed bounding cube is stored in `min_point`, `max_point`, and
   * `cube_size`.
   */
  void compute_bounding_cube();

  /**
   * @brief Computes a unique spatial key (Z-order value) for a given point.
   *
   * This function normalizes the input point within the bounding cube and maps
   * it to an integer 3D grid of size '2^max_depth × 2^max_depth × 2^max_depth'.
   * The computed grid coordinates are then interleaved using bitwise operations
   * to generate a Z-order value key that represents the point's hierarchical
   * position in the Octree.
   *
   * @param point The 3D point for which to compute the spatial key.
   * @return A SpatialKey (unsigned integer) representing the spatial key of the
   * point.
   */
  SpatialKey compute_spatial_key(const Eigen::Vector3d& point) const;

  /**
   * @brief Coarsens the octree depending on the leaf parameter.
   *
   * This function applies spatial coarsening to the octree with max_depth
   * levels. If a potential coarses parent cell holds more than or exactly leaf
   * points, the current level of subdivision persists. Otherwise the
   * corresponding points are grouped according tho the spatial key of the
   * parent cell.
   *
   * @param leaf The leaf parameter defines the maximum number of points a cell
   * can hold without being subdivided into finer cells.
   */
  void apply_leaf_parameter(unsigned int leaf);

  /**
   * @brief Finds all neighboring Octree cells at a given depth level.
   *
   * This function identifies all neighboring cells that may intersect
   * a sphere search region by examining spatial keys at a given depth.
   *
   * @param truncated_key The truncated spatial key of the query point at the
   * given level.
   * @param level The Octree depth level at which to find neighboring cells.
   * @return A vector of spatial keys representing neighboring cells.
   */
  std::vector<SpatialKey> find_neighbors(SpatialKey truncated_key,
                                         int level) const;

public:
  /** @brief Construct instance of Octree from a given point cloud
   *
   * This is implemented as a static function instead of a public constructor
   * to ease the implementation of Python bindings.
   *
   * @param cloud The point cloud to construct the search tree for
   */
  static Octree create(const EigenPointCloudRef& cloud);

  /** @brief Save the Octree index to a (file) stream */
  std::ostream& saveIndex(std::ostream& stream) const;

  /** @brief Load the Octree index from a (file) stream */
  std::istream& loadIndex(std::istream& stream);

  /** @brief Build the Octree index
   *
   * This initializes the Octree search index. Calling this method is required
   * before performing any nearest neighbors or radius searches.
   *
   * @param leaf The maximum number of points a node can contain before
   * splitting.
   */
  void build_tree(unsigned int leaf);

  /**
   * @brief Clears the Octree structure, effectively resetting it.
   *
   * This function deallocates all nodes in the Octree by setting the root node
   * to nullptr. This operation invalidates the current Octree, requiring it
   * to be rebuilt before further use.
   */
  void invalidate();

  /** @brief Perform radius search around given query point
   *
   * This method determines all the points from the point cloud within the given
   * radius of the query point. It returns only the indices and the result is
   * not sorted according to distance.
   */
  std::size_t radius_search(const double* querypoint,
                            double radius,
                            RadiusSearchResult& result) const;

  /** @brief Perform radius search around given query point exporting distance
   * information */
  std::size_t radius_search_with_distances(
    const double* querypoint,
    double radius,
    RadiusSearchDistanceResult& result) const;

  /** @brief Calculate the nearest neighbors for an entire point cloud */
  void nearest_neighbors(EigenPointCloudConstRef cloud,
                         NearestNeighborsResult& result,
                         int k) const;

  /** @brief Calculate the nearest neighbors with Euclidean distance for an
   * entire point cloud */
  void nearest_neighbors_with_distances(EigenPointCloudConstRef cloud,
                                        NearestNeighborsDistanceResult& result,
                                        int k) const;

  /** @brief Return the leaf parameter this octree has been built with */
  unsigned int get_leaf_parameter() const { return leaf_parameter; }
};

} // namespace py4dgeo
