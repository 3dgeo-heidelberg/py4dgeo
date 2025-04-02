#pragma once

#include <Eigen/Eigen>

#include <istream>
#include <memory>
#include <optional>
#include <ostream>
#include <utility>
#include <vector>

#include "py4dgeo.hpp"

namespace py4dgeo {

// Expand BITS-bit value into (BITS x 3)-bit interleaved format
template<typename T, unsigned int BITS>
constexpr T
dilate(T x)
{
  T result = 0;
  for (unsigned int i = 0; i < BITS; ++i) {
    result |= ((x >> i) & 1ULL) << (3 * i);
  }
  return result;
}

// Helper to build a lookup table of dilated integers
template<typename T, unsigned int BITS, size_t TABLE_SIZE = (1ULL << BITS)>
static constexpr std::array<T, TABLE_SIZE>
make_dilate_table()
{
  std::array<T, TABLE_SIZE> table = {};
  for (size_t i = 0; i < TABLE_SIZE; ++i) {
    table[i] = dilate<T, BITS>(static_cast<T>(i));
  }
  return table;
}

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
  //! Alias for the spatial key type used for Z-order value encoding
  using SpatialKey = uint64_t; // 16-bit allows 5 depth levels, 32-bit allows 10
                               // levels, 64-bit allows 21 levels

  //! Return type used for points
  using PointContainer = std::vector<IndexType>;

  //! Return type used for cell searches
  using KeyContainer = std::vector<SpatialKey>;

  //! Struct combining Z-order value and original point index
  struct IndexAndKey
  {
    KeyContainer keys;      //!< Z-order values
    PointContainer indices; //!< Indices of the corresponding points in cloud
  };

  //! Return type used for radius searches
  using RadiusSearchResult = std::vector<IndexType>;

  //! Return type used for radius searches that export calculated distances
  using RadiusSearchDistanceResult = std::vector<std::pair<IndexType, double>>;

  //! Return type used for nearest neighbor with Euclidian distances searches
  using NearestNeighborsDistanceResult =
    std::vector<std::pair<std::vector<IndexType>, std::vector<double>>>;

  //! Return type used for nearest neighbor searches
  using NearestNeighborsResult = std::vector<std::vector<IndexType>>;

private:
  //! Enum to describe the geometric relationship between a cell and a sphere.
  enum class cell_relation_to_sphere
  {
    Inside,
    Intersecting,
    Outside
  };

  //! Reference to the point cloud
  EigenPointCloudRef cloud;
  //! Number of points in the cloud
  unsigned int number_of_points = 0;

  //! Pairs of spatial key (Z-order values) and corresponding index, sorted by
  //! z-order value
  IndexAndKey indexed_keys;

  //! Min point of the bounding box
  Eigen::Vector3d min_point;
  //! Max point of the bounding box
  Eigen::Vector3d max_point;
  //! Size of the bounding box
  Eigen::Vector3d box_size;
  //! Inverse of box size
  Eigen::Vector3d inv_box_size;

  //! Max depth level, depends solely on spatial key integer representation
  static constexpr unsigned int max_depth = (sizeof(SpatialKey) * 8) / 3;

  //! Number of cells per axis at the lowest level (2^max_depth)
  inline static constexpr unsigned int grid_size = (1 << max_depth);

  //! Bit shift per level
  inline static constexpr std::array<SpatialKey, max_depth + 1> bit_shift =
    []() {
      std::array<SpatialKey, max_depth + 1> arr{};
      for (size_t level = 0; level <= max_depth; ++level) {
        arr[level] = 3 * (max_depth - level);
      }
      return arr;
    }();

  //! Generic 8-bit dilation table already built:
  static constexpr auto dilate8_table =
    make_dilate_table<SpatialKey, 8>(); // 256 entries

  //! Generic 5-bit dilation table already built:
  static constexpr auto dilate5_table =
    make_dilate_table<SpatialKey, 5>(); // 32 entries

  //! Generic 2-bit dilation table already built:
  static constexpr auto dilate2_table =
    make_dilate_table<SpatialKey, 2>(); // 4 entries

  //! Cell size as a function of depth level
  std::array<Eigen::Vector3d, max_depth + 1> cell_size;

  //! Number of occupied cells per depth level
  std::array<unsigned int, max_depth + 1> occupied_cells_per_level;

  //! Maximum number of points per depth level
  std::array<unsigned int, max_depth + 1> max_cell_population_per_level;

  //! Average number of points per depth level
  std::array<double, max_depth + 1> average_cell_population_per_level;

  //! Standard deviation of points per depth level
  std::array<double, max_depth + 1> std_cell_population_per_level;

  //! Allow the Epoch class to directly call the private constructor
  friend Epoch;

  //! Private constructor from point cloud - use through @ref Octree::create
  Octree(const EigenPointCloudRef&);

private:
  /**
   * @brief Computes the smallest power-of-two bounding box that contains all
   * points.
   *
   * This function calculates the minimum and maximum coordinates of the point
   * cloud and determines a cuboid that fully encloses the data. The box size is
   * rounded up to the nearest power of two.
   *
   * The computed bounding box is stored in `min_point`, `max_point`, and
   * `box_size`.
   */
  void compute_bounding_box();

  /** @brief Computes the average cell properties at all depth levels. */
  void compute_statistics();

  /**
   * @brief Computes a unique spatial key (Z-order value) for a given point.
   *
   * This function normalizes the input point within the bounding box and maps
   * it to an integer 3D grid of size '2^max_depth × 2^max_depth × 2^max_depth'.
   * The computed grid coordinates are then interleaved using bitwise operations
   * to generate a Z-order value key that represents the point's hierarchical
   * position in the Octree.
   *
   * @param point The 3D point for which to compute the spatial key.
   *
   * @return A SpatialKey (unsigned integer) representing the spatial key of the
   * point.
   */
  inline SpatialKey compute_spatial_key(const Eigen::Vector3d& point) const
  {
    Eigen::Vector3d normalized = (point - min_point).cwiseProduct(inv_box_size);

    SpatialKey ix = static_cast<SpatialKey>(normalized.x() * grid_size);
    SpatialKey iy = static_cast<SpatialKey>(normalized.y() * grid_size);
    SpatialKey iz = static_cast<SpatialKey>(normalized.z() * grid_size);

    // Interleave bits of x,y,z coordinates
    SpatialKey key = 0;

    if constexpr (sizeof(SpatialKey) == 8) {
      // For a 64-bit SpatialKey (using 21 bits per coordinate):
      // Lower 8-bit chunk (bits 0-7)
      key |= dilate8_table[ix & 0xFF];
      key |= dilate8_table[iy & 0xFF] << 1;
      key |= dilate8_table[iz & 0xFF] << 2;
      // Next 8-bit chunk (bits 8-15)
      key |= dilate8_table[(ix >> 8) & 0xFF] << 24;
      key |= dilate8_table[(iy >> 8) & 0xFF] << 25;
      key |= dilate8_table[(iz >> 8) & 0xFF] << 26;
      // Final 5-bit chunk (bits 16-20)
      key |= dilate5_table[(ix >> 16) & 0x1F] << 48;
      key |= dilate5_table[(iy >> 16) & 0x1F] << 49;
      key |= dilate5_table[(iz >> 16) & 0x1F] << 50;
    } else if (sizeof(SpatialKey) == 4) {
      // For a 32-bit SpatialKey (using 10 bits per coordinate):
      // Lower 8-bit chunk (bits 0-7)
      key |= dilate8_table[ix & 0xFF];
      key |= dilate8_table[iy & 0xFF] << 1;
      key |= dilate8_table[iz & 0xFF] << 2;
      // Final 2-bit chunk (bits 8-9)
      key |= dilate2_table[(ix >> 8) & 0x03] << 24;
      key |= dilate2_table[(iy >> 8) & 0x03] << 25;
      key |= dilate2_table[(iz >> 8) & 0x03] << 26;
    }

    return key;
  }

public:
  /** @brief Construct instance of Octree from a given point cloud
   *
   * This is implemented as a static function instead of a public constructor
   * to ease the implementation of Python bindings.
   *
   * @param cloud The point cloud to construct the search tree for
   */
  static Octree create(const EigenPointCloudRef& cloud);

  /** @brief Save the index to a (file) stream */
  std::ostream& saveIndex(std::ostream& stream) const;

  /** @brief Load the index from a (file) stream */
  std::istream& loadIndex(std::istream& stream);

  /** @brief Build the Octree index
   *
   * This initializes the Octree search index. Calling this method is required
   * before performing any nearest neighbors or radius searches.
   */
  void build_tree();

  /**
   * @brief Clears the Octree structure, effectively resetting it.
   *
   * This function deallocates the Octree by clearing the sorted array of
   * indices and keys. This operation invalidates the current Octree, requiring
   * it to be rebuilt before further use.
   */
  void invalidate();

  /** @brief Perform radius search around given query point
   *
   * This method determines all the points from the point cloud within the given
   * radius of the query point. It returns only the indices and the result is
   * not sorted according to distance.
   *
   * @param[in] query_point A pointer to the 3D coordinate of the query point
   * @param[in] radius The radius to search within
   * @param[in] level The depth level at which to perform the search
   * @param[out] result A data structure to hold the result. It will be cleared
   * during application.
   *
   */
  std::size_t radius_search(const Eigen::Vector3d& query_point,
                            double radius,
                            unsigned int level,
                            RadiusSearchResult& result) const;

  /** @brief Return the number of points in the associated cloud */
  inline unsigned int get_number_of_points() const { return number_of_points; };

  /** @brief Return the side length of the bounding box */
  inline Eigen::Vector3d get_box_size() const { return box_size; };

  /** @brief Return the minimum point of the bounding box */
  inline Eigen::Vector3d get_min_point() const { return min_point; };

  /** @brief Return the maximum point of the bounding box */
  inline Eigen::Vector3d get_max_point() const { return max_point; };

  /** @brief Return the 8-bit delater lookup table*/
  inline SpatialKey get_dilate8_table(unsigned int i) const
  {
    return dilate8_table[i];
  };

  /** @brief Return the 5-bit delater lookup table*/
  inline SpatialKey get_dilate5_table(unsigned int i) const
  {
    return dilate5_table[i];
  };

  /** @brief Return the 2-bit delater lookup table*/
  inline SpatialKey get_dilate2_table(unsigned int i) const
  {
    return dilate2_table[i];
  };

  /** @brief Return the size of cells at a level of depth */
  inline Eigen::Vector3d get_cell_size(unsigned int level) const
  {
    return cell_size[level];
  };

  /** @brief Return the number of occupied cells per level of depth */
  inline unsigned int get_occupied_cells_per_level(unsigned int level) const
  {
    return occupied_cells_per_level[level];
  };

  /** @brief Return the number of occupied cells per level of depth */
  inline unsigned int get_max_cell_population_per_level(
    unsigned int level) const
  {
    return max_cell_population_per_level[level];
  };

  /** @brief Return the average number of points per level of depth */
  inline double get_average_cell_population_per_level(unsigned int level) const
  {
    return average_cell_population_per_level[level];
  };

  /** @brief Return the standard deviation of number of points per level of
   * depth */
  inline double get_std_cell_population_per_level(unsigned int level) const
  {
    return std_cell_population_per_level[level];
  };

  //! @brief Get all spatial keys (Z-order values) of the Octree
  //! @return Vector of spatial keys (Z-order values)
  const KeyContainer& get_spatial_keys() const { return indexed_keys.keys; }

  //! @brief Get all point indices corresponding to spatial keys
  //! @return Vector of point indices
  const PointContainer& get_point_indices() const
  {
    return indexed_keys.indices;
  }

  /**
   * @brief Returns the first occurrence of theindex of a cell in the sorted
   * array of point indices and point spatial keys
   *
   * @param key The spatial key of the query cell
   * @param bitShift The bit shift corresponding to the Octree depth level of
   * the query cell
   * @param start_index Optional start index
   *
   * @return The index of first occurrence of the cell spatial key
   */
  std::optional<IndexType> get_cell_index_start(
    SpatialKey truncated_key,
    SpatialKey bitShift,
    IndexType start_index = 0) const;

  /**
   * @brief Returns the last occurrence of theindex of a cell in the sorted
   * array of point indices and point spatial keys
   *
   * @param key The spatial key of the query cell
   * @param bitShift The bit shift corresponding to the Octree depth level of
   * the query cell
   * @param start_index Optional start index
   *
   * @return The index of last occurrence of the cell spatial key
   */
  std::optional<IndexType> get_cell_index_end(SpatialKey truncated_key,
                                              SpatialKey bitShift,
                                              IndexType start_index = 0) const;

  /**
   * @brief Returns spatial keys of cells intersected by a sphere with specified
   * radius with it's center at the query point
   *
   * @param[in] query_point A reference to  of the query point
   * @param[in] radius The radius to search within
   * @param[in] level The depth level to be considered
   * @param[out] inside A vector of spatial keys of the cells entirely in the
   * sphere
   * @param[out] intersecting A vector of spatial keys of the intersected cells
   *
   * @return The spatial keys of the intersected cells
   */
  std::size_t get_cells_intersected_by_sphere(
    const Eigen::Vector3d& query_point,
    double radius,
    unsigned int level,
    KeyContainer& inside,
    KeyContainer& intersecting) const;

  /**
   * @brief Returns indices and spatial keys of points lying in
   * multiple cells on a specified depth level
   *
   * @param[in] truncated_keys The spatial keys of the query cell, truncated to
   * the level of depth
   * @param[in] level The Octree depth level of the query cell
   * @param[out] result A data structure to hold the result. It will be cleared
   * during application.
   *
   * @return The amount of points in the return set
   */
  std::size_t get_points_indices_from_cells(const KeyContainer& truncated_keys,
                                            unsigned int level,
                                            RadiusSearchResult& result) const;

  /**
   * @brief Returns the level of depth at which a radius search will be most
   * efficient
   *
   * @param radius The radius of the search sphere
   *
   * @return The depth level at which to perform a radius search
   */
  unsigned int find_appropriate_level_for_radius_search(double radius) const;
};

} // namespace py4dgeo
