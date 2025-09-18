#pragma once

#include <py4dgeo/py4dgeo.hpp>
#include <py4dgeo/searchtree.hpp>

#include <Eigen/Core>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
template<typename T, unsigned int BITS, std::size_t TABLE_SIZE = (1ULL << BITS)>
static constexpr std::array<T, TABLE_SIZE>
make_dilate_table()
{
  std::array<T, TABLE_SIZE> table = {};
  for (std::size_t i = 0; i < TABLE_SIZE; ++i) {
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
  // ==========================================================
  //                            Types
  // ==========================================================

  //! Alias for the spatial key type used for Z-order value encoding
  using SpatialKey = std::uint32_t; // 16-bit allows 5 depth levels, 32-bit
                                    // allows 10 levels, 64-bit allows 21 levels

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

private:
  //! Enum to describe the geometric relationship between a cell and a sphere.
  enum class cell_relation_to_sphere
  {
    Inside,
    Intersecting,
    Outside
  };

  // ==========================================================
  //                          Constants
  // ==========================================================

  //! Max depth level, depends solely on spatial key integer representation
  static constexpr unsigned int max_depth = (sizeof(SpatialKey) * 8) / 3;

  //! Number of cells per axis at the lowest level (2^max_depth)
  inline static constexpr unsigned int grid_size = (1 << max_depth);

  //! Bit shift per level
  inline static constexpr std::array<SpatialKey, max_depth + 1> bit_shift =
    []() {
      std::array<SpatialKey, max_depth + 1> arr{};
      for (std::size_t level = 0; level <= max_depth; ++level) {
        arr[level] = 3 * (max_depth - level);
      }
      return arr;
    }();

  // Precomputed dilation tables for fast spatial key encoding.
  // These tables map compact N-bit indices to interleaved bit patterns.
  // Used in compute_spatial_key(), avoiding runtime bit-manipulation.

  //! Lookup table for dilating 8-bit spatial keys (256 entries), computed at
  //! compile time.
  inline static constexpr auto dilate8_table =
    make_dilate_table<SpatialKey, 8>(); // 256 entries

  //! Lookup table for dilating 5-bit spatial keys (32 entries), computed at
  //! compile time
  inline static constexpr auto dilate5_table =
    make_dilate_table<SpatialKey, 5>(); // 32 entries

  //! Lookup table for dilating 2-bit spatial keys (4 entries), computed at
  //! compile time
  inline static constexpr auto dilate2_table =
    make_dilate_table<SpatialKey, 2>(); // 4 entries

  // ==========================================================
  //                     Bounding box info
  // ==========================================================

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

  // ============================================================
  //                     Per-level statistics
  // ============================================================

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
  // =============================================================
  //                        Key computation
  // =============================================================

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

    const SpatialKey ix = static_cast<SpatialKey>(normalized.x() * grid_size);
    const SpatialKey iy = static_cast<SpatialKey>(normalized.y() * grid_size);
    const SpatialKey iz = static_cast<SpatialKey>(normalized.z() * grid_size);

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
    } else if constexpr (sizeof(SpatialKey) == 4) {
      // For a 32-bit SpatialKey (using 10 bits per coordinate):
      // Lower 8-bit chunk (bits 0-7)
      key |= dilate8_table[ix & 0xFF];
      key |= dilate8_table[iy & 0xFF] << 1;
      key |= dilate8_table[iz & 0xFF] << 2;
      // Final 2-bit chunk (bits 8-9)
      key |= dilate2_table[(ix >> 8) & 0x03] << 24;
      key |= dilate2_table[(iy >> 8) & 0x03] << 25;
      key |= dilate2_table[(iz >> 8) & 0x03] << 26;
    } else {
      static_assert(
        sizeof(SpatialKey) == 4 || sizeof(SpatialKey) == 8,
        "SpatialKey must be either 32-bit (uint32_t) or 64-bit (uint64_t)");
    }

    return key;
  }

  // =============================================================
  //                         Tree building
  // =============================================================

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
   *
   * @param force_cubic If true, the bounding box will be forced to have equal
   * side lengths, resulting in a cubic shape.
   *
   * @param min_corner Optional minimum point (lower corner of the bounding
   * box). If not provided, the minimum point is computed automatically from the
   * input data.
   *
   * @param max_corner Optional maximum point (upper corner of the bounding
   * box). If not provided, the maximum point is computed automatically from the
   * input data.
   */
  void compute_bounding_box(
    bool force_cubic = false,
    std::optional<Eigen::Vector3d> min_corner = std::nullopt,
    std::optional<Eigen::Vector3d> max_corner = std::nullopt);

  /** @brief Computes the average cell properties at all depth levels. */
  void compute_statistics();

  // ================================================================
  //                         Lookup functions
  // ================================================================

  /** @brief Return the 8-bit dilated lookup table*/
  static constexpr SpatialKey get_dilate8_table(unsigned int i)
  {
    return dilate8_table[i];
  };

  /** @brief Return the 5-bit dilated lookup table*/
  static constexpr SpatialKey get_dilate5_table(unsigned int i)
  {
    return dilate5_table[i];
  };

  /** @brief Return the 2-bit dilated lookup table*/
  static constexpr SpatialKey get_dilate2_table(unsigned int i)
  {
    return dilate2_table[i];
  };

  /** @brief Return the bit shift per level */
  static constexpr unsigned int get_bit_shift(unsigned int level)
  {
    return bit_shift[level];
  };

  /**
   * @brief Perform radius search with a callback for each matching point
   *
   * Searches all points within the given radius around a query point at a
   * specified octree level. For each point inside the radius, the provided
   * callback is called with its index and distance.
   *
   * Used internally to implement both index-only and distance-returning
   * search variants.
   *
   * @tparam Callback A callable accepting (IndexType, double)
   * @param[in] query_point The 3D coordinates of the query point
   * @param[in] radius The search radius
   * @param[in] level The octree depth level
   * @param[in] check_candidate Called for each candidate point found within the
   * sphere
   * @param[in] take_all Called for all points in cells that are completely
   * within the sphere
   */
  template<typename Reserve,
           typename CallbackCandidate,
           typename CallbackInside>
  void radius_search_backend(const Eigen::Vector3d& query_point,
                             double radius,
                             unsigned int level,
                             Reserve&& reserve,
                             CallbackCandidate&& check_candidate,
                             CallbackInside&& take_all) const
  {
    constexpr std::size_t estimated_cell_count = 1024;

    // Step 1: Retrieve all spatial keys of cells intersected by the sphere
    KeyContainer cells_inside, cells_intersecting;
    cells_inside.reserve(estimated_cell_count);
    cells_intersecting.reserve(estimated_cell_count);
    get_cells_intersected_by_sphere(
      query_point, radius, level, cells_inside, cells_intersecting);

    // Step 2: Get candidate point indices from the intersected cells
    RadiusSearchResult points_inside_sphere, candidate_points;
    points_inside_sphere.reserve(cells_inside.size() *
                                 max_cell_population_per_level[level]);
    candidate_points.reserve(cells_intersecting.size() *
                             max_cell_population_per_level[level]);
    get_points_indices_from_cells(cells_inside, level, points_inside_sphere);
    get_points_indices_from_cells(cells_intersecting, level, candidate_points);

    // Step 3: Reserve based on size of cells_inside and cells_intersecting
    reserve(points_inside_sphere.size() + candidate_points.size());

    // Step 4: Check each candidate point
    double radius_squared = radius * radius;
    for (const auto& candidate : candidate_points) {
      const Eigen::Vector3d candidate_point = cloud.row(candidate);
      double squared_dist = (candidate_point - query_point).squaredNorm();
      if (squared_dist <= radius_squared) {
        check_candidate(candidate, squared_dist);
      }
    }

    // Step 5: Points from fully included cells do not need a distance check
    take_all(points_inside_sphere);
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

  /** @brief Build the Octree index
   *
   * This initializes the Octree search index. Calling this method is required
   * before performing any nearest neighbors or radius searches.
   *
   * @param force_cubic If true, the bounding box will be forced to have equal
   * side lengths, resulting in a cubic shape.
   *
   * @param min_corner Optional minimum point (lower corner of the bounding
   * box). If not provided, the minimum point is computed automatically from the
   * input data.
   *
   * @param max_corner Optional maximum point (upper corner of the bounding
   * box). If not provided, the maximum point is computed automatically from the
   * input data.
   *
   */
  void build_tree(bool force_cubic = false,
                  std::optional<Eigen::Vector3d> min_corner = std::nullopt,
                  std::optional<Eigen::Vector3d> max_corner = std::nullopt);

  /**
   * @brief Clears the Octree structure, effectively resetting it
   *
   * This function deallocates the Octree by clearing the sorted array of
   * indices and keys. This operation invalidates the current Octree, requiring
   * it to be rebuilt before further use.
   */
  void invalidate();

  /** @brief Save the index to a (file) stream */
  std::ostream& saveIndex(std::ostream& stream) const;

  /** @brief Load the index from a (file) stream */
  std::istream& loadIndex(std::istream& stream);

  // ================================================================
  //                          Search methods
  // ================================================================

  /** @brief Perform radius search around given query point
   *
   * This method determines all the points from the point cloud within the given
   * radius of the query point. It returns only the indices and the result is
   * not sorted according to distance.
   *
   * @param[in] query_point A pointer to the 3D coordinate of the query point
   * @param[in] radius The radius to search within
   * @param[in] level The depth level at which to perform the search
   * @param[out] result A data structure to hold the point indices. It will be
   * cleared during application.
   *
   * @return Number of points found
   */
  std::size_t radius_search(const Eigen::Vector3d& query_point,
                            double radius,
                            unsigned int level,
                            RadiusSearchResult& result) const;

  /** @brief Perform radius search around given query point
   *
   * This method determines all the points from the point cloud within the given
   * radius of the query point. It returns only the indices and the result is
   * not sorted according to distance.
   *
   * @param[in] query_point A pointer to the 3D coordinate of the query point
   * @param[in] radius The radius to search within
   * @param[in] level The depth level at which to perform the search
   * @param[out] result A data structure to hold the point indices and
   * corresponding distances. It will be cleared during application.
   *
   * @return Number of points found
   */
  std::size_t radius_search_with_distances(
    const Eigen::Vector3d& query_point,
    double radius,
    unsigned int level,
    RadiusSearchDistanceResult& result) const;

  // =================================================================
  //                           Metadata
  // =================================================================

  /** @brief Return the number of points in the associated cloud */
  inline unsigned int get_number_of_points() const { return number_of_points; };

  /** @brief Return the side length of the bounding box */
  inline Eigen::Vector3d get_box_size() const { return box_size; };

  /** @brief Return the minimum point of the bounding box */
  inline Eigen::Vector3d get_min_point() const { return min_point; };

  /** @brief Return the maximum point of the bounding box */
  inline Eigen::Vector3d get_max_point() const { return max_point; };

  // ======================================================================
  //                        Per-level statistics
  // ======================================================================

  /** @brief Return the size of cells at a level of depth */
  inline Eigen::Vector3d get_cell_size(unsigned int level) const
  {
    if (level > max_depth) {
      throw std::out_of_range("Requested level " + std::to_string(level) +
                              " exceeds the maximum depth of " +
                              std::to_string(max_depth) + ".");
    }
    return cell_size[level];
  };

  /** @brief Return the number of occupied cells per level of depth */
  inline unsigned int get_occupied_cells_per_level(unsigned int level) const
  {
    if (level > max_depth) {
      throw std::out_of_range("Requested level " + std::to_string(level) +
                              " exceeds the maximum depth of " +
                              std::to_string(max_depth) + ".");
    }
    return occupied_cells_per_level[level];
  };

  /** @brief Return the maximum cell population per level of depth */
  inline unsigned int get_max_cell_population_per_level(
    unsigned int level) const
  {
    if (level > max_depth) {
      throw std::out_of_range("Requested level " + std::to_string(level) +
                              " exceeds the maximum depth of " +
                              std::to_string(max_depth) + ".");
    }
    return max_cell_population_per_level[level];
  };

  /** @brief Return the average number of points per level of depth */
  inline double get_average_cell_population_per_level(unsigned int level) const
  {
    if (level > max_depth) {
      throw std::out_of_range("Requested level " + std::to_string(level) +
                              " exceeds the maximum depth of " +
                              std::to_string(max_depth) + ".");
    }
    return average_cell_population_per_level[level];
  };

  /** @brief Return the standard deviation of number of points per level of
   * depth */
  inline double get_std_cell_population_per_level(unsigned int level) const
  {
    if (level > max_depth) {
      throw std::out_of_range("Requested level " + std::to_string(level) +
                              " exceeds the maximum depth of " +
                              std::to_string(max_depth) + ".");
    }
    return std_cell_population_per_level[level];
  };

  // ======================================================================
  //                     Access raw keys and indices
  // ======================================================================

  //! @brief Get all spatial keys (Z-order values) of the Octree
  //! @return Vector of spatial keys (Z-order values)
  const KeyContainer& get_spatial_keys() const { return indexed_keys.keys; }

  //! @brief Get all point indices corresponding to spatial keys
  //! @return Vector of point indices
  const PointContainer& get_point_indices() const
  {
    return indexed_keys.indices;
  }

  // ======================================================================
  //                           Fine-grain helpers
  // ======================================================================

  /**
   * @brief Returns the first occurrence of the index of a cell in the sorted
   * array of point indices and point spatial keys
   *
   * @param key The spatial key of the query cell
   * @param bitShift The bit shift corresponding to the Octree depth level of
   * the query cell
   * @param start_index Start index
   * @param end_index End index

   *
   * @return The index of first occurrence of the cell spatial key
   */
  std::optional<IndexType> get_cell_index_start(SpatialKey truncated_key,
                                                SpatialKey bitShift,
                                                IndexType start_index,
                                                IndexType end_index) const;

  /**
   * @brief Returns the last occurrence of the index of a cell in the sorted
   * array of point indices and point spatial keys
   *
   * @param truncated_key The spatial key of the query cell
   * @param bitShift The bit shift corresponding to the Octree depth level of
   * the query cell
   * @param start_index Start index
   * @param end_index End index
   *
   * @return The index of last occurrence of the cell spatial key
   */
  std::optional<IndexType> get_cell_index_end(SpatialKey truncated_key,
                                              SpatialKey bitShift,
                                              IndexType start_index,
                                              IndexType end_index) const;

  /**
   * @brief Efficiently finds the end index of a block of points belonging to
   * the same truncated spatial key
   *
   * This function assumes the data is sorted by spatial keys and performs a
   * stride-based search to narrow the interval where the last matching key may
   * occur. The search begins from the known start index of a cell and advances
   * by the estimated average number of points per cell (based on depth level
   * statistics). Once a non-matching key is detected or the global bounds are
   * exceeded, a binary search is performed within the refined interval to find
   * the precise end.
   *
   * This hybrid approach combines data-aware range narrowing with binary search
   * for performance.
   *
   * @param truncated_key The spatial key (already bit-shifted) identifying the
   * target cell
   * @param level Octree depth level (used to get average cell population)
   * @param first_index Start index of the block in the sorted array
   * @param global_end_index Absolute upper bound for the search (typically the
   * size of the array)
   *
   * @return Index one past the last occurrence of the given spatial key, or
   * std::nullopt if not found
   */
  std::optional<IndexType> get_cell_index_end_exponential(
    SpatialKey truncated_key,
    unsigned int level,
    IndexType first_index,
    IndexType global_end_index) const;

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
   * @return The number of intersected cells and cells that are fully inside the
   * sphere
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
