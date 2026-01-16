#pragma once

#include <py4dgeo/py4dgeo.hpp>
#include <py4dgeo/searchtree.hpp>

#include <Eigen/Core>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace py4dgeo {

/**
 * @brief Dilate (bit-spread) a small integer so its bits occupy every third bit
 * position.
 *
 * This function takes a BITS-bit integer x and returns a value in which
 * each original bit has been moved to position (3*i). Two zero bits are
 * inserted between every bit of the input. This operation is the core of
 * Morton (Z-order) encoding, where the x, y, and z bits of a coordinate
 * are interleaved to form a single integer key.
 *
 * Example (for BITS = 4):
 *      input:      b3 b2 b1 b0
 *      output:     0 0 b3  0 0 b2  0 0 b1  0 0 b0
 *
 * The result corresponds to the x-axis contribution of a Morton code
 * before interleaving with y and z bits.
 */
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

/**
 * @brief Generate a compile-time lookup table of dilated integers.
 *
 * This function builds a table of size (1 << BITS), where entry i
 * contains dilate(i). The table allows Morton encoding to be done
 * efficiently in large fixed-size chunks (e.g., 8 bits or 5 bits at a time),
 * instead of looping over individual bits.
 *
 * These tables are computed at compile time (constexpr) and enable the
 * encoder to construct full Morton keys using a handful of table lookups
 * and bit shifts, which is significantly faster than iterative bit
 * interleaving.
 */
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

// Forward declaration of OctreeTestAccess
struct OctreeTestAccess;

/** @brief Efficient Octree data structure for nearest neighbor/radius searches
 *
 * This data structure allows efficient radius searches in 3D point cloud data.
 * Unlike KDTree, it recursively subdivides space into eight octants.
 */
class Octree
{
  friend struct OctreeTestAccess;

public:
  // ==========================================================
  //                            Types
  // ==========================================================

  //! Alias for the spatial key type used for Z-order value encoding
  using SpatialKey =
    std::uint32_t; // 32-bit allows 10 levels, 64-bit allows 21 levels
  static_assert(std::is_same_v<SpatialKey, std::uint32_t> ||
                  std::is_same_v<SpatialKey, std::uint64_t>,
                "SpatialKey must be uint32_t or uint64_t");

  //! Return type used for points
  using PointContainer = std::vector<IndexType>;

  //! Return type used for cell searches
  using KeyContainer = std::vector<SpatialKey>;

  //! Coordinate within the octree
  using OctreeCoordinate = std::array<uint32_t, 3>;

  //! Coordinates within the octree
  using OctreeCoordinates =
    Eigen::Matrix<uint32_t, Eigen::Dynamic, 3, Eigen::RowMajor>;

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

  //! Invalid spatial key constant
  static constexpr SpatialKey invalid_key =
    std::numeric_limits<SpatialKey>::max();

  //! Max depth level, depends solely on spatial key integer representation
  static constexpr unsigned int max_depth = (sizeof(SpatialKey) * 8) / 3;

  //! Number of cells per axis at the lowest level (2^max_depth)
  static constexpr unsigned int grid_size = (1 << max_depth);

  //! Bit shift per level
  static constexpr std::array<SpatialKey, max_depth + 1> bit_shift = []() {
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
  static constexpr auto dilate8_table =
    make_dilate_table<SpatialKey, 8>(); // 256 entries

  //! Lookup table for dilating 5-bit spatial keys (32 entries), computed at
  //! compile time
  static constexpr auto dilate5_table =
    make_dilate_table<SpatialKey, 5>(); // 32 entries

  //! Lookup table for dilating 2-bit spatial keys (4 entries), computed at
  //! compile time
  static constexpr auto dilate2_table =
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

  //! Number of cells as a function of depth level
  static constexpr std::array<SpatialKey, max_depth + 1> number_of_cells =
    []() constexpr {
      std::array<SpatialKey, max_depth + 1> a{};
      for (unsigned int i = 0; i <= max_depth; ++i)
        a[i] = SpatialKey(1) << 3 * i;
      return a;
    }();

  //! Number of cells per axis as a function of depth level
  static constexpr std::array<uint32_t, max_depth + 1>
    number_of_cells_per_axis = []() constexpr {
      std::array<uint32_t, max_depth + 1> a{};
      for (unsigned int i = 0; i <= max_depth; ++i)
        a[i] = uint32_t(1) << i;
      return a;
    }();

  //! Number of occupied cells per depth level
  std::array<unsigned int, max_depth + 1> occupied_cells;

  //! Maximum number of points per depth level
  std::array<unsigned int, max_depth + 1> max_cell_population;

  //! Average number of points per depth level
  std::array<double, max_depth + 1> average_cell_population;

  //! Standard deviation of points per depth level
  std::array<double, max_depth + 1> std_cell_population;

  //! Allow the Epoch class to directly call the private constructor
  friend Epoch;

private:
  //! Private constructor from point cloud - use through @ref Octree::create
  Octree(const EigenPointCloudRef&);

  // =============================================================
  //                 Key computation / decoding
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

    return compute_spatial_key(ix, iy, iz);
  }

  /**
   * @brief Computes a unique spatial key (Z-order value) for a given octree
   * cell.
   *
   * @param query_cell The 3D octree coordinate for which to compute the spatial
   * key.
   *
   * @return A SpatialKey (unsigned integer) representing the spatial key of the
   * point.
   */
  inline SpatialKey compute_spatial_key(const OctreeCoordinate& query_cell,
                                        unsigned int level) const
  {
    const unsigned int shift = max_depth - level;

    const SpatialKey ix = SpatialKey(query_cell[0]) << shift;
    const SpatialKey iy = SpatialKey(query_cell[1]) << shift;
    const SpatialKey iz = SpatialKey(query_cell[2]) << shift;

    return compute_spatial_key(ix, iy, iz);
  }

  /**
   * @brief Computes a unique spatial key (Z-order value) for a given point.
   *
   * This function takes integer grid coordinates and interleaves their bits
   * to generate a Z-order value key that represents the point's hierarchical
   * position in the Octree.
   *
   * @param ix The integer x-coordinate in the grid.
   * @param iy The integer y-coordinate in the grid.
   * @param iz The integer z-coordinate in the grid.
   * @return A SpatialKey (unsigned integer) representing the spatial key of the
   * point.
   */
  inline SpatialKey compute_spatial_key(SpatialKey ix,
                                        SpatialKey iy,
                                        SpatialKey iz) const
  {
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

  /**
   * @brief Extracts and compacts every third bit from a Morton-encoded word.
   *
   * Morton (Z-order) encoding interleaves the bits of three coordinates:
   *    x0 y0 z0 x1 y1 z1 x2 y2 z2 ...
   *
   * Given a 64-bit integer `v` that represents one of these three shifted
   * Morton streams (i.e., either `key >> 0`, `key >> 1`, or `key >> 2`), this
   * function isolates the bits belonging to one coordinate and compacts them
   * into contiguous low-order bits:
   *
   *    input bits:  v = _ _ x2 _ _ x1 _ _ x0  (spread every 3rd bit)
   *    output:          x = 0 0 0 x2 x1 x0   (packed)
   *
   * The implementation applies a sequence of bit masks and shift-OR operations.
   * Each step reduces the spacing between valid bits while preventing bits from
   * moving across structural boundaries. This is the standard high-performance
   * Morton decode algorithm used in libmorton, Embree, and OptiX.
   *
   * The routine works for both 32-bit and 64-bit Morton keys: if `v` originates
   * from a 32-bit Morton code, all upper bits are zero and the additional
   * compaction steps simply have no effect. No separate 32-bit implementation
   * is required.
   *
   * @param v  The shifted Morton stream (x = key>>0, y = key>>1, z = key>>2).
   * @return   The compacted coordinate (up to 22 bits for max depth 21).
   */
  static inline uint64_t compact3(uint64_t v)
  {
    v &= 0x9249249249249249ULL;
    v = (v | (v >> 2)) & 0x30c30c30c30c30c3ULL;
    v = (v | (v >> 4)) & 0xf00f00f00f00f00fULL;
    v = (v | (v >> 8)) & 0x00ff0000ff0000ffULL;
    v = (v | (v >> 16)) & 0x00ff00000000ffffULL;
    v = (v | (v >> 32)) & 0x00000000003fffffULL;
    return v;
  }

  /**
   * @brief Decodes a spatial key into integer coordinates.
   *
   * A Morton (Z-order) key interleaves the bits of three coordinates x, y, z:
   *
   *     morton = z2 y2 x2  z1 y1 x1  z0 y0 x0 ...
   *
   * This function extracts each bit stream by shifting the Morton key:
   *     x-bits: key >> 0
   *     y-bits: key >> 1
   *     z-bits: key >> 2
   *
   * and then compacts them using `compact3()`. The result is the integer
   * octree-grid coordinate of the point at the *maximum* tree depth. If a
   * coarser level is required, the caller can right-shift the results by
   * (max_depth - level).
   *
   * Works for both 32-bit and 64-bit Morton keys. Larger word sizes simply
   * provide more Morton layers; compact3() automatically ignores unused bits.
   *
   * @param key  The Morton-encoded spatial key.
   *
   * @return The decoded (x, y, z) octree-grid coordinate at maximum depth.
   */
  static inline OctreeCoordinate decode_spatial_key(SpatialKey key)
  {
    return { static_cast<uint32_t>(compact3(key >> 0)),
             static_cast<uint32_t>(compact3(key >> 1)),
             static_cast<uint32_t>(compact3(key >> 2)) };
  }

  /**
   * @brief Decode a spatial key into (x, y, z) coordinates at a given octree
   * level.
   *
   * The input @p SpatialKey is a Morton-encoded (Z-order) spatial key where the
   * bits of x, y, and z are interleaved as:
   *
   *     ... z2 y2 x2  z1 y1 x1  z0 y0 x0
   *
   * At maximum depth (level == max_depth), decode_spatial_key() returns the
   * full-resolution integer coordinates (max_depth bits per axis).
   *
   * For coarser octree levels, this function first discards the least
   * significant Morton "triples" (x,y,z) by shifting the key to the right by
   *
   *     bit_shift[level] = 3 * (max_depth - level)
   *
   * which removes all detail below the requested level while preserving the
   * x/y/z bit-lane alignment. The truncated code is then decoded via
   * decode_spatial_key(), yielding the cell coordinates at the requested level.
   *
   * In other words:
   *
   *   - level == max_depth: full-resolution coordinates
   *   - level  < max_depth: coordinates on a coarser 2^level grid
   *
   * @param key Morton-encoded spatial key.
   * @param level Target octree level in [0, max_depth].
   *
   * @return (x, y, z) octree-grid coordinates at the specified level.
   */
  inline OctreeCoordinate decode_spatial_key_at_level(SpatialKey key,
                                                      unsigned int level) const
  {
    return decode_spatial_key(key >> bit_shift[level]);
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

  /** @brief Computes the average cell properties of occupied cells at all depth
   * levels. */
  void compute_statistics();

  // ================================================================
  //                       Radius search backend
  // ================================================================

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
                                 max_cell_population[level]);
    candidate_points.reserve(cells_intersecting.size() *
                             max_cell_population[level]);
    get_point_indices_from_cells(cells_inside, level, points_inside_sphere);
    get_point_indices_from_cells(cells_intersecting, level, candidate_points);

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

  // ======================================================================
  //                           Other helpers
  // ======================================================================

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
   * @brief Compute the index range of points belonging to a single octree cell
   *
   * Given a truncated spatial key identifying an octree cell at a given depth
   * level, this function locates the contiguous block of point indices
   * belonging to that cell in the internally sorted index structure.
   *
   * The returned index range is half-open: [first, last).
   *
   * @param[in] truncated_key Truncated spatial key identifying the query cell
   * @param[in] level Octree depth level of the query cell
   * @param[in] search_start Lower bound index for the search window
   * @param[in] search_end Upper bound index for the search window (exclusive)
   *
   * @return Optional pair (first, last) describing the index range of points in
   *         the cell, or std::nullopt if the cell contains no points
   */
  std::optional<std::pair<IndexType, IndexType>> get_cell_index_range(
    SpatialKey truncated_key,
    unsigned int level,
    IndexType search_start,
    IndexType search_end) const;

  /**
   * @brief Append indices of points belonging to a single octree cell
   *
   * Finds all points contained in the specified octree cell and appends their
   * point indices to the provided result container.
   *
   * The search window is advanced to start after the current cell's block,
   * enabling efficient sequential processing of sorted cell queries.
   *
   * @param[in] truncated_key Truncated spatial key identifying the query cell,
   * truncated to the specified depth level
   * @param[in] level Octree depth level of the query cell
   * @param[in,out] current_start_index Start index for the search window;
   * updated to the end of the current cell's block
   * @param[in] search_limit_index Upper bound of the search window (exclusive)
   * @param[in,out] result Container to which point indices are appended
   *
   * @return True if the cell contains points, false otherwise
   */
  bool append_points_from_cell(SpatialKey truncated_key,
                               unsigned int level,
                               IndexType& current_start_index,
                               IndexType search_limit_index,
                               RadiusSearchResult& result) const;

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

  // ================================================================
  //                      Public search methods
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
  //                         Metadata Getters
  // =================================================================

  /** @brief Return the number of points in the associated cloud */
  inline unsigned int get_number_of_points() const { return number_of_points; };

  /** @brief Return the maximum octree depth */
  constexpr unsigned int get_max_depth() const { return max_depth; };

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

  /** @brief Return the number of cells at a level of depth */
  constexpr unsigned int get_number_of_cells(unsigned int level) const
  {
    if (level > max_depth) {
      throw std::out_of_range("Requested level " + std::to_string(level) +
                              " exceeds the maximum depth of " +
                              std::to_string(max_depth) + ".");
    }
    return number_of_cells[level];
  };

  /** @brief Return the number of cells per axis at a level of depth */
  constexpr unsigned int get_number_of_cells_per_axis(unsigned int level) const
  {
    if (level > max_depth) {
      throw std::out_of_range("Requested level " + std::to_string(level) +
                              " exceeds the maximum depth of " +
                              std::to_string(max_depth) + ".");
    }
    return number_of_cells_per_axis[level];
  };

  /** @brief Return the number of occupied cells per level of depth */
  inline unsigned int get_number_of_occupied_cells(unsigned int level) const
  {
    if (level > max_depth) {
      throw std::out_of_range("Requested level " + std::to_string(level) +
                              " exceeds the maximum depth of " +
                              std::to_string(max_depth) + ".");
    }
    return occupied_cells[level];
  };

  /** @brief Return the maximum cell population per level of depth */
  inline unsigned int get_max_cell_population(unsigned int level) const
  {
    if (level > max_depth) {
      throw std::out_of_range("Requested level " + std::to_string(level) +
                              " exceeds the maximum depth of " +
                              std::to_string(max_depth) + ".");
    }
    return max_cell_population[level];
  };

  /** @brief Return the average number of points per level of depth */
  inline double get_average_cell_population(unsigned int level) const
  {
    if (level > max_depth) {
      throw std::out_of_range("Requested level " + std::to_string(level) +
                              " exceeds the maximum depth of " +
                              std::to_string(max_depth) + ".");
    }
    return average_cell_population[level];
  };

  /** @brief Return the standard deviation of number of points per level of
   * depth */
  inline double get_std_cell_population(unsigned int level) const
  {
    if (level > max_depth) {
      throw std::out_of_range("Requested level " + std::to_string(level) +
                              " exceeds the maximum depth of " +
                              std::to_string(max_depth) + ".");
    }
    return std_cell_population[level];
  };

  // ======================================================================
  //                      Queries - Get Cell Population
  // ======================================================================

  /**
   * @brief Returns the number of points lying in a single octree cell at a
   * specified depth level
   *
   * @param[in] truncated_key Truncated spatial key identifying the query cell
   * @param[in] level Octree depth level of the query cell
   *
   * @return The number of points in the specified cell
   */
  std::size_t get_cell_population(SpatialKey truncated_key,
                                  unsigned int level) const;

  /**
   * @brief Returns the cell populations for multiple octree cells at a
   * specified depth level
   *
   * @param[in] truncated_keys Truncated spatial keys identifying the query
   * cells
   * @param[in] level Octree depth level of the query cell
   *
   * @return The number of points in the specified cells
   */
  std::vector<std::size_t> get_cell_population(
    const KeyContainer& truncated_keys,
    unsigned int level) const;

  // ======================================================================
  //                      Queries - Get Point Indices
  // ======================================================================

  /**
   * @brief Return indices of points lying in a single octree cell at a
   * specified depth level
   *
   * Given a spatial key identifying the octree cell (truncated to the same
   * depth level), this function gathers all point indices belonging to this
   * cell.
   *
   * The search exploits the sorted order of the input keys to avoid redundant
   * scans and achieve efficient sequential access.
   *
   * @param[in] truncated_key Spatial keys of the query cell, truncated to the
   * specified depth level
   * @param[in] level Octree depth level of the query cells
   * @param[out] result Container to hold the resulting point indices.
   * Existing contents will be preserved and appended to.
   *
   * @return The total number of points appended to the result container
   */
  std::size_t get_point_indices_from_cells(SpatialKey truncated_key,
                                           unsigned int level,
                                           RadiusSearchResult& result) const;

  /**
   * @brief Return indices of points lying in multiple octree cells at a
   * specified depth level
   *
   * Given a sorted list of spatial keys identifying octree cells (truncated
   * to the same depth level), this function gathers all point indices
   * belonging to those cells.
   *
   * The search exploits the sorted order of the input keys to avoid redundant
   * scans and achieve efficient sequential access.
   *
   * @param[in] truncated_keys Sorted spatial keys of the query cells,
   * truncated to the specified depth level
   * @param[in] level Octree depth level of the query cells
   * @param[out] result Container to hold the resulting point indices.
   * Existing contents will be preserved and appended to.
   *
   * @return The total number of points appended to the result container
   */
  std::size_t get_point_indices_from_cells(const KeyContainer& truncated_keys,
                                           unsigned int level,
                                           RadiusSearchResult& result) const;

  // ======================================================================
  //                           Get Unique Cells
  // ======================================================================

  /** @brief Returns the unique spatial keys of occupied cells at a specified
   * depth level
   *
   * @param[in] level The depth level at which to retrieve the unique cells
   * @return A container of unique (truncated) spatial keys of occupied
   * cells
   */
  KeyContainer get_unique_cells(unsigned int level) const;

  // ======================================================================
  //                        Get Octree Coordinates
  // ======================================================================

  /**
   * @brief Return the octree-grid coordinate (x,y,z) of a spatial key at a
   * given level.
   *
   * If no level is provided, the coordinate at maximum depth is returned.
   */
  OctreeCoordinate get_coordinates(SpatialKey truncated_key) const;

  /**
   * @brief Return the octree-grid coordinates (x,y,z) of a set of spatial
   * keys at a given level.
   *
   * If no level is provided, the coordinates at maximum depth is returned.
   */
  OctreeCoordinates get_coordinates(const KeyContainer& truncated_keys) const;

  /**
   * @brief Return the octree-grid coordinates (x,y,z) of all points at a given
   * level.
   *
   * If no level is provided, the coordinates at maximum depth is returned.
   */
  OctreeCoordinates get_coordinates_at_level(unsigned int level) const;

  // ======================================================================
  //                           Other helpers
  // ======================================================================

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
