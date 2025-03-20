#ifdef PY4DGEO_WITH_OPENMP
#include <omp.h>
#endif
#include <bitset> // Include for bitset

#include "py4dgeo/octree.hpp"
#include "py4dgeo/py4dgeo.hpp"

#include <vector>

namespace py4dgeo {

Octree::Octree(const EigenPointCloudRef& cloud)
  : cloud{ cloud }
{
}

void
Octree::compute_bounding_cube()
{
  // Find the smallest value in each column (x,y,z)
  min_point = cloud.colwise().minCoeff();

  // Find the biggest value in each column (x,y,z)
  max_point = cloud.colwise().maxCoeff();
  Eigen::Vector3d center = (min_point + max_point) * 0.5;

  // (max_point - min_point) gives the width, height and depth of the bounding
  // box
  double max_extent = (max_point - min_point).maxCoeff();

  // Gives the exponent x of 2^x needed, as ceiled integer
  cube_size = std::pow(2, std::ceil(std::log2(max_extent)));

  // Gives the corner of the min point
  min_point = center - Eigen::Vector3d::Constant(cube_size * 0.5);

  // Gives the corner of the max point
  max_point = center + Eigen::Vector3d::Constant(cube_size * 0.5);

  // Compute cell sizes
  cell_size[0] = cube_size;

  for (size_t i = 1; i <= max_depth; ++i) {
    cell_size[i] = cell_size[i - 1] * 0.5;
  }
}

void
Octree::compute_statistics()
{
  occupied_cells_per_level[0] = 1;
  max_cell_population_per_level[0] = static_cast<double>(number_of_points);
  ;
  average_cell_population_per_level[0] = static_cast<double>(number_of_points);
  std_cell_population_per_level[0] = 0.0;

  for (size_t level = 1; level <= max_depth; ++level) {

    unsigned int unique_cells = 0;
    unsigned int max_population = 0;
    double sum = 0.0;
    double sum2 = 0.0;
    unsigned int current_cell_population = 0;

    // Initialize with the first element's truncated key
    SpatialKey previous_key = indexed_keys[0].key >> bit_shift[level];
    current_cell_population = 1; // Count the first point

    for (IndexType i = 0; i < indexed_keys.size(); ++i) {
      SpatialKey current_key = indexed_keys[i].key >> bit_shift[level];
      if (current_key == previous_key) {
        current_cell_population++;
      } else {
        // New cell encountered
        unique_cells++;
        sum += current_cell_population;
        sum2 += static_cast<double>(current_cell_population) *
                current_cell_population;
        if (current_cell_population > max_population)
          max_population = current_cell_population;

        // Reset for the new cell
        previous_key = current_key;
        current_cell_population = 1;
      }
    }
    // Process the final cell
    unique_cells++;
    sum += current_cell_population;
    sum2 +=
      static_cast<double>(current_cell_population) * current_cell_population;
    if (current_cell_population > max_population)
      max_population = current_cell_population;

    // Store computed statistics for this level
    occupied_cells_per_level[level] = unique_cells;
    max_cell_population_per_level[level] = max_population;
    average_cell_population_per_level[level] =
      static_cast<double>(indexed_keys.size()) / unique_cells;
    std_cell_population_per_level[level] = std::sqrt(
      sum2 / unique_cells - average_cell_population_per_level[level] *
                              average_cell_population_per_level[level]);
  }
}

Octree::SpatialKey
Octree::compute_spatial_key(const Eigen::Vector3d& point) const
{
  Eigen::Vector3d normalized = (point - min_point) / cube_size;

  SpatialKey ix = static_cast<SpatialKey>(normalized.x() * grid_size);
  SpatialKey iy = static_cast<SpatialKey>(normalized.y() * grid_size);
  SpatialKey iz = static_cast<SpatialKey>(normalized.z() * grid_size);

  // Interleave bits of x,y,z coordinates
  SpatialKey key = 0;
  for (SpatialKey i = 0; i < max_depth; i++) {
    key |= ((ix >> i) & 1) << (3 * i);
    key |= ((iy >> i) & 1) << (3 * i + 1);
    key |= ((iz >> i) & 1) << (3 * i + 2);
  }
  return key;
}

Octree
Octree::create(const EigenPointCloudRef& cloud)
{
  return Octree(cloud);
}

void
Octree::build_tree()
{
  compute_bounding_cube();

  number_of_points = cloud.rows();
  indexed_keys.resize(number_of_points);

  // Step 1: Compute Z-order values and store point indices
  for (IndexType i = 0; i < number_of_points; ++i) {
    indexed_keys[i] = { compute_spatial_key(cloud.row(i)), i };
  }

  // Step 2: Sort the indexed keys by Z-order value
  std::sort(indexed_keys.begin(), indexed_keys.end());

  // Step 3: Compute related properties (max cell population, avg, ...)
  compute_statistics();
}

void
Octree::invalidate()
{
  number_of_points = 0;
  cube_size = 0.0;
  min_point = max_point = Eigen::Vector3d::Zero();

  indexed_keys.clear();
  indexed_keys.shrink_to_fit();

  std::fill(cell_size.begin(), cell_size.end(), 0.0);
  std::fill(
    occupied_cells_per_level.begin(), occupied_cells_per_level.end(), 0.0);
  std::fill(max_cell_population_per_level.begin(),
            max_cell_population_per_level.end(),
            0.0);
  std::fill(average_cell_population_per_level.begin(),
            average_cell_population_per_level.end(),
            0.0);
  std::fill(std_cell_population_per_level.begin(),
            std_cell_population_per_level.end(),
            0.0);
}

void
Octree::radius_search(const Eigen::Vector3d& query_point,
                      double radius,
                      unsigned int level,
                      RadiusSearchResult& result) const
{
  result.clear();

  constexpr std::size_t estimated_cell_count = 128;
  constexpr std::size_t estimated_candidate_count = 256;

  // Step 1: Retrieve all spatial keys of cells intersected by the sphere
  KeyContainer cell_keys;
  cell_keys.reserve(estimated_cell_count);
  get_cells_intersected_by_sphere(query_point, radius, level, cell_keys);

  // Step 2: Get candidate point indices from the intersected cells
  RadiusSearchResult candidate_points;
  candidate_points.reserve(estimated_candidate_count);
  get_points_indices_from_cells(cell_keys, level, candidate_points);

  // Precompute squared radius for efficiency
  double radius_square = radius * radius;

  // Precompute query point components to avoid repeated function calls.
  const double qx = query_point.x();
  const double qy = query_point.y();
  const double qz = query_point.z();

  // Step 3: Check each candidate point
  for (const auto& candidate : candidate_points) {
    // Direct element access from the cloud matrix.
    const double px = cloud(candidate, 0);
    const double py = cloud(candidate, 1);
    const double pz = cloud(candidate, 2);

    // Compute squared Euclidean distance.
    const double dx = px - qx;
    const double dy = py - qy;
    const double dz = pz - qz;
    const double dist_sq = dx * dx + dy * dy + dz * dz;

    if (dist_sq <= radius_square) {
      result.push_back(candidate);
    }
  }
}

Octree::KeyContainer
Octree::get_spatial_keys() const
{
  Octree::KeyContainer keys;
  keys.reserve(indexed_keys.size());
  for (const auto& entry : indexed_keys) {
    keys.push_back(entry.key);
  }
  return keys;
}

std::vector<IndexType>
Octree::get_point_indices() const
{
  std::vector<IndexType> indices;
  indices.reserve(indexed_keys.size());
  for (const auto& entry : indexed_keys) {
    indices.push_back(entry.index);
  }
  return indices;
}

std::optional<IndexType>
Octree::get_cell_index(Octree::SpatialKey truncated_key,
                       unsigned int level,
                       IndexType start_index) const
{
  assert(level <= max_depth);

  // Perform binary search for the first occurrence of the truncated key
  auto it = std::lower_bound(indexed_keys.begin() + start_index,
                             indexed_keys.end(),
                             truncated_key, // Use only the truncated query key
                             [level, this](const IndexAndKey& a, SpatialKey b) {
                               return (a.key >> bit_shift[level]) < b;
                             });

  // Ensure the found key actually matches the truncated query key
  if (it != indexed_keys.end() &&
      (it->key >> bit_shift[level]) == truncated_key) {
    return std::distance(indexed_keys.begin(), it);
  }

  return std::nullopt; // Cell not found
}

Eigen::Vector3d
Octree::get_cell_position(Octree::SpatialKey key, unsigned int level) const
{
  assert(level <= max_depth);

  Eigen::Vector3d position = min_point;

  fflush(stdout);

  for (int l = level; l > 0; --l) {
    SpatialKey x_bit = (key >> bit_shift[l]) & 1;
    SpatialKey y_bit = (key >> bit_shift[l]) & 1;
    SpatialKey z_bit = (key >> bit_shift[l]) & 1;

    Eigen::Vector3d step(x_bit, y_bit, z_bit);

    position += step * cell_size[l];
  }

  return position;
}

std::size_t
Octree::get_cells_intersected_by_sphere(const Eigen::Vector3d& query_point,
                                        double radius,
                                        unsigned int level,
                                        KeyContainer& result) const
{
  result.clear();

  // Number of cells per axis at this level is 2^level
  const unsigned int num_cells = 1u << level;
  const double cellSize = cell_size[level];

  // Compute the AABB of the sphere.
  Eigen::Vector3d sphere_min = query_point - Eigen::Vector3d::Constant(radius);
  Eigen::Vector3d sphere_max = query_point + Eigen::Vector3d::Constant(radius);

  // Helper: compute index range along a given axis
  auto compute_index_range =
    [this, cellSize, num_cells](double coord_min, double coord_max, int axis)
    -> std::pair<unsigned int, unsigned int> {
    const double inv_cell_size = 1.0 / cellSize;
    int i_min = static_cast<int>(
      std::floor((coord_min - min_point[axis]) * inv_cell_size));
    int i_max = static_cast<int>(
      std::floor((coord_max - min_point[axis]) * inv_cell_size));
    if (i_min < 0)
      i_min = 0;
    if (i_max >= static_cast<int>(num_cells))
      i_max = num_cells - 1;
    return { static_cast<unsigned int>(i_min),
             static_cast<unsigned int>(i_max) };
  };

  // Compute index ranges for x, y, and z
  auto [imin, imax] = compute_index_range(sphere_min.x(), sphere_max.x(), 0);
  auto [jmin, jmax] = compute_index_range(sphere_min.y(), sphere_max.y(), 1);
  auto [kmin, kmax] = compute_index_range(sphere_min.z(), sphere_max.z(), 2);

  // Precompute the squared radius
  const double radius_squared = radius * radius;

  // Lambda to check sphere-AABB intersection for a cell
  auto sphereIntersectsCell =
    [this, &query_point, cellSize, radius_squared](
      unsigned int i, unsigned int j, unsigned int k) -> bool {
    // Compute the cell's axis-aligned bounding box
    double cell_min_x = min_point.x() + i * cellSize;
    double cell_min_y = min_point.y() + j * cellSize;
    double cell_min_z = min_point.z() + k * cellSize;
    double cell_max_x = cell_min_x + cellSize;
    double cell_max_y = cell_min_y + cellSize;
    double cell_max_z = cell_min_z + cellSize;
    double distance_squared = 0.0;

    // For each axis, add squared distance if query_point is outside the cell
    double v0 = query_point[0];
    double v1 = query_point[1];
    double v2 = query_point[2];
    double dist_squared = 0.0;

    if (v0 < cell_min_x) {
      double diff = cell_min_x - v0;
      distance_squared += diff * diff;
    } else if (v0 > cell_max_x) {
      double diff = v0 - cell_max_x;
      distance_squared += diff * diff;
    }

    if (v1 < cell_min_y) {
      double diff = cell_min_y - v1;
      distance_squared += diff * diff;
    } else if (v1 > cell_max_y) {
      double diff = v1 - cell_max_y;
      distance_squared += diff * diff;
    }

    if (v2 < cell_min_z) {
      double diff = cell_min_z - v2;
      distance_squared += diff * diff;
    } else if (v2 > cell_max_z) {
      double diff = v2 - cell_max_z;
      distance_squared += diff * diff;
    }

    return distance_squared <= radius_squared;
  };

  // Iterate over all cells in the AABB
  for (unsigned int i = imin; i <= imax; ++i) {
    for (unsigned int j = jmin; j <= jmax; ++j) {
      for (unsigned int k = kmin; k <= kmax; ++k) {
        if (sphereIntersectsCell(i, j, k)) {
          Eigen::Vector3d cell_center =
            min_point + Eigen::Vector3d((i + 0.5) * cellSize,
                                        (j + 0.5) * cellSize,
                                        (k + 0.5) * cellSize);
          // Compute the full Morton key using the existing function
          SpatialKey full_key = compute_spatial_key(cell_center);
          // Truncate the key to the desired level
          SpatialKey cell_key = full_key >> bit_shift[level];
          result.push_back(cell_key);
        }
      }
    }
  }

  // Sort the keys before returning.
  std::sort(result.begin(), result.end());
  return result.size();
}

std::size_t
Octree::get_points_indices_from_cells(
  const std::vector<Octree::SpatialKey>& truncated_keys,
  unsigned int level,
  Octree::RadiusSearchResult& result) const
{
  assert(level <= max_depth);
  assert(std::is_sorted(truncated_keys.begin(), truncated_keys.end()));

  result.clear();

  IndexType current_start = 0;

  // Process each truncated key (i.e. each cell that intersects the search
  // sphere)
  for (const SpatialKey& key : truncated_keys) {
    // Find the first occurrence of the cell
    auto opt_first_index = get_cell_index(key, level, current_start);
    if (!opt_first_index)
      continue;

    IndexType first_index = *opt_first_index;

    // Use upper_bound to find the end of the range of points that belong to the
    // same cell
    auto lower = indexed_keys.begin() + first_index;
    auto upper = std::upper_bound(
      lower,
      indexed_keys.end(),
      key,
      [this, level](SpatialKey value, const IndexAndKey& entry) {
        return value < (entry.key >> bit_shift[level]);
      });

    // Insert all point indices from the block into the result vector
    // Since we only need the indices, we transform each IndexAndKey to its
    // index
    std::transform(lower,
                   upper,
                   std::back_inserter(result),
                   [](const IndexAndKey& ik) { return ik.index; });

    // Update current_start for the next search: start after the current cell's
    // block
    current_start = std::distance(indexed_keys.begin(), upper);
  }

  return result.size();
}

unsigned int
Octree::find_appropriate_level_for_radius_search(double radius) const
{
  constexpr double min_population_threshold = 1.5; // Threshold from CC

  double aim = radius / 2.5; // CC uses r/2.5

  unsigned int best_level = 1;
  double diff = cell_size[1] - aim;
  double min_diff_sq = diff * diff;

  for (unsigned int level = 2; level <= max_depth; ++level) {
    if (average_cell_population_per_level[level] < min_population_threshold)
      break;

    diff = cell_size[level] - aim;
    double diff_sq = diff * diff;

    if (diff_sq < min_diff_sq) {
      best_level = level;
      min_diff_sq = diff_sq;
    }
  }

  return best_level;
}

} // namespace py4dgeo
