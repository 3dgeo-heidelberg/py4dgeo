#ifdef PY4DGEO_WITH_OPENMP
#include <omp.h>
#endif

#include "py4dgeo/octree.hpp"
#include "py4dgeo/py4dgeo.hpp"

#include <numeric>
#include <vector>

namespace py4dgeo {

Octree::Octree(const EigenPointCloudRef& cloud)
  : cloud{ cloud }
{
}

void
Octree::compute_bounding_box()
{
  // Find the smallest value in each column (x,y,z)
  min_point = cloud.colwise().minCoeff();

  // Find the biggest value in each column (x,y,z)
  max_point = cloud.colwise().maxCoeff();
  Eigen::Vector3d center = (min_point + max_point) * 0.5;

  // (max_point - min_point) gives width, height and depth of the bounding box
  Eigen::Vector3d extent = max_point - min_point;

  // For each axis, round extent to next power of two
  for (int i = 0; i < 3; ++i) {
    box_size[i] = std::pow(2.0, std::ceil(std::log2(extent[i])));
  }

  inv_box_size = box_size.cwiseInverse();

  // Gives the corner of the min point
  min_point = center - box_size * 0.5;

  // Gives the corner of the max point
  max_point = center + box_size * 0.5;

  // Compute cell sizes
  cell_size[0] = box_size;

  for (size_t i = 1; i <= max_depth; ++i) {
    cell_size[i] = cell_size[i - 1] * 0.5;
  }
}

void
Octree::compute_statistics()
{
  occupied_cells_per_level[0] = 1;
  max_cell_population_per_level[0] = number_of_points;
  average_cell_population_per_level[0] = static_cast<double>(number_of_points);
  std_cell_population_per_level[0] = 0.0;

  for (size_t level = 1; level <= max_depth; ++level) {

    unsigned int unique_cells = 0;
    unsigned int max_population = 0;
    double sum = 0.0;
    double sum2 = 0.0;
    unsigned int current_cell_population = 0;

    // Initialize with the first element's truncated key
    SpatialKey previous_key = indexed_keys.keys[0] >> bit_shift[level];
    current_cell_population = 1; // Count the first point

    for (IndexType i = 1; i < indexed_keys.keys.size(); ++i) {
      SpatialKey current_key = indexed_keys.keys[i] >> bit_shift[level];
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
      static_cast<double>(indexed_keys.keys.size()) / unique_cells;
    std_cell_population_per_level[level] = std::sqrt(
      sum2 / unique_cells - average_cell_population_per_level[level] *
                              average_cell_population_per_level[level]);
  }
}

Octree
Octree::create(const EigenPointCloudRef& cloud)
{
  return Octree(cloud);
}

void
Octree::build_tree()
{
  compute_bounding_box();

  number_of_points = cloud.rows();
  indexed_keys.keys.resize(number_of_points);
  indexed_keys.indices.resize(number_of_points);

  // Step 1: Compute Z-order values and store point indices
  for (IndexType i = 0; i < number_of_points; ++i) {
    // Compute the spatial key for the point
    indexed_keys.keys[i] = compute_spatial_key(cloud.row(i));
    // Save the original index
    indexed_keys.indices[i] = i;
  }

  // Step 2: Sort the indexed keys by Z-order value
  std::vector<IndexType> permutation(number_of_points);
  std::iota(permutation.begin(), permutation.end(), 0);

  std::sort(
    permutation.begin(), permutation.end(), [&](IndexType a, IndexType b) {
      return indexed_keys.keys[a] < indexed_keys.keys[b];
    });

  // Reorder the keys and indices based on the sorted permutation
  std::vector<SpatialKey> sorted_keys(number_of_points);
  std::vector<IndexType> sorted_indices(number_of_points);
  for (IndexType i = 0; i < number_of_points; ++i) {
    sorted_keys[i] = indexed_keys.keys[permutation[i]];
    sorted_indices[i] = indexed_keys.indices[permutation[i]];
  }

  // Swap the sorted vectors back into the member structure
  indexed_keys.keys.swap(sorted_keys);
  indexed_keys.indices.swap(sorted_indices);

  // Step 3: Compute related properties (max cell population, avg, ...)
  compute_statistics();
}

void
Octree::invalidate()
{
  number_of_points = 0;
  box_size[0] = box_size[1] = box_size[2] = 0.0;
  inv_box_size[0] = inv_box_size[1] = inv_box_size[2] = 0.0;
  min_point = max_point = Eigen::Vector3d::Zero();

  indexed_keys.keys.clear();
  indexed_keys.indices.clear();
  indexed_keys.keys.shrink_to_fit();
  indexed_keys.indices.shrink_to_fit();

  std::fill(cell_size.begin(), cell_size.end(), Eigen::Vector3d::Zero());
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
  constexpr std::size_t estimated_candidate_count = 512;

  // Step 1: Retrieve all spatial keys of cells intersected by the sphere
  KeyContainer cells_inside, cells_intersecting;
  cells_inside.reserve(estimated_cell_count);
  cells_intersecting.reserve(estimated_cell_count);
  get_cells_intersected_by_sphere(
    query_point, radius, level, cells_inside, cells_intersecting);

  // Step 2: Get candidate point indices from the intersected cells
  RadiusSearchResult points_inside_sphere, candidate_points;
  points_inside_sphere.reserve(estimated_candidate_count);
  candidate_points.reserve(estimated_candidate_count);
  get_points_indices_from_cells(cells_inside, level, points_inside_sphere);
  get_points_indices_from_cells(cells_intersecting, level, candidate_points);

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

  result.insert(
    result.end(), points_inside_sphere.begin(), points_inside_sphere.end());
}

Octree::KeyContainer
Octree::get_spatial_keys() const
{
  Octree::KeyContainer keys;
  keys.reserve(indexed_keys.keys.size());
  for (const auto& key : indexed_keys.keys) {
    keys.push_back(key);
  }
  return keys;
}

Octree::PointContainer
Octree::get_point_indices() const
{
  std::vector<IndexType> indices;
  indices.reserve(indexed_keys.indices.size());
  for (const auto& index : indexed_keys.indices) {
    indices.push_back(index);
  }
  return indices;
}

std::optional<IndexType>
Octree::get_cell_index_start(Octree::SpatialKey truncated_key,
                             SpatialKey bitShift,
                             IndexType start_index) const
{
  // Perform binary search for the first occurrence of the truncated key
  auto it =
    std::lower_bound(indexed_keys.keys.begin() + start_index,
                     indexed_keys.keys.end(),
                     truncated_key, // search value (already truncated)
                     [bitShift, this](SpatialKey element, SpatialKey value) {
                       return (element >> bitShift) < value;
                     });

  // Ensure the found key actually matches the truncated query key
  if (it != indexed_keys.keys.end() && ((*it) >> bitShift) == truncated_key) {
    return std::distance(indexed_keys.keys.begin(), it);
  }

  return std::nullopt; // Cell not found
}

std::optional<IndexType>
Octree::get_cell_index_end(SpatialKey truncated_key,
                           SpatialKey bitShift,
                           IndexType start_index) const
{
  // Use upper_bound to find the first index after the block of keys that match
  auto it =
    std::upper_bound(indexed_keys.keys.begin() + start_index,
                     indexed_keys.keys.end(),
                     truncated_key, // search value (already truncated)
                     [bitShift, this](SpatialKey value, SpatialKey element) {
                       // Compare the search value with the truncated value of
                       // the container element
                       return value < (element >> bitShift);
                     });

  // If we've found an element (or reached the end), return its index.
  if (it != indexed_keys.keys.begin()) {
    return std::distance(indexed_keys.keys.begin(), it);
  }

  return std::nullopt; // Cell not found
}

std::size_t
Octree::get_cells_intersected_by_sphere(const Eigen::Vector3d& query_point,
                                        double radius,
                                        unsigned int level,
                                        KeyContainer& inside,
                                        KeyContainer& intersecting) const
{
  inside.clear();
  intersecting.clear();

  // Number of cells per axis at this level is 2^level
  const unsigned int num_cells = 1u << level;
  const Eigen::Vector3d cellSize = cell_size[level];

  // Compute the AABB of the sphere.
  Eigen::Vector3d sphere_min = query_point - Eigen::Vector3d::Constant(radius);
  Eigen::Vector3d sphere_max = query_point + Eigen::Vector3d::Constant(radius);

  // Helper: compute index range along a given axis
  const Eigen::Vector3d inv_cell_size = cellSize.cwiseInverse();
  auto compute_index_range = [this, inv_cell_size, num_cells](
                               double coord_min, double coord_max, int axis)
    -> std::pair<unsigned int, unsigned int> {
    int i_min = static_cast<int>(
      std::floor((coord_min - min_point[axis]) * inv_cell_size[axis]));
    int i_max = static_cast<int>(
      std::floor((coord_max - min_point[axis]) * inv_cell_size[axis]));
    i_min = std::max(i_min, 0);
    i_max = std::min(i_max, static_cast<int>(num_cells - 1));
    return { static_cast<unsigned int>(i_min),
             static_cast<unsigned int>(i_max) };
  };

  // Compute index ranges of the AABB for x, y, and z
  auto [imin, imax] = compute_index_range(sphere_min.x(), sphere_max.x(), 0);
  auto [jmin, jmax] = compute_index_range(sphere_min.y(), sphere_max.y(), 1);
  auto [kmin, kmax] = compute_index_range(sphere_min.z(), sphere_max.z(), 2);

  // Precompute the squared radius
  const double radius_squared = radius * radius;

  // Lambda to check sphere-AABB intersection for a cell
  // Classify whether the cell is fully inside or just intersecting the sphere
  auto classify_cell_relation_to_sphere =
    [this, &query_point, cellSize, radius_squared](
      unsigned int i,
      unsigned int j,
      unsigned int k) -> cell_relation_to_sphere {
    // Compute the cell's axis-aligned bounding box
    double cell_min_x = min_point.x() + i * cellSize.x();
    double cell_min_y = min_point.y() + j * cellSize.y();
    double cell_min_z = min_point.z() + k * cellSize.z();
    double cell_max_x = cell_min_x + cellSize.x();
    double cell_max_y = cell_min_y + cellSize.y();
    double cell_max_z = cell_min_z + cellSize.z();

    // Check whether all 8 corners of the cell are inside the sphere
    bool any_inside = false;
    bool any_outside = false;

    for (size_t dx = 0; dx <= 1; ++dx) {
      for (size_t dy = 0; dy <= 1; ++dy) {
        for (size_t dz = 0; dz <= 1; ++dz) {
          double cx = dx ? cell_max_x : cell_min_x;
          double cy = dy ? cell_max_y : cell_min_y;
          double cz = dz ? cell_max_z : cell_min_z;

          Eigen::Vector3d corner(cx, cy, cz);
          double dist_sq = (query_point - corner).squaredNorm();

          if (dist_sq <= radius_squared)
            any_inside = true;
          else
            any_outside = true;
          if (any_inside && any_outside)
            return cell_relation_to_sphere::Intersecting;
        }
      }
    }
    if (any_inside)
      return cell_relation_to_sphere::Inside;

    // Check if sphere center is inside the cell
    if (query_point[0] >= cell_min_x && query_point[0] <= cell_max_x &&
        query_point[1] >= cell_min_y && query_point[1] <= cell_max_y &&
        query_point[2] >= cell_min_z && query_point[2] <= cell_max_z)
      return cell_relation_to_sphere::Intersecting;

    // Final fallback: conservative AABB-sphere check
    // For each axis, add squared distance if query_point is outside the cell
    double v0 = query_point[0];
    double v1 = query_point[1];
    double v2 = query_point[2];
    double distance_squared = 0.0;

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

    if (distance_squared <= radius_squared)
      return cell_relation_to_sphere::Intersecting;

    // Truly outside
    return cell_relation_to_sphere::Outside;
  };

  // Iterate over all cells in the AABB
  for (unsigned int i = imin; i <= imax; ++i) {
    for (unsigned int j = jmin; j <= jmax; ++j) {
      for (unsigned int k = kmin; k <= kmax; ++k) {
        auto relation = classify_cell_relation_to_sphere(i, j, k);
        if (relation == cell_relation_to_sphere::Outside)
          continue;
        Eigen::Vector3d cell_center =
          min_point + Eigen::Vector3d((i + 0.5) * cellSize.x(),
                                      (j + 0.5) * cellSize.y(),
                                      (k + 0.5) * cellSize.z());
        SpatialKey full_key = compute_spatial_key(cell_center);
        SpatialKey cell_key = full_key >> bit_shift[level];

        if (relation == cell_relation_to_sphere::Inside) {
          inside.push_back(cell_key);
        } else {
          intersecting.push_back(cell_key);
        }
      }
    }
  }

  // Sort the keys before returning.
  std::sort(inside.begin(), inside.end());
  std::sort(intersecting.begin(), intersecting.end());

  return intersecting.size() + inside.size();
}

std::size_t
Octree::get_points_indices_from_cells(
  const Octree::KeyContainer& truncated_keys,
  unsigned int level,
  Octree::RadiusSearchResult& result) const
{
  assert(level <= max_depth);
  assert(std::is_sorted(truncated_keys.begin(), truncated_keys.end()));
  result.clear();

  IndexType current_start = 0;
  SpatialKey bitShift = bit_shift[level];

  // Process each truncated key (i.e. each cell that intersects the search
  // sphere)
  for (const SpatialKey& key : truncated_keys) {
    // Find the first occurrence of the cell
    auto opt_first_index = get_cell_index_start(key, bitShift, current_start);
    if (!opt_first_index)
      continue;
    IndexType first_index = *opt_first_index;

    // Find the last occurrence of the cell
    auto opt_last_index = get_cell_index_end(key, bitShift, first_index);
    if (!opt_last_index)
      continue;
    IndexType last_index = *opt_last_index;

    // Copy the corresponding indices into the result.
    std::copy(indexed_keys.indices.begin() + first_index,
              indexed_keys.indices.begin() + last_index,
              std::back_inserter(result));

    // Update current_start for the next search: start after the current cell's
    // block
    current_start = last_index;
  }

  return result.size();
}

unsigned int
Octree::find_appropriate_level_for_radius_search(double radius) const
{
  constexpr double min_population_threshold = 1.5; // Threshold from CC

  double aim = radius / 2.5; // CC uses r/2.5

  unsigned int best_level = 1;
  double min_diff_sq =
    (cell_size[1] - Eigen::Vector3d::Constant(aim)).squaredNorm();

  for (size_t level = 2; level <= max_depth; ++level) {
    if (average_cell_population_per_level[level] < min_population_threshold)
      break;

    Eigen::Vector3d diff = cell_size[level] - Eigen::Vector3d::Constant(aim);
    double diff_sq = diff.squaredNorm();

    if (diff_sq < min_diff_sq) {
      best_level = level;
      min_diff_sq = diff_sq;
    }
  }

  return best_level;
}

} // namespace py4dgeo
