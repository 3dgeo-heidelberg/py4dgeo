#include "py4dgeo/octree.hpp"
#include "py4dgeo/py4dgeo.hpp"

#include <Eigen/Core>

#include <istream>
#include <numeric>
#include <optional>
#include <ostream>
#include <utility>
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
    if (extent[i] <= 0) { // If the extent is zero or negative, set it to 1
      box_size[i] = 1.0;
    } else {
      box_size[i] = std::pow(2.0, std::ceil(std::log2(extent[i])));
    }
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
  PointContainer permutation(number_of_points);
  std::iota(permutation.begin(), permutation.end(), 0);

  std::sort(
    permutation.begin(), permutation.end(), [&](IndexType a, IndexType b) {
      return indexed_keys.keys[a] < indexed_keys.keys[b];
    });

  // Reorder the keys and indices based on the sorted permutation
  KeyContainer sorted_keys(number_of_points);
  PointContainer sorted_indices(number_of_points);
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

std::ostream&
Octree::saveIndex(std::ostream& stream) const
{
  // Write number of points as an indicator
  stream.write(reinterpret_cast<const char*>(&number_of_points),
               sizeof(number_of_points));

  // If no points, skip the rest
  if (number_of_points == 0) {
    return stream;
  }

  // Save keys
  IndexType size = static_cast<IndexType>(indexed_keys.indices.size());
  stream.write(reinterpret_cast<const char*>(&size), sizeof(IndexType));

  stream.write(reinterpret_cast<const char*>(indexed_keys.indices.data()),
               sizeof(IndexType) * size);
  stream.write(reinterpret_cast<const char*>(indexed_keys.keys.data()),
               sizeof(SpatialKey) * size);

  return stream;
}

std::istream&
Octree::loadIndex(std::istream& stream)
{
  // Read the number of points, serving as indicator
  stream.read(reinterpret_cast<char*>(&number_of_points),
              sizeof(number_of_points));

  // If no points, skip loading
  if (number_of_points == 0) {
    return stream;
  }

  // Load keys
  IndexType size;
  stream.read(reinterpret_cast<char*>(&size), sizeof(IndexType));

  indexed_keys.indices.resize(size);
  indexed_keys.keys.resize(size);

  stream.read(reinterpret_cast<char*>(indexed_keys.indices.data()),
              sizeof(IndexType) * size);
  stream.read(reinterpret_cast<char*>(indexed_keys.keys.data()),
              sizeof(SpatialKey) * size);

  compute_bounding_box(); // Recalculates box_size, cell_size, etc.
  compute_statistics();   // Recomputes level-wise stats
  assert(box_size.allFinite());
  assert(!cell_size[1].isZero());

  return stream;
}

std::size_t
Octree::radius_search(const Eigen::Vector3d& query_point,
                      double radius,
                      unsigned int level,
                      RadiusSearchResult& result) const
{
  result.clear();

  radius_search_backend(
    query_point,
    radius,
    level,
    [&](std::size_t size) { result.reserve(size); },
    [&](IndexType index, double squared_dist) { result.push_back(index); },
    [&](const RadiusSearchResult& all_inside) {
      result.insert(result.end(), all_inside.begin(), all_inside.end());
    });

  return result.size();
}

std::size_t
Octree::radius_search_with_distances(const Eigen::Vector3d& query_point,
                                     double radius,
                                     unsigned int level,
                                     RadiusSearchDistanceResult& result) const
{
  result.clear();

  radius_search_backend(
    query_point,
    radius,
    level,
    [&](std::size_t size) { result.reserve(size); },
    [&](IndexType index, double squared_dist) {
      result.emplace_back(index, squared_dist);
    },
    [&](const RadiusSearchResult& all_inside) {
      for (const auto& idx : all_inside) {
        const Eigen::Vector3d point = cloud.row(idx);
        double squared_dist = (point - query_point).squaredNorm();
        result.emplace_back(idx, squared_dist);
      }
    });

  // Sort the result by distance (ascending)
  std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
    return a.second < b.second;
  });

  return result.size();
}

std::optional<IndexType>
Octree::get_cell_index_start(Octree::SpatialKey truncated_key,
                             SpatialKey bitShift,
                             IndexType start_index,
                             IndexType end_index) const
{
  const auto keys_begin = indexed_keys.keys.begin();
  auto it = std::lower_bound(keys_begin + start_index,
                             keys_begin + end_index,
                             truncated_key,
                             [bitShift](SpatialKey element, SpatialKey value) {
                               return (element >> bitShift) < value;
                             });

  if (it != keys_begin + end_index && ((*it) >> bitShift) == truncated_key) {
    return std::distance(keys_begin, it);
  }

  return std::nullopt;
}

std::optional<IndexType>
Octree::get_cell_index_end(SpatialKey truncated_key,
                           SpatialKey bitShift,
                           IndexType start_index,
                           IndexType end_index) const
{
  const auto keys_begin = indexed_keys.keys.begin();

  auto it = std::upper_bound(keys_begin + start_index,
                             keys_begin + end_index,
                             truncated_key,
                             [bitShift](SpatialKey value, SpatialKey element) {
                               return value < (element >> bitShift);
                             });

  // Check if the element just before it (if any) matches the key
  if (it != keys_begin + start_index) {
    auto prev = it - 1;
    if (((*prev) >> bitShift) == truncated_key) {
      return std::distance(keys_begin, it);
    }
  }

  return std::nullopt;
}

std::optional<IndexType>
Octree::get_cell_index_end_exponential(SpatialKey truncated_key,
                                       unsigned int level,
                                       IndexType first_index,
                                       IndexType global_end_index) const
{
  // 1. Estimate the last occurrence of the truncated key using the average cell
  // population at this level
  IndexType avg_cell_pop = average_cell_population_per_level[level];
  IndexType bound = avg_cell_pop;
  SpatialKey bitShift = bit_shift[level];

  // 2. Perform exponential search to find an upper bound, where the
  // truncated_key is no longer present
  while (
    (first_index + bound < global_end_index) &&
    (indexed_keys.keys[first_index + bound] >> bitShift == truncated_key)) {
    bound += avg_cell_pop;
  }

  // 3. Delegate to binary search (upper_bound) within the discovered narrow
  // range Cap it to avoid overshooting the array
  return get_cell_index_end(truncated_key,
                            bitShift,
                            first_index,
                            std::min(first_index + bound, global_end_index));
}

std::size_t
Octree::get_cells_intersected_by_sphere(const Eigen::Vector3d& query_point,
                                        double radius,
                                        unsigned int level,
                                        KeyContainer& inside,
                                        KeyContainer& intersecting) const
{
  // Number of cells per axis at this level is 2^level
  const unsigned int num_cells = 1u << level;
  const Eigen::Vector3d cellSize = cell_size[level];

  // Compute the AABB of the sphere
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

  // Early out: if sphere AABB does not intersect octree AABB
  if ((sphere_max.array() < min_point.array()).any() ||
      (sphere_min.array() > max_point.array()).any()) {
    return 0;
  }

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
    // Compute the axis-aligned bounding box of the cell
    Eigen::Vector3d cell_min =
      min_point + cellSize.cwiseProduct(Eigen::Vector3d(i, j, k));
    Eigen::Vector3d cell_max = cell_min + cellSize;

    // First: Cheap AABB–sphere intersection test
    double distance_squared = 0.0;
    const double qx = query_point.x();
    const double qy = query_point.y();
    const double qz = query_point.z();

    if (qx < cell_min.x()) {
      double diff = cell_min.x() - qx;
      distance_squared += diff * diff;
    } else if (qx > cell_max.x()) {
      double diff = qx - cell_max.x();
      distance_squared += diff * diff;
    }

    if (qy < cell_min.y()) {
      double diff = cell_min.y() - qy;
      distance_squared += diff * diff;
    } else if (qy > cell_max.y()) {
      double diff = qy - cell_max.y();
      distance_squared += diff * diff;
    }

    if (qz < cell_min.z()) {
      double diff = cell_min.z() - qz;
      distance_squared += diff * diff;
    } else if (qz > cell_max.z()) {
      double diff = qz - cell_max.z();
      distance_squared += diff * diff;
    }

    if (distance_squared > radius_squared)
      return cell_relation_to_sphere::Outside;

    // Next: check if all 8 corners are inside the sphere
    bool any_outside = false;

    for (size_t cx = 0; cx <= 1; ++cx) {
      const double x = cx ? cell_max.x() : cell_min.x();
      for (size_t cy = 0; cy <= 1; ++cy) {
        const double y = cy ? cell_max.y() : cell_min.y();
        for (size_t cz = 0; cz <= 1; ++cz) {
          const double z = cz ? cell_max.z() : cell_min.z();

          const double dx = x - qx;
          const double dy = y - qy;
          const double dz = z - qz;
          const double dist_sq = dx * dx + dy * dy + dz * dz;

          if (dist_sq > radius_squared) {
            return cell_relation_to_sphere::Intersecting;
          }
        }
      }
    }

    return cell_relation_to_sphere::Inside;
  };

  // Iterate over all cells in the AABB
  Eigen::Vector3d cell_center;
  for (size_t i = imin; i <= imax; ++i) {
    for (size_t j = jmin; j <= jmax; ++j) {
      for (size_t k = kmin; k <= kmax; ++k) {
        auto relation = classify_cell_relation_to_sphere(i, j, k);
        if (relation == cell_relation_to_sphere::Outside)
          continue;

        cell_center = min_point + cellSize.cwiseProduct(
                                    Eigen::Vector3d(i + 0.5, j + 0.5, k + 0.5));
        SpatialKey full_key = compute_spatial_key(cell_center);

        if (relation == cell_relation_to_sphere::Inside) {
          inside.push_back(full_key >> bit_shift[level]);
        } else {
          intersecting.push_back(full_key >> bit_shift[level]);
        }
      }
    }
  }

  // Sort the keys before returning (required)
  std::sort(inside.begin(), inside.end());
  std::sort(intersecting.begin(), intersecting.end());

  return intersecting.size() + inside.size();
}

std::size_t
Octree::get_points_indices_from_cells(
  const Octree::KeyContainer& truncated_keys,
  unsigned int level,
  RadiusSearchResult& result) const
{
  assert(level <= max_depth);
  assert(std::is_sorted(truncated_keys.begin(), truncated_keys.end()));

  const SpatialKey bitShift = bit_shift[level];
  const auto indices_begin = indexed_keys.indices.begin();

  IndexType current_start_index = 0;
  IndexType search_limit_index = indexed_keys.keys.size();

  // Process each truncated key and store corresponding point indices
  for (const SpatialKey& key : truncated_keys) {
    // Find the first occurrence of the cell
    auto opt_first_index = get_cell_index_start(
      key, bitShift, current_start_index, search_limit_index);
    if (!opt_first_index)
      continue;
    IndexType first_index = *opt_first_index;

    // Find the last occurrence of the cell
    auto opt_last_index = get_cell_index_end_exponential(
      key,
      level,
      first_index,
      std::min(first_index + max_cell_population_per_level[level],
               search_limit_index));
    if (!opt_last_index)
      continue;
    IndexType last_index = *opt_last_index;

    result.insert(
      result.end(), indices_begin + first_index, indices_begin + last_index);

    // Update current_start: start after the current cell's block
    current_start_index = last_index;
  }

  return result.size();
}

unsigned int
Octree::find_appropriate_level_for_radius_search(double radius) const
{
  constexpr double min_population_threshold = 1.5; // Threshold from CC

  Eigen::Vector3d aim =
    Eigen::Vector3d::Constant(radius / 1.5); // CC uses r/2.5

  unsigned int best_level = 1;

  double min_error = (cell_size[1] - aim).cwiseAbs().maxCoeff();
  for (size_t level = 2; level <= max_depth; ++level) {
    if (average_cell_population_per_level[level] < min_population_threshold)
      break;

    double error = (cell_size[level] - aim).cwiseAbs().maxCoeff();

    if (error < min_error) {
      best_level = level;
      min_error = error;
    }
  }

  return best_level;
}

} // namespace py4dgeo
