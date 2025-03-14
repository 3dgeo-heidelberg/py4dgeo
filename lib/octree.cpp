#ifdef PY4DGEO_WITH_OPENMP
#include <omp.h>
#endif

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

  for (int i = 1; i <= max_depth; i++) {
    cell_size[i] = cell_size[i - 1] * 0.5;
  }
}

Octree::SpatialKey
Octree::compute_spatial_key(const Eigen::Vector3d& point) const
{
  Eigen::Vector3d normalized = (point - min_point) / cube_size;
  unsigned int grid_size = (1 << max_depth); // 2^max_depth
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
}

void
Octree::invalidate()
{
  number_of_points = 0;
  cube_size = 0.0;
  min_point.setZero();
  max_point.setZero();

  indexed_keys.clear();
  indexed_keys.shrink_to_fit();
}

void
Octree::radius_search(const double* query,
                      double radius,
                      RadiusSearchResult& result) const
{
  // TODO
}

double const
Octree::get_cube_size() const
{
  return cube_size;
}

Eigen::Vector3d const
Octree::get_min_point() const
{
  return min_point;
}

Eigen::Vector3d const
Octree::get_max_point() const
{
  return max_point;
}

inline const unsigned int
Octree::get_number_of_points() const
{
  return number_of_points;
}

inline const double
Octree::get_cell_size(unsigned int level) const
{
  return cell_size[level];
}

std::vector<Octree::SpatialKey>
Octree::get_spatial_keys() const
{
  std::vector<SpatialKey> keys;
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

} // namespace py4dgeo
