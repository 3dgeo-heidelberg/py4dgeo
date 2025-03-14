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
  min_point = cloud.colwise()
                .minCoeff(); // Find the smallest value in each column (x,y,z)
  max_point =
    cloud.colwise().maxCoeff(); // Find the biggest value in each column (x,y,z)
  Eigen::Vector3d center = (min_point + max_point) * 0.5;
  double max_extent = (max_point - min_point)
                        .maxCoeff(); // (max_point - min_point) gives the width,
                                     // height and depth of the bounding box
  cube_size = std::pow(
    2,
    std::ceil(std::log2(
      max_extent))); // Gives the exponent x of 2^x needed, as ceiled integer
  min_point = center - Eigen::Vector3d::Constant(
                         cube_size * 0.5); // Gives the corner of the min point
  max_point = center + Eigen::Vector3d::Constant(
                         cube_size * 0.5); // Gives the corner of the max point

  // Compute cell sizes
  cell_size[0] = max_point.x() - min_point.x();

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

void
Octree::apply_leaf_parameter(unsigned int leaf)
{
  if (leaf <= 1)
    return; // No merging needed for leaf <= 1

  std::vector<IndexAndKey> temp_group; // Temporary storage for merging
  std::vector<IndexAndKey> new_keys;   // Store final merged spatial keys

  for (int depth = max_depth - 1; depth >= 0; --depth) {
    new_keys.clear();
    temp_group.clear();

    // Walk through all octants at a given level (amount: 2^(3*level))
    unsigned int global_index = 0;
    for (SpatialKey parent_key = 0; parent_key < (1 << depth); ++parent_key) {
      unsigned int point_count = 0;

      // Walk through all indices, starting from the current cell (TODO)
      for (const IndexAndKey& entry : indexed_keys) {
        SpatialKey current_key = entry.key >> (3 * (max_depth - depth));
        if (current_key != parent_key) {
          // truncate all point_count indices by (3 * (max_depth - depth)
          global_index = point_count;
          break;
        }
        ++point_count;

        if (point_count > leaf) {
          break;
        }
      }
    }
  }
}

Octree
Octree::create(const EigenPointCloudRef& cloud)
{
  return Octree(cloud);
}

void
Octree::build_tree(unsigned int leaf)
{
  compute_bounding_cube();

  number_of_points = cloud.rows();
  indexed_keys.resize(number_of_points);

  // Step 1: Compute Z-order values and store point indices
  for (IndexType i = 0; i < number_of_points; ++i) {
    indexed_keys[i] = { compute_spatial_key(cloud.row(i)), i };
  }

  // Step 2: Sort the indexed keys by Z-value code
  std::sort(indexed_keys.begin(), indexed_keys.end());
}

void
Octree::invalidate()
{
  // TODO
  leaf_parameter = 0;
}

std::ostream&
Octree::saveIndex(std::ostream& stream) const
{
  stream.write(reinterpret_cast<const char*>(&leaf_parameter), sizeof(int));

  // TODO
  if (leaf_parameter != 0) {
  }

  return stream;
}

std::istream&
Octree::loadIndex(std::istream& stream)
{
  // Read the leaf parameter
  stream.read(reinterpret_cast<char*>(&leaf_parameter), sizeof(int));

  // TODO
  if (leaf_parameter != 0) {
  }

  return stream;
}

std::size_t
Octree::radius_search(const double* query,
                      double radius,
                      RadiusSearchResult& result) const
{
  // TODO
}

std::size_t
Octree::radius_search_with_distances(const double* query,
                                     double radius,
                                     RadiusSearchDistanceResult& result) const
{
  // TODO
}

void
Octree::nearest_neighbors_with_distances(EigenPointCloudConstRef cloud,
                                         NearestNeighborsDistanceResult& result,
                                         int k) const
{
  result.resize(cloud.rows());

#ifdef PY4DGEO_WITH_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (IndexType i = 0; i < cloud.rows(); ++i) {
    std::pair<std::vector<IndexType>, std::vector<double>> pointResult;

    std::vector<IndexType>& ret_indices = pointResult.first;
    std::vector<double>& out_dists_sqr = pointResult.second;
    ret_indices.resize(k);
    out_dists_sqr.resize(k);

    // TODO
  }
}

void
Octree::nearest_neighbors(EigenPointCloudConstRef cloud,
                          NearestNeighborsResult& result,
                          int k) const
{
  result.resize(cloud.rows());

#ifdef PY4DGEO_WITH_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (IndexType i = 0; i < cloud.rows(); ++i) {
    std::vector<IndexType> pointResult;
    std::vector<double> dis_skip;

    std::vector<IndexType>& ret_indices = pointResult;
    std::vector<double>& out_dists_sqr = dis_skip;
    ret_indices.resize(k);
    out_dists_sqr.resize(k);

    // TODO
  }
}

} // namespace py4dgeo
