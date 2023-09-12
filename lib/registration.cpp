#include <py4dgeo/registration.hpp>

#include <unordered_map>
#include <vector>

namespace py4dgeo {

void
transform_pointcloud_inplace(EigenPointCloudRef cloud,
                             const Transformation& trafo,
                             EigenPointCloudConstRef reduction_point)
{
  cloud.transpose() =
    (trafo.linear() * (cloud.rowwise() - reduction_point.row(0)).transpose())
      .colwise() +
    (trafo.translation() + reduction_point.row(0).transpose());
}

DisjointSet::DisjointSet(IndexType size)
  : size_(size)
  , numbers_(size, 1)
  , subsets_(size)
{
  std::iota(subsets_.begin(), subsets_.end(), 0);
}

IndexType
DisjointSet::find(IndexType i) const
{
  // TODO
}

IndexType DisjointSet::union(IndexType i, IndexType j, bool balance_sizes)
{
  // TODO
}

std::size_t
estimate_supervoxel_count(EigenPointCloudConstRef cloud, double seed_resolution)
{
  // Calculate the point cloud point bounding box
  using Coordinate = Eigen::Array<double, 1, 3>;
  Coordinate lowerleft = cloud.colwise().minCoeff();
  Coordinate upperright = cloud.colwise().maxCoeff();

  // Subdivide the bounding box according to the given seed_resolution
  using std::floor;
  using MultiIndex = Eigen::Array<std::size_t, 1, 3>;
  MultiIndex voxelsizes =
    floor(((upperright - lowerleft) / seed_resolution) + 1)
      .cast<std::size_t>()
      .eval();

  // Assign all points to their voxels by throwing them in a hashmap
  std::unordered_map<std::size_t, std::vector<Eigen::Index>> voxelmap;
  for (IndexType i = 0; i < cloud.rows(); ++i) {
    Coordinate coord(cloud.row(i));
    auto ind = floor(((coord - lowerleft) / seed_resolution) + 1)
                 .cast<std::size_t>()
                 .eval();
    voxelmap[ind[2] * voxelsizes[1] * voxelsizes[0] + ind[1] * voxelsizes[0] +
             ind[0]]
      .push_back(i);
  }

  return voxelmap.size();
}

void
supervoxel_segmentation(EigenPointCloudConstRef cloud,
                        const KDTree& kdtree,
                        double resolution)
{
  // Determine the supervoxel target count from the given resolution
  auto n_supervoxels = estimate_supervoxel_count(cloud, resolution);

  // The resulting data structures
  std::vector<int> labels(cloud.rows(), -1);

  // The internal data structures for the region growing
  DisjointSet set(cloud.rows());

  // TODO
}

} // namespace py4dgeo
