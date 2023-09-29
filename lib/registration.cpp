#include <py4dgeo/registration.hpp>

#include <numeric>
#include <queue>
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
DisjointSet::Find(IndexType i) const
{
  assert(i >= 0 && i < size_);

  // Path compression: Make all nodes in the path point to the root
  IndexType root = i;
  while (root != subsets_[root]) {
    root = subsets_[root];
  }

  // Path compression: Update the parent of all nodes in the path
  while (i != root) {
    IndexType next = subsets_[i];
    subsets_[i] = root;
    i = next;
  }

  return root;
}

IndexType
DisjointSet::Union(IndexType i, IndexType j, bool balance_sizes)
{
  // Ensure i and j are valid indices
  assert(i >= 0 && i < size_);
  assert(j >= 0 && j < size_);

  // Find the root representatives of the subsets containing i and j
  IndexType root_i = Find(i);
  IndexType root_j = Find(j);

  if (root_i != root_j) {
    if (balance_sizes) {
      // If balance_sizes is true, merge the smaller subset into the larger one
      if (numbers_[root_i] >= numbers_[root_j]) {
        numbers_[root_i] += numbers_[root_j] + 1;
        subsets_[root_j] = root_i;
        return root_i;
      } else {
        numbers_[root_j] += numbers_[root_i] + 1;
        subsets_[root_i] = root_j;
        return root_j;
      }
    } else {
      // Always merge j's subset into i's subset
      subsets_[root_j] = root_i;
      return root_i;
    }
  }

  // If the subsets already have the same root representative, no action is
  // needed
  return root_i;
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

std::vector<std::vector<int>>
supervoxel_segmentation(EigenPointCloudConstRef cloud,
                        const KDTree& kdtree,
                        double resolution)
{

  auto n_supervoxels = estimate_supervoxel_count(cloud, resolution);
  std::vector<int> labels(cloud.rows(), -1);
  DisjointSet set(cloud.rows());

  // Lambda function to expand a supervoxel using a basic region-growing
  // approach
  auto expand_supervoxel = [&](int seed_point, int supervoxel_id) {
    std::queue<int> queue;
    queue.push(seed_point);

    while (!queue.empty()) {
      int current_point = queue.front();
      queue.pop();

      if (labels[current_point] == -1) {
        labels[current_point] = supervoxel_id;

        std::pair<std::vector<IndexType>, std::vector<double>> result;
        kdtree.nearest_neighbors_with_distances(cloud, result);

        // Add neighboring points to the queue if they meet the criteria
        for (size_t i = 0; i < result.first.size(); ++i) {
          int neighbor = result.first[i];
          double distance = result.second[i];

          if (labels[neighbor] == -1 && distance <= resolution) {
            queue.push(neighbor);
            set.Union(current_point, neighbor, true);
          }
        }
      }
    }
  };

  int current_supervoxel_id = 0;

  for (int point_index = 0; point_index < cloud.rows(); ++point_index) {
    if (labels[point_index] == -1) {
      expand_supervoxel(point_index, current_supervoxel_id);
      current_supervoxel_id++;
    }
  }

  // Organize supervoxels into separate vectors
  std::vector<std::vector<int>> supervoxels(n_supervoxels);
  for (auto point_index = 0; point_index < cloud.rows(); ++point_index) {
    auto supervoxel_id = set.Find(
      labels[point_index]); // Find the representative label in the DisjointSet
    supervoxels[supervoxel_id].push_back(point_index);
  }

  return supervoxels;
}

} // namespace py4dgeo
