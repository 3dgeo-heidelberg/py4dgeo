#include <py4dgeo/registration.hpp>

#include <numeric>
#include <queue>
#include <unordered_map>
#include <vector>

#define LMBD_MAX 1e20

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

  while (i != subsets_[i]) {
    subsets_[i] = subsets_[subsets_[i]];
    i = subsets_[i];
  }

  return i;
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
      // If balance_sizes is true, merge the larger subset into the smaller one
      if (numbers_[root_i] > numbers_[root_j]) {
        numbers_[root_j] += numbers_[root_i];
        subsets_[root_i] = root_j;
        return root_j;
      } else {
        numbers_[root_i] += numbers_[root_j];
        subsets_[root_j] = root_i;

        return root_i;
      }
    } else {
      // Always merge i's subset into j's subset
      numbers_[root_j] += numbers_[root_i];
      subsets_[root_i] = root_j;
      numbers_[root_i] = 0;
      return root_j;
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

double
squaredEuclideanDistance(const Eigen::RowVector3d& point1,
                         const Eigen::RowVector3d& point2)
{
  const auto diff = point1 - point2;
  return diff.squaredNorm();
}

double
point_2_point_VCCS_distance(EigenPointCloudConstRef point1,
                            EigenPointCloudConstRef point2,
                            EigenNormalSetRef normal1,
                            EigenNormalSetRef normal2,
                            double resolution)
{
  double x = point1.row(0)(0) - point2.row(0)(0);
  double y = point1.row(0)(1) - point2.row(0)(1);
  double z = point1.row(0)(2) - point2.row(0)(2);

  double n1 = std::sqrt(normal1.row(0)(0) * normal1.row(0)(0) +
                        normal1.row(0)(1) * normal1.row(0)(1) +
                        normal1.row(0)(2) * normal1.row(0)(2));
  double n2 = std::sqrt(normal2.row(0)(0) * normal2.row(0)(0) +
                        normal2.row(0)(1) * normal2.row(0)(1) +
                        normal2.row(0)(2) * normal2.row(0)(2));

  return 1.0 - std::fabs(n1 * n2) +
         std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2)) /
           resolution * 0.4;
}

std::vector<std::vector<int>>
supervoxel_segmentation(Epoch& epoch,
                        const KDTree& kdtree,
                        double resolution,
                        int k,
                        EigenNormalSet normals)
{

  // Define number of supervoxels and labels.
  auto n_supervoxels = estimate_supervoxel_count(epoch.cloud, resolution);
  std::vector<int> labels(epoch.cloud.rows(), -1);
  DisjointSet set(epoch.cloud.rows());

  // Calculate k neigbors and its distances for each point
  KDTree::NearestNeighborsDistanceResult result;
  kdtree.nearest_neighbors_with_distances(epoch.cloud, result, k);

  int supervoxels_amount = epoch.cloud.rows();

  // calculate lambda for segmentation
  DistanceVector lambda_distances;
  for (const auto& pair : result) {
    lambda_distances.push_back(pair.second[1]);
    // because [0] is the distance to the same point, so it's 1
  }

  double lambda = median_calculation(lambda_distances);

  // initialize temporary vars for supervoxel segmentation
  std::vector<int> temporary_supervoxels(epoch.cloud.rows());
  std::iota(temporary_supervoxels.begin(), temporary_supervoxels.end(), 0);
  std::vector<int> sizes(epoch.cloud.rows(), 1);
  std::queue<int> point_queue;
  std::vector<std::vector<long unsigned int>> neighborIndexes(result.size());

  // fill in temporal containment of neighbors for each points
  for (size_t i = 0; i < result.size(); ++i) {
    for (const long unsigned int& index : result[i].first) {
      neighborIndexes[i].push_back(index);
    }
  }

  DistanceVector distances;
  // searching for supervoxels and first segmentation
  for (; lambda < LMBD_MAX; lambda *= 2.0) {
    for (auto i : temporary_supervoxels) {
      if (neighborIndexes[i].empty())
        continue;

      labels[i] = i;
      point_queue.push(i);

      for (auto j : neighborIndexes[i]) {
        j = set.Find(j);
        if (labels[j] == -1) {
          point_queue.push(j);
        }
      }

      std::vector<long unsigned int> neighborIndexes_per_point;
      while (!point_queue.empty()) {
        int current = point_queue.front();
        point_queue.pop();

        double loss =
          sizes[current] *
          squaredEuclideanDistance(
            epoch.cloud.row(i),
            epoch.cloud.row(
              current)); // point_2_point_VCCS_distance(epoch.cloud.row(i),epoch.cloud.row(current),
                         // normals.row(i), normals.row(current), resolution);

        double improvement =
          lambda - loss; // metric for start union of supervoxels

        if (improvement > 0.0) {
          set.Union(current, i, false);

          sizes[i] += sizes[current];

          for (auto k : neighborIndexes[current]) {
            k = set.Find(k);
            if (labels[k] == -1) {
              labels[k] = k;
              point_queue.push(k);
            }
          }

          neighborIndexes[current].clear();

        } else {
          neighborIndexes_per_point.push_back(current);
        }
        if (--supervoxels_amount == n_supervoxels)
          break;
      }
      neighborIndexes[i].swap(neighborIndexes_per_point);

      // relabel elements for next iterations
      for (size_t j = 0; j < result[i].first.size(); ++j) {
        labels[result[i].first[j]] = -1;
      }

      if (supervoxels_amount == n_supervoxels)
        break;
    }

    // Update supervoxels
    supervoxels_amount = 0;
    for (auto i : temporary_supervoxels) {
      if (set.Find(i) == i) {
        temporary_supervoxels[supervoxels_amount++] = i;
      }
    }
    temporary_supervoxels.resize(supervoxels_amount);
    if (supervoxels_amount == n_supervoxels)
      break;
  }

  // temporal vars for the refinement of supervoxel boundaries
  std::queue<int> boundaries_queue;
  std::vector<bool> is_in_boundaries_queue(epoch.cloud.rows(), false);

  for (size_t i = 0; i < epoch.cloud.rows(); ++i) {
    labels[i] = set.Find(i);
    distances.push_back(
      squaredEuclideanDistance(epoch.cloud.row(i), epoch.cloud.row(labels[i])));
  }

  for (size_t i = 0; i < epoch.cloud.rows(); ++i) {
    for (auto j : result[i].first) {
      if (labels[i] != labels[j]) {
        if (!is_in_boundaries_queue[i]) {
          boundaries_queue.push(i);
          is_in_boundaries_queue[i] = true;
        }
        if (!is_in_boundaries_queue[j]) {
          boundaries_queue.push(j);
          is_in_boundaries_queue[j] = true;
        }
      }
    }
  }

  // refinement of supervoxel boundaries
  while (!boundaries_queue.empty()) {
    int current_point = boundaries_queue.front();
    boundaries_queue.pop();
    is_in_boundaries_queue[current_point] = false;

    bool change = false;
    for (auto j : result[current_point].first) {
      if (labels[current_point] == labels[j])
        continue;

      double point_distance = squaredEuclideanDistance(
        epoch.cloud.row(current_point), epoch.cloud.row(labels[labels[j]]));
      if (point_distance < distances[current_point]) {
        labels[current_point] = labels[j];
        distances[current_point] = point_distance;
        change = true;
      }
    }

    if (change) {
      for (auto j : result[current_point].first) {
        if (labels[current_point] != labels[j]) {
          if (!is_in_boundaries_queue[j]) {
            boundaries_queue.push(j);
            is_in_boundaries_queue[j] = true;
          }
        }
      }
    }
  }

  // Relabel the supervoxels points
  std::vector<int> map(epoch.cloud.rows());
  for (size_t i = 0; i < temporary_supervoxels.size(); ++i) {
    map[temporary_supervoxels[i]] = i;
  }
  for (size_t i = 0; i < epoch.cloud.rows(); ++i) {
    labels[i] = map[labels[i]];
  }

  // Create supervoxel point lists
  std::vector<std::vector<int>> supervoxels(n_supervoxels);
  for (auto point_index = 0; point_index < epoch.cloud.rows(); ++point_index) {
    supervoxels[labels[point_index]].push_back(point_index);
  }

  return supervoxels;
}

} // namespace py4dgeo
