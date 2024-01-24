#include <py4dgeo/registration.hpp>

#include <numeric>
#include <queue>
#include <unordered_map>
#include <vector>

#include <iostream>//DELETE
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
point_2_point_VCCS_distance(const Eigen::RowVector3d& point1,
                            const Eigen::RowVector3d& point2,
                            const Eigen::RowVector3d& normal1,
                            const Eigen::RowVector3d& normal2,
                            double resolution)
{
  const Eigen::RowVector3d diff = point1 - point2;

  double n1 = normal1.norm();
  double n2 = normal2.norm();
  double squaredDistance = diff.squaredNorm();

  return 1.0 - std::fabs(n1 * n2) +
         std::sqrt(squaredDistance) / resolution * 0.4;
}

std::vector<std::vector<int>>
supervoxel_segmentation(Epoch& epoch,
                        const KDTree& kdtree,
                        double resolution,
                        int k,
                        EigenNormalSet normals)
{
  // Check if normals are provided
  if (normals.size() < epoch.cloud.rows()) {
    throw std::invalid_argument(
      "Normals must be provided for supervoxel segmentation.");
  }

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
          sizes[current] * point_2_point_VCCS_distance(epoch.cloud.row(i),
                                                       epoch.cloud.row(current),
                                                       normals.row(i),
                                                       normals.row(current),
                                                       resolution);

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
    distances.push_back(point_2_point_VCCS_distance(epoch.cloud.row(i),
                                                    epoch.cloud.row(labels[i]),
                                                    normals.row(i),
                                                    normals.row(labels[i]),
                                                    resolution));
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

      double point_distance =
        point_2_point_VCCS_distance(epoch.cloud.row(current_point),
                                    epoch.cloud.row(labels[labels[j]]),
                                    normals.row(current_point),
                                    normals.row(labels[labels[j]]),
                                    resolution);
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

Eigen::Vector3d calculateCentroid(const EigenPointCloud& cloud) {
    Eigen::Vector3d sum = cloud.colwise().sum();
    Eigen::Vector3d centroid = sum / static_cast<double>(cloud.rows());
    return centroid;

}

EigenPointCloud calculateBoundaryPoints(const EigenPointCloud& cloud) {
    
  Eigen::Vector3d BPXmax(-std::numeric_limits<double>::infinity(), 0.0, 0.0);
  Eigen::Vector3d BPXmin(std::numeric_limits<double>::infinity(), 0.0, 0.0);
  Eigen::Vector3d BPYmax(0.0, -std::numeric_limits<double>::infinity(), 0.0);
  Eigen::Vector3d BPYmin(0.0, std::numeric_limits<double>::infinity(), 0.0);
  Eigen::Vector3d BPZmax(0.0, 0.0, -std::numeric_limits<double>::infinity());
  Eigen::Vector3d BPZmin(0.0, 0.0, std::numeric_limits<double>::infinity());

  // Calculate boundary points
  for (int i = 0; i < cloud.rows(); ++i) {
      const Eigen::Vector3d& point = cloud.row(i);

      BPXmax = BPXmax.cwiseMax(point);
      BPXmin = BPXmin.cwiseMin(point);
      BPYmax = BPYmax.cwiseMax(point);
      BPYmin = BPYmin.cwiseMin(point);
      BPZmax = BPZmax.cwiseMax(point);
      BPZmin = BPZmin.cwiseMin(point);
  }
  EigenPointCloud boundary_points(6, 3);
  boundary_points << BPXmax, BPXmin, BPYmax, BPYmin, BPZmax, BPZmin;
  
  return boundary_points;
}

std::vector<EigenPointCloud> //change this to ???
segment_pc(Epoch& epoch,
           const KDTree& kdtree,
           double resolution,
           int k,
           EigenNormalSet normals
           )
{
  int minSVPnumb = 10; // rename it and make it an incoming argument
  std::vector<EigenPointCloud> clouds_SV;

  std::vector<std::vector<int>> sv_labels = supervoxel_segmentation(epoch,
                                                                    epoch.kdtree,
                                                                    resolution,
                                                                    k,
                                                                    normals);
  
   // Number of valid SV
  int svValid = 0;
  // Number of invalid SV
  int svInvalid = 0;

  EigenPointCloud centroid_points;
  
  std::vector<EigenPointCloud> boundary_points;
  //OR use it as a vector of EigenPointClouds
  for (auto& sv : sv_labels) {
    if (sv.size() < minSVPnumb) {
      svInvalid++;
      continue;
    }
    //point.cloud.row(sv[0]);
    EigenPointCloud cloud(sv.size(), 3);
    for (int i = 0; i < sv.size(); i++) {
      cloud.row(i) = epoch.cloud.row(sv[i]);
    }
    clouds_SV.push_back(cloud);

    centroid_points.row(svValid) = calculateCentroid(cloud);
    boundary_points.push_back(calculateBoundaryPoints(cloud));
    
    svValid++;
  }
  return clouds_SV;
}

} // namespace py4dgeo
