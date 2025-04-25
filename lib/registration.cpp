#include <py4dgeo/registration.hpp>
#include <py4dgeo/searchtree.hpp>

#include <Eigen/Core>

#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <vector>

#define LMBD_MAX 1e20

namespace py4dgeo {

void
transform_pointcloud_inplace(EigenPointCloudRef cloud,
                             const Transformation& trafo,
                             EigenPointCloudConstRef reduction_point,
                             EigenNormalSetRef normals)
{
  cloud.transpose() =
    (trafo.linear() * (cloud.rowwise() - reduction_point.row(0)).transpose())
      .colwise() +
    (trafo.translation() + reduction_point.row(0).transpose());

  if (normals.size() == cloud.rows())
    normals.transpose() = (trafo.linear() * normals.transpose()).transpose();
}

DisjointSet::DisjointSet(IndexType size)
  : size_(size)
  , numbers_(size, 0)
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
    floor(((upperright - lowerleft) / seed_resolution) + 1).cast<std::size_t>();

  // Assign all points to their voxels by throwing them in a hashmap
  std::unordered_map<std::size_t, std::vector<Eigen::Index>> voxelmap;
  for (IndexType i = 0; i < cloud.rows(); ++i) {
    Coordinate coord(cloud.row(i));
    MultiIndex ind =
      floor(((coord - lowerleft) / seed_resolution) + 1).cast<std::size_t>();
    voxelmap[ind[2] * voxelsizes[1] * voxelsizes[0] + ind[1] * voxelsizes[0] +
             ind[0]]
      .push_back(i);
  }

  return voxelmap.size();
}

double
point_2_point_VCCS_distance(const Eigen::RowVector3d& point1,
                            const Eigen::RowVector3d& point2,
                            const Eigen::RowVector3d& normal1,
                            const Eigen::RowVector3d& normal2,
                            double resolution)
{
  double distance = (point1 - point2).norm();

  return 1.0 - std::fabs(normal1.dot(normal2)) + distance / resolution * 0.4;
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
  NearestNeighborsDistanceResult result;
  kdtree.nearest_neighbors_with_distances(epoch.cloud, result, k);

  int supervoxels_amount = epoch.cloud.rows();

  // calculate lambda for segmentation
  DistanceVector lambda_distances(epoch.cloud.rows());
  for (size_t i = 0; i < epoch.cloud.rows(); ++i) {
    int current = result[i].first[1];
    lambda_distances.push_back(
      point_2_point_VCCS_distance(epoch.cloud.row(i),
                                  epoch.cloud.row(current),
                                  normals.row(i),
                                  normals.row(current),
                                  resolution));
  }

  double lambda = std::max(std::numeric_limits<double>::epsilon(),
                           median_calculation(lambda_distances));

  // initialize temporary vars for supervoxel segmentation
  std::vector<int> temporary_supervoxels(epoch.cloud.rows());
  std::iota(temporary_supervoxels.begin(), temporary_supervoxels.end(), 0);
  std::vector<int> sizes(epoch.cloud.rows(), 1);
  std::queue<int> point_queue;
  std::vector<std::vector<long unsigned int>> neighborIndexes(result.size());

  std::vector<int> queue_DEL(epoch.cloud.rows());
  std::vector<bool> isVisited(epoch.cloud.rows(), false);

  for (size_t i = 0; i < result.size(); ++i) {

    for (size_t j = 1; j < result[i].first.size(); ++j) {
      neighborIndexes[i].push_back(result[i].first[j]);
    }
  }

  DistanceVector distances;
  // searching for supervoxels and first segmentation
  for (; lambda < LMBD_MAX; lambda *= 2.0) {
    for (auto i : temporary_supervoxels) {
      if (neighborIndexes[i].empty())
        continue;

      isVisited[i] = true;
      int front = 0, back = 1;
      queue_DEL[front++] = i;
      for (auto j : neighborIndexes[i]) {
        j = set.Find(j);
        if (!isVisited[j]) {
          isVisited[j] = true;
          queue_DEL[back++] = j;
        }
      }

      std::vector<long unsigned int> neighborIndexes_per_point;

      while (front < back) {
        int current = queue_DEL[front++];
        if (i == current)
          continue;

        double loss =
          sizes[current] * point_2_point_VCCS_distance(epoch.cloud.row(i),
                                                       epoch.cloud.row(current),
                                                       normals.row(i),
                                                       normals.row(current),
                                                       resolution);

        double improvement = lambda - loss;

        if (improvement > 0.0) {
          set.Union(current, i, false);
          sizes[i] += sizes[current];

          for (auto k : neighborIndexes[current]) {
            k = set.Find(k);
            if (!isVisited[k]) {
              isVisited[k] = true;
              queue_DEL[back++] = k;
            }
          }

          neighborIndexes[current].clear();
          if (--supervoxels_amount == n_supervoxels)
            break;
        } else {
          neighborIndexes_per_point.push_back(current);
        }
      }

      neighborIndexes[i].swap(neighborIndexes_per_point);

      for (int j = 0; j < back; ++j) {
        isVisited[queue_DEL[j]] = false;
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
                                    epoch.cloud.row(labels[j]),
                                    normals.row(current_point),
                                    normals.row(labels[j]),
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

Eigen::Vector3d
calculateCentroid(EigenPointCloudConstRef cloud)
{
  return cloud.colwise().mean();
}

EigenPointCloud
calculateBoundaryPoints(EigenPointCloudConstRef cloud)
{
  Eigen::Vector3d BPXmax(-std::numeric_limits<double>::infinity(), 0.0, 0.0);
  Eigen::Vector3d BPXmin(std::numeric_limits<double>::infinity(), 0.0, 0.0);
  Eigen::Vector3d BPYmax(0.0, -std::numeric_limits<double>::infinity(), 0.0);
  Eigen::Vector3d BPYmin(0.0, std::numeric_limits<double>::infinity(), 0.0);
  Eigen::Vector3d BPZmax(0.0, 0.0, -std::numeric_limits<double>::infinity());
  Eigen::Vector3d BPZmin(0.0, 0.0, std::numeric_limits<double>::infinity());

  // Calculate boundary points
  for (int i = 0; i < cloud.rows(); ++i) {
    const Eigen::Vector3d& point = cloud.row(i);

    if (point.x() > BPXmax.x())
      BPXmax = point;
    if (point.x() < BPXmin.x())
      BPXmin = point;
    if (point.y() > BPYmax.y())
      BPYmax = point;
    if (point.y() < BPYmin.y())
      BPYmin = point;
    if (point.z() > BPZmax.z())
      BPZmax = point;
    if (point.z() < BPZmin.z())
      BPZmin = point;
  }

  EigenPointCloud boundary_points(6, 3);
  boundary_points.row(0) = BPXmax;
  boundary_points.row(1) = BPXmin;
  boundary_points.row(2) = BPYmax;
  boundary_points.row(3) = BPYmin;
  boundary_points.row(4) = BPZmax;
  boundary_points.row(5) = BPZmin;

  return boundary_points;
}

std::vector<Supervoxel>
segment_pc(Epoch& epoch,
           const KDTree& kdtree,
           EigenNormalSet normals,
           double seed_resolution,
           int k,
           int minSVPvalue = 10)
{
  std::vector<Supervoxel> clouds_SV;

  std::vector<std::vector<int>> sv_labels =
    supervoxel_segmentation(epoch, epoch.kdtree, seed_resolution, k, normals);

  //  Number of valid SV
  int svValid = 0;
  int svInvalid = 0;

  for (auto& sv_iter : sv_labels) {
    Supervoxel sv;
    if (sv_iter.size() < minSVPvalue) {
      svInvalid++;
      continue;
    }

    sv.cloud.resize(sv_iter.size(), 3);
    sv.normals.resize(sv_iter.size(), 3);
    for (int i = 0; i < sv_iter.size(); i++) {
      sv.cloud.row(i) = epoch.cloud.row(sv_iter[i]);
      sv.normals.row(i) = normals.row(sv_iter[i]);
    }

    sv.centroid = calculateCentroid(sv.cloud);
    sv.boundary_points = calculateBoundaryPoints(sv.cloud);

    clouds_SV.push_back(sv);
    svValid++;
  }

  return clouds_SV;
}

// function for calculating Jacobian for point to plane method
Eigen::Matrix<double, 1, 6>
plane_Jacobian(Eigen::Vector3d Rot_a, Eigen::Vector3d n)
{
  Eigen::Matrix<double, 1, 6> J;
  J.block<1, 3>(0, 0) = -Rot_a.cross(n);
  J.block<1, 3>(0, 3) = n;
  return J;
}

// function for calculating rotation and transformation matrix
std::pair<Eigen::Matrix3d, Eigen::Vector3d>
set_rot_trans(const Eigen::Matrix<double, 6, 1>& euler_array)
{

  double alpha = euler_array[0];
  double beta = euler_array[1];
  double gamma = euler_array[2];
  double x = euler_array[3];
  double y = euler_array[4];
  double z = euler_array[5];

  Eigen::Matrix3d Rot_z;
  Rot_z << cos(gamma), sin(gamma), 0, -sin(gamma), cos(gamma), 0, 0, 0, 1;
  Eigen::Matrix3d Rot_y;
  Rot_y << cos(beta), 0, -sin(beta), 0, 1, 0, sin(beta), 0, cos(beta);
  Eigen::Matrix3d Rot_x;
  Rot_x << 1, 0, 0, 0, cos(alpha), sin(alpha), 0, -sin(alpha), cos(alpha);
  Eigen::Matrix3d Rot = Rot_z * Rot_y * Rot_x;

  return std::make_pair(Rot, Eigen::Vector3d(x, y, z));
}

// function for finding a transformation that fits two point clouds onto
// each other using Gauss-Newton method
Eigen::Matrix4d
fit_transform_GN(EigenPointCloudConstRef trans_cloud,
                 EigenPointCloudConstRef reference_cloud,
                 EigenNormalSetConstRef reference_normals)
{

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  int min_matr_size = std::min(trans_cloud.rows(), reference_cloud.rows());
  // hessian matrix
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> euler_array = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix3d Rot;
  Eigen::Vector3d trans;
  std::tie(Rot, trans) = set_rot_trans(euler_array);

  int chi = 0;
  for (int i = 0; i < min_matr_size; i++) {
    Eigen::Vector3d a;
    Eigen::Vector3d b;
    Eigen::Vector3d n;
    a = trans_cloud.row(i);
    b = reference_cloud.row(i);
    n = reference_normals.row(i);

    Eigen::Vector3d Rot_a = Rot * a;
    double e = (Rot_a + trans - b).dot(n);

    Eigen::Matrix<double, 1, 6> J = plane_Jacobian(Rot_a, n);
    H += J.transpose() * J;
    g += J.transpose() * e;
  }

  Eigen::Matrix<double, 6, 1> update_euler;
  update_euler = -H.inverse() * g;
  euler_array += update_euler;

  Eigen::Matrix3d Rot_f;
  Eigen::Vector3d trans_f;
  std::tie(Rot_f, trans_f) = set_rot_trans(euler_array);
  T.block<3, 3>(0, 0) = Rot_f;
  T.block<3, 1>(0, 3) = trans_f;

  return T;
}

} // namespace py4dgeo
