#include <Eigen/Eigen>

#include "py4dgeo/py4dgeo.hpp"

namespace py4dgeo {

template<typename WorkingSetFinder>
void
compute_distances_impl(WorkingSetFinder&& workingsetfinder,
                       EigenPointCloudRef corepoints,
                       double scale,
                       EigenPointCloudRef cloud1,
                       const KDTree& kdtree1,
                       EigenPointCloudRef cloud2,
                       const KDTree& kdtree2,
                       EigenPointCloudRef direction,
                       double max_cylinder_length,
                       EigenVectorRef distances)
{
  for (IndexType i = 0; i < corepoints.rows(); ++i) {
    auto subset1 = workingsetfinder(cloud1,
                                    kdtree1,
                                    scale,
                                    corepoints.row(i),
                                    direction.row(i),
                                    max_cylinder_length,
                                    i);
    auto subset2 = workingsetfinder(cloud2,
                                    kdtree2,
                                    scale,
                                    corepoints.row(i),
                                    direction.row(i),
                                    max_cylinder_length,
                                    i);

    // Distance calculation
    double dist =
      direction.row(i).dot(subset1.colwise().mean() - subset2.colwise().mean());

    // Store in result vector
    distances(i, 0) = std::abs(dist);
  }
}

EigenPointCloud
radius_workingset_finder(EigenPointCloudRef cloud,
                         const KDTree& kdtree,
                         double radius,
                         EigenPointCloudRef,
                         EigenPointCloudRef,
                         double,
                         IndexType core_idx)
{
  // Find the working set in the other epoch
  KDTree::RadiusSearchResult points;
  kdtree.precomputed_radius_search(core_idx, radius, points);
  return cloud(points, Eigen::all);
}

EigenPointCloud
cylinder_workingset_finder(EigenPointCloudRef cloud,
                           const KDTree& kdtree,
                           double radius,
                           EigenPointCloudRef corepoint,
                           EigenPointCloudRef direction,
                           double max_cylinder_length,
                           IndexType core_idx)
{
  // Find the points in the radius of max_cylinder_length
  KDTree::RadiusSearchResult ball_points;
  kdtree.precomputed_radius_search(core_idx, max_cylinder_length, ball_points);
  auto superset = cloud(ball_points, Eigen::all);

  // Calculate the squared distances to the cylinder axis
  auto distances = (superset.rowwise() - corepoint.row(0))
                     .rowwise()
                     .cross(direction.row(0))
                     .rowwise()
                     .squaredNorm();

  // Non-performance oriented version of index extraction. There should
  // be a version using Eigen masks, but I could not find it.
  std::vector<Eigen::Index> indices;
  for (Eigen::Index i = 0; i < superset.rows(); ++i)
    if (distances(i, 0) < radius * radius)
      indices.push_back(i);

  // Select only those indices that are within the cylinder
  return superset(indices, Eigen::all);
}

void
compute_distances(EigenPointCloudRef corepoints,
                  double scale,
                  EigenPointCloudRef cloud1,
                  const KDTree& kdtree1,
                  EigenPointCloudRef cloud2,
                  const KDTree& kdtree2,
                  EigenPointCloudRef direction,
                  double max_cylinder_length,
                  EigenVectorRef distances)
{
  if (max_cylinder_length > scale)
    return compute_distances_impl(cylinder_workingset_finder,
                                  corepoints,
                                  scale,
                                  cloud1,
                                  kdtree1,
                                  cloud2,
                                  kdtree2,
                                  direction,
                                  max_cylinder_length,
                                  distances);
  else
    return compute_distances_impl(radius_workingset_finder,
                                  corepoints,
                                  scale,
                                  cloud1,
                                  kdtree1,
                                  cloud2,
                                  kdtree2,
                                  direction,
                                  max_cylinder_length,
                                  distances);
}

} // namespace py4dgeo
