#include <Eigen/Eigen>

#include "py4dgeo/py4dgeo.hpp"

namespace py4dgeo {

void
compute_distances(const EigenPointCloudRef& corepoints,
                  double scale,
                  const EigenPointCloudRef& cloud1,
                  const KDTree& kdtree1,
                  const EigenPointCloudRef& cloud2,
                  const KDTree& kdtree2,
                  const EigenPointCloudRef& direction,
                  EigenVectorRef distances)
{
  for (IndexType i = 0; i < corepoints.rows(); ++i) {
    // Find the working set in the reference epoch
    KDTree::RadiusSearchResult points1;
    kdtree1.precomputed_radius_search(i, scale, points1);
    auto subset1 = cloud1(points1, Eigen::all);

    // Find the working set in the other epoch
    KDTree::RadiusSearchResult points2;
    kdtree2.precomputed_radius_search(i, scale, points2);
    auto subset2 = cloud2(points2, Eigen::all);

    // Distance calculation
    double dist =
      direction.row(i).dot(subset1.colwise().mean() - subset2.colwise().mean());

    // Store in result vector
    distances(i, 0) = std::abs(dist);
  }
}

} // namespace py4dgeo
