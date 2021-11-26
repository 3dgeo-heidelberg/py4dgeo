#include <Eigen/Eigen>

#ifdef PY4DGEO_WITH_OPENMP
#include <omp.h>
#endif

#include "py4dgeo/compute.hpp"
#include "py4dgeo/kdtree.hpp"
#include "py4dgeo/py4dgeo.hpp"

namespace py4dgeo {

void
compute_distances(EigenPointCloudConstRef corepoints,
                  double scale,
                  const Epoch& epoch1,
                  const Epoch& epoch2,
                  EigenPointCloudConstRef directions,
                  double max_cylinder_length,
                  DistanceVector& distances,
                  UncertaintyVector& uncertainties,
                  const WorkingSetFinderCallback& workingsetfinder,
                  const UncertaintyMeasureCallback& uncertaintycalculator)
{
  // Resize the output data structures
  distances.resize(corepoints.rows());
  uncertainties.resize(corepoints.rows());

#ifdef PY4DGEO_WITH_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (IndexType i = 0; i < corepoints.rows(); ++i) {
    // Either choose the ith row or the first (if there is no per-corepoint
    // direction)
    auto dir = directions.row(directions.rows() > 1 ? i : 0);

    auto subset1 = workingsetfinder(
      epoch1, scale, corepoints.row(i), dir, max_cylinder_length, i);
    auto subset2 = workingsetfinder(
      epoch2, scale, corepoints.row(i), dir, max_cylinder_length, i);

    // Distance calculation
    distances[i] = dir.dot(subset2.colwise().mean() - subset1.colwise().mean());

    // Uncertainty calculation
    uncertainties[i] = uncertaintycalculator(subset1, subset2, dir);
  }
}

EigenPointCloud
radius_workingset_finder(const Epoch& epoch,
                         double radius,
                         EigenPointCloudConstRef,
                         EigenPointCloudConstRef,
                         double,
                         IndexType core_idx)
{
  // Find the working set in the other epoch
  KDTree::RadiusSearchResult points;
  epoch.kdtree.precomputed_radius_search(core_idx, radius, points);
  return epoch.cloud(points, Eigen::all);
}

EigenPointCloud
cylinder_workingset_finder(const Epoch& epoch,
                           double radius,
                           EigenPointCloudConstRef corepoint,
                           EigenPointCloudConstRef direction,
                           double max_cylinder_length,
                           IndexType core_idx)
{
  // Cut the cylinder into N segments, perform radius searches around the
  // segment midpoints and create the union of indices. Afterwards, select
  // only those points that are within the cylinder

  // The number of segments - later cast to int
  double N = 1.0;
  if (max_cylinder_length > radius)
    N = std::ceil(max_cylinder_length / radius);
  else
    max_cylinder_length = radius;

  // The search radius for each segment
  double r_cyl = std::sqrt(radius * radius +
                           max_cylinder_length * max_cylinder_length / (N * N));

  // Perform radius searches and merge results
  std::vector<IndexType> merged;
  for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i) {
    auto qp = (corepoint.row(0) +
               ((2 * i + 1 - N) / N) * max_cylinder_length * direction.row(0))
                .eval();
    KDTree::RadiusSearchResult ball_points;
    epoch.kdtree.radius_search(&(qp(0, 0)), r_cyl, ball_points);
    merged.reserve(merged.capacity() + ball_points.size());
    std::copy(
      ball_points.begin(), ball_points.end(), std::back_inserter(merged));
  }

  // Making indices unique
  std::sort(merged.begin(), merged.end());
  merged.erase(std::unique(merged.begin(), merged.end()), merged.end());

  // Extracting points
  auto superset = epoch.cloud(merged, Eigen::all);

  // Calculate the squared distances to the cylinder axis and to the plane
  // perpendicular to the axis that contains the corepoint
  auto to_corepoint = (superset.rowwise() - corepoint.row(0)).eval();
  auto to_corepoint_plane = (to_corepoint * direction.transpose()).eval();
  auto to_axis2 = (to_corepoint - to_corepoint_plane * direction)
                    .rowwise()
                    .squaredNorm()
                    .eval();

  // Non-performance oriented version of index extraction. There should
  // be a version using Eigen masks, but I could not find it.
  std::vector<Eigen::Index> indices;
  for (Eigen::Index i = 0; i < superset.rows(); ++i)
    if ((to_axis2(i) <= radius * radius) &&
        (std::abs(to_corepoint_plane(i)) <= max_cylinder_length))
      indices.push_back(i);

  // Select only those indices that are within the cylinder
  return superset(indices, Eigen::all);
}

double
variance(EigenPointCloudConstRef subset, EigenPointCloudConstRef direction)
{
  auto centered = subset.rowwise() - subset.colwise().mean();
  auto cov = (centered.adjoint() * centered) / double(subset.rows() - 1);
  auto multiplied = direction.row(0) * cov * direction.row(0).transpose();
  return multiplied.eval()(0, 0);
}

DistanceUncertainty
standard_deviation_uncertainty(EigenPointCloudConstRef set1,
                               EigenPointCloudConstRef set2,
                               EigenPointCloudConstRef direction)
{
  double variance1 = variance(set1, direction);
  double variance2 = variance(set2, direction);

  // Calculate the standard deviations for both point clouds
  double stddev1 = std::sqrt(variance1);
  double stddev2 = std::sqrt(variance2);

  // Calculate the level of  from above variances
  double lodetection =
    1.96 * std::sqrt(variance1 / static_cast<double>(set1.rows()) +
                     variance2 / static_cast<double>(set2.rows()));

  return DistanceUncertainty{
    lodetection, stddev1, set1.rows(), stddev2, set2.rows()
  };
}

} // namespace py4dgeo
