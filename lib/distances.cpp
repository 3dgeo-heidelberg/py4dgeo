#include <Eigen/Eigen>

#include "py4dgeo/compute.hpp"
#include "py4dgeo/kdtree.hpp"
#include "py4dgeo/openmp.hpp"
#ifdef PY4DGEO_WITH_TBB
#include <tbb/parallel_for.h>
#endif
#include "py4dgeo/py4dgeo.hpp"

#include <algorithm>

namespace py4dgeo {

void
compute_distances(
  EigenPointCloudConstRef corepoints,
  double scale,
  const Epoch& epoch1,
  const Epoch& epoch2,
  EigenNormalSetConstRef directions,
  double max_distance,
  double registration_error,
  DistanceVector& distances,
  UncertaintyVector& uncertainties,
  const WorkingSetFinderCallback& workingsetfinder,
  const DistanceUncertaintyCalculationCallback& distancecalculator)
{
  // Resize the output data structures
  distances.resize(corepoints.rows());
  uncertainties.resize(corepoints.rows());

  // Instantiate a container for the first thrown exception in
  // the following parallel region.
  CallbackExceptionVault vault;

#ifdef PY4DGEO_WITH_TBB
  tbb::parallel_for(
    tbb::blocked_range<IndexType>(0, corepoints.rows()),
    [&](const tbb::blocked_range<IndexType>& r) {
      for (IndexType i = r.begin(); i < r.end(); ++i) {
        // Either choose the ith row or the first (if there is no per-corepoint
        // direction)
        auto dir = directions.row(directions.rows() > 1 ? i : 0);

        WorkingSetFinderParameters params1{
          epoch1, scale, corepoints.row(i), dir, max_distance
        };
        auto subset1 = workingsetfinder(params1);
        WorkingSetFinderParameters params2{
          epoch2, scale, corepoints.row(i), dir, max_distance
        };
        auto subset2 = workingsetfinder(params2);

        // Distance calculation
        DistanceUncertaintyCalculationParameters d_params{
          subset1, subset2, corepoints.row(i), dir, registration_error
        };
        auto dist = distancecalculator(d_params);

        // Write distances into the resulting array
        distances[i] = std::get<0>(dist);
        uncertainties[i] = std::get<1>(dist);
      }
    });
#elif defined(PY4DGEO_WITH_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (IndexType i = 0; i < corepoints.rows(); ++i) {
    vault.run([&]() {
      // Either choose the ith row or the first (if there is no per-corepoint
      // direction)
      auto dir = directions.row(directions.rows() > 1 ? i : 0);

      WorkingSetFinderParameters params1{
        epoch1, scale, corepoints.row(i), dir, max_distance
      };
      auto subset1 = workingsetfinder(params1);
      WorkingSetFinderParameters params2{
        epoch2, scale, corepoints.row(i), dir, max_distance
      };
      auto subset2 = workingsetfinder(params2);

      // Distance calculation
      DistanceUncertaintyCalculationParameters d_params{
        subset1, subset2, corepoints.row(i), dir, registration_error
      };
      auto dist = distancecalculator(d_params);

      // Write distances into the resulting array
      distances[i] = std::get<0>(dist);
      uncertainties[i] = std::get<1>(dist);
    });
  }

  // Potentially rethrow an exception that occurred in above parallel region
  vault.rethrow();
}

EigenPointCloud
radius_workingset_finder(const WorkingSetFinderParameters& params)
{
  // Find the working set in the other epoch
  KDTree::RadiusSearchResult points;
  params.epoch.kdtree.radius_search(
    params.corepoint.data(), params.radius, points);
  return params.epoch.cloud(points, Eigen::all);
}

EigenPointCloud
cylinder_workingset_finder(const WorkingSetFinderParameters& params)
{
  // Cut the cylinder into N segments, perform radius searches around the
  // segment midpoints and create the union of indices. Afterwards, select
  // only those points that are within the cylinder

  // The number of segments - later cast to int
  double N = 1.0;
  double cylinder_length = params.max_distance;
  if (cylinder_length > params.radius)
    N = std::ceil(cylinder_length / params.radius);
  else
    cylinder_length = params.radius;

  // The search radius for each segment
  double r_cyl = std::sqrt(params.radius * params.radius +
                           cylinder_length * cylinder_length / (N * N));

  // Perform radius searches and merge results
  std::vector<IndexType> merged;
  for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i) {
    auto qp = (params.corepoint.row(0) +
               (static_cast<double>(2 * i + 1 - N) / static_cast<double>(N)) *
                 cylinder_length * params.cylinder_axis.row(0))
                .eval();
    KDTree::RadiusSearchResult ball_points;
    params.epoch.kdtree.radius_search(&(qp(0, 0)), r_cyl, ball_points);
    merged.reserve(merged.capacity() + ball_points.size());

    // Extracting points
    auto superset = params.epoch.cloud(ball_points, Eigen::all);

    // Calculate the squared distances to the cylinder axis and to the plane
    // perpendicular to the axis that contains the corepoint
    auto to_midpoint = (superset.rowwise() - qp.row(0)).eval();
    auto to_midpoint_plane =
      (to_midpoint * params.cylinder_axis.transpose()).eval();
    auto to_axis2 = (to_midpoint - to_midpoint_plane * params.cylinder_axis)
                      .rowwise()
                      .squaredNorm()
                      .eval();

    // Non-performance oriented version of index extraction. There should
    // be a version using Eigen masks, but I could not find it.
    for (Eigen::Index i = 0; i < superset.rows(); ++i)
      if ((to_axis2(i) <= params.radius * params.radius) &&
          (std::abs(to_midpoint_plane(i)) < (cylinder_length / N)))
        merged.push_back(ball_points[i]);
  }

  // Select only those indices that are within the cylinder
  return params.epoch.cloud(merged, Eigen::all);
}

double
variance(EigenPointCloudConstRef subset,
         const Eigen::Matrix<double, 1, 3>& mean,
         EigenNormalSetConstRef direction)
{
  auto centered = subset.rowwise() - mean;
  auto cov = (centered.adjoint() * centered) / double(subset.rows() - 1);
  auto multiplied = direction.row(0) * cov * direction.row(0).transpose();
  return multiplied.eval()(0, 0);
}

std::tuple<double, DistanceUncertainty>
mean_stddev_distance(const DistanceUncertaintyCalculationParameters& params)
{
  std::tuple<double, DistanceUncertainty> ret;

  auto mean1 = params.workingset1.colwise().mean().eval();
  auto mean2 = params.workingset2.colwise().mean().eval();
  std::get<0>(ret) = params.normal.row(0).dot(mean2 - mean1);

  double variance1 = variance(params.workingset1, mean1, params.normal);
  double variance2 = variance(params.workingset2, mean2, params.normal);

  // Calculate the standard deviations for both point clouds
  double stddev1 = std::sqrt(variance1);
  double stddev2 = std::sqrt(variance2);

  // Calculate the level of detection from above variances
  double lodetection =
    1.96 *
    (std::sqrt(variance1 / static_cast<double>(params.workingset1.rows()) +
               variance2 / static_cast<double>(params.workingset2.rows())) +
     params.registration_error);

  std::get<1>(ret).lodetection = lodetection;
  std::get<1>(ret).spread1 = stddev1;
  std::get<1>(ret).num_samples1 = params.workingset1.rows();
  std::get<1>(ret).spread2 = stddev2;
  std::get<1>(ret).num_samples2 = params.workingset2.rows();

  return ret;
}

double
find_element_with_averaging(Eigen::Matrix<double, Eigen::Dynamic, 1>& v,
                            std::size_t start,
                            std::size_t pos,
                            bool average)
{
  // ADL-friendly version of using STL algorithms
  using std::max_element;
  using std::nth_element;

  // Perform a partial sorting
  nth_element(v.begin() + start, v.begin() + pos, v.end());
  auto med = v[pos];

  if (average) {
    auto max_it = max_element(v.begin() + start, v.begin() + pos);
    med = (*max_it + med) / 2.0;
  }

  return med;
}

std::array<double, 2>
median(Eigen::Matrix<double, Eigen::Dynamic, 1>& v)
{
  if (v.size() == 0)
    return { std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::quiet_NaN() };

  // General implementation idea taken from the followins posts
  // * https://stackoverflow.com/a/34077478
  // * https://stackoverflow.com/a/11965377
  auto q1 = find_element_with_averaging(v, 0, v.size() / 4, v.size() % 4 == 0);
  auto q2 = find_element_with_averaging(
    v, v.size() / 4, v.size() / 2, v.size() % 2 == 0);
  auto q3 = find_element_with_averaging(
    v, v.size() / 2, 3 * v.size() / 4, v.size() % 4 == 0);

  return { q2, q3 - q1 };
}

std::tuple<double, DistanceUncertainty>
median_iqr_distance(const DistanceUncertaintyCalculationParameters& params)
{
  // Calculate distributions across the cylinder axis
  auto dist1 = (params.workingset1 * params.normal.row(0).transpose()).eval();
  auto dist2 = (params.workingset2 * params.normal.row(0).transpose()).eval();

  // Find median and interquartile range of that distribution
  auto [med1, iqr1] = median(dist1);
  auto [med2, iqr2] = median(dist2);

  return std::make_tuple(
    med2 - med1,
    DistanceUncertainty{
      1.96 * (std::sqrt(
                iqr1 * iqr1 / static_cast<double>(params.workingset1.rows()) +
                iqr2 * iqr2 / static_cast<double>(params.workingset2.rows())) +
              params.registration_error),
      iqr1,
      params.workingset1.rows(),
      iqr2,
      params.workingset2.rows() });
}

} // namespace py4dgeo
