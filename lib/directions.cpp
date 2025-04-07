#include <Eigen/Eigen>
#include <Eigen/Eigenvalues>

#include "py4dgeo/compute.hpp"
#include "py4dgeo/kdtree.hpp"
#include "py4dgeo/openmp.hpp"
#ifdef PY4DGEO_WITH_TBB
#include <tbb/parallel_for.h>
#endif
#include "py4dgeo/py4dgeo.hpp"

#include <algorithm>
#include <complex>
#include <vector>

#include <iostream>

namespace py4dgeo {

void
compute_multiscale_directions(const Epoch& epoch,
                              EigenPointCloudConstRef corepoints,
                              const std::vector<double>& normal_radii,
                              EigenNormalSetConstRef orientation,
                              EigenNormalSetRef result,
                              std::vector<double>& used_radii)
{
  used_radii.resize(corepoints.rows());
  const auto orientation_vector = orientation.row(0).transpose();

  // Instantiate a container for the first thrown exception in
  // the following parallel region.
  CallbackExceptionVault vault;

#ifdef PY4DGEO_WITH_TBB
  tbb::parallel_for(
    tbb::blocked_range<IndexType>(0, corepoints.rows()),
    [&](const tbb::blocked_range<IndexType>& r) {
      for (IndexType i = r.begin(); i < r.end(); ++i) {
        double highest_planarity = 0.0;
        for (auto radius : normal_radii) {
          // Find the working set on this scale
          KDTree::RadiusSearchResult points;
          auto qp = corepoints.row(i).eval();
          epoch.kdtree.radius_search(&(qp(0, 0)), radius, points);
          auto subset = epoch.cloud(points, Eigen::all);

          // Calculate covariance matrix
          auto centered = (subset.rowwise() - subset.colwise().mean()).eval();
          auto cov =
            ((centered.adjoint() * centered) / double(subset.rows() - 1))
              .eval();

          // Calculate Eigen vectors
          Eigen::SelfAdjointEigenSolver<decltype(cov)> solver(cov);
          const auto& evalues = solver.eigenvalues();

          // Calculate planarity
          double planarity = (evalues[1] - evalues[0]) / evalues[2];
          if (planarity > highest_planarity) {
            highest_planarity = planarity;

            double prod = solver.eigenvectors().col(0).dot(orientation_vector);
            double sign = (prod < 0.0) ? -1.0 : 1.0;
            result.row(i) = sign * solver.eigenvectors().col(0);
            used_radii[i] = radius;
          }
        }
      }
    });
#elif defined(PY4DGEO_WITH_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
  for (IndexType i = 0; i < corepoints.rows(); ++i) {
    vault.run([&]() {
      double highest_planarity = 0.0;
      for (auto radius : normal_radii) {
        // Find the working set on this scale
        KDTree::RadiusSearchResult points;
        auto qp = corepoints.row(i).eval();
        epoch.kdtree.radius_search(&(qp(0, 0)), radius, points);
        auto subset = epoch.cloud(points, Eigen::all);

        // Calculate covariance matrix
        auto centered = (subset.rowwise() - subset.colwise().mean()).eval();
        auto cov =
          ((centered.adjoint() * centered) / double(subset.rows() - 1)).eval();

        // Calculate Eigen vectors
        Eigen::SelfAdjointEigenSolver<decltype(cov)> solver(cov);
        const auto& evalues = solver.eigenvalues();

        // Calculate planarity
        double planarity = (evalues[1] - evalues[0]) / evalues[2];
        if (planarity > highest_planarity) {
          highest_planarity = planarity;

          double prod = (solver.eigenvectors().col(0).dot(orientation_vector));
          double sign = (prod < 0.0) ? -1.0 : 1.0;
          result.row(i) = sign * solver.eigenvectors().col(0);
          used_radii[i] = radius;
        }
      }
    });
  }
#endif

  // Potentially rethrow an exception that occurred in above parallel region
  vault.rethrow();
}

std::vector<double>
compute_correspondence_distances(const Epoch& epoch,
                                 EigenPointCloudConstRef transformated_pc,
                                 std::vector<EigenPointCloud> corepoints,
                                 unsigned int check_size)
{

  KDTree::NearestNeighborsDistanceResult result;
  epoch.kdtree.nearest_neighbors_with_distances(transformated_pc, result, 1);
  std::vector<double> p2pdist(transformated_pc.rows());

#ifdef PY4DGEO_WITH_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (IndexType i = 0; i < transformated_pc.rows(); ++i) {
    if (epoch.cloud.rows() != check_size) {
      auto subset = corepoints[result[i].first[0]];
      // Calculate covariance matrix
      auto centered = (subset.rowwise() - subset.colwise().mean()).eval();
      auto cov =
        ((centered.adjoint() * centered) / double(subset.rows() - 1)).eval();
      // Calculate Eigen vectors
      Eigen::SelfAdjointEigenSolver<decltype(cov)> solver(cov);
      Eigen::Vector3d normal_vector = solver.eigenvectors().col(0);
      // calculate cor distance
      Eigen::Vector3d displacement_vector =
        epoch.cloud.row(result[i].first[0]) - transformated_pc.row(i);
      p2pdist[i] = std::abs(displacement_vector.dot(normal_vector));

    }

    else
      p2pdist[i] = std::sqrt(result[i].second[0]);
  }
  return p2pdist;
}

} // namespace py4dgeo
