#include <Eigen/Eigen>
#include <Eigen/Eigenvalues>

#include "py4dgeo/compute.hpp"
#include "py4dgeo/kdtree.hpp"
#include "py4dgeo/openmp.hpp"
#include "py4dgeo/py4dgeo.hpp"

#include <algorithm>
#include <complex>
#include <vector>

namespace py4dgeo {

void
compute_multiscale_directions(const Epoch& epoch,
                              EigenPointCloudConstRef corepoints,
                              const std::vector<double>& normal_radii,
                              EigenNormalSetConstRef orientation,
                              EigenNormalSetRef result)
{
  // Instantiate a container for the first thrown exception in
  // the following parallel region.
  CallbackExceptionVault vault;
#ifdef PY4DGEO_WITH_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
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
        auto centered = subset.rowwise() - subset.colwise().mean();
        auto cov = (centered.adjoint() * centered) / double(subset.rows() - 1);
        auto coveval = cov.eval();

        // Calculate Eigen vectors
        Eigen::SelfAdjointEigenSolver<decltype(coveval)> solver(coveval);
        const auto& evalues = solver.eigenvalues();

        // Calculate planarity
        double planarity = (evalues[1] - evalues[0]) / evalues[2];
        if (planarity > highest_planarity) {
          highest_planarity = planarity;

          double prod =
            (solver.eigenvectors().col(0).dot(orientation.row(0).transpose()));
          double sign = (prod < 0.0) ? -1.0 : 1.0;
          result.row(i) = sign * solver.eigenvectors().col(0);
        }
      }
    });
  }

  // Potentially rethrow an exception that occurred in above parallel region
  vault.rethrow();
}

}
