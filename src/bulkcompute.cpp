#include <Eigen/Eigen>
#include <Eigen/Eigenvalues>

#include "py4dgeo/py4dgeo.hpp"

#include <algorithm>
#include <complex>
#include <vector>

namespace py4dgeo {

void
compute_multiscale_directions(const EigenPointCloudRef& cloud,
                              const EigenPointCloudRef& corepoints,
                              const std::vector<double>& scales,
                              const KDTree& kdtree,
                              EigenPointCloudRef result)
{
  // TODO: Make sure that precomputation has been triggered.

  for (IndexType i = 0; i < corepoints.rows(); ++i) {
    double highest_planarity = 0.0;
    for (auto scale : scales) {
      // Find the working set on this scale
      KDTree::RadiusSearchResult points;
      kdtree.precomputed_radius_search(i, scale, points);
      auto subset = cloud(points, Eigen::all);

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
        const auto& evec = solver.eigenvectors().col(2);
        for (IndexType j = 0; j < 3; ++j)
          result(i, j) = std::real(evec[j]);
      }
    }
  }
}

}