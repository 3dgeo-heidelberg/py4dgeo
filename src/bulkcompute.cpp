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
      Eigen::EigenSolver<decltype(coveval)> solver(coveval);
      auto evalues = solver.eigenvalues();

      // Sort eigenvalues through a permutation
      std::array<std::size_t, 3> permutation{ 0, 1, 2 };
      std::sort(
        permutation.begin(), permutation.end(), [&evalues](auto a, auto b) {
          return std::real(evalues[a]) < std::real(evalues[b]);
        });

      // Calculate planarity
      double planarity = (std::real(evalues[permutation[1]]) -
                          std::real(evalues[permutation[0]])) /
                         std::real(evalues[permutation[2]]);
      if (planarity > highest_planarity) {
        highest_planarity = planarity;
        const auto& evec = solver.eigenvectors().col(permutation[2]);
        for (IndexType j = 0; j < 3; ++j)
          result(i, j) = std::real(evec[j]);
      }
    }
  }
}

}