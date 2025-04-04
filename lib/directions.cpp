#include <Eigen/Eigen>
#include <Eigen/Eigenvalues>

#include "py4dgeo/compute.hpp"
#include "py4dgeo/kdtree.hpp"
#include "py4dgeo/octree.hpp"
#include "py4dgeo/openmp.hpp"
#include "py4dgeo/py4dgeo.hpp"
#include "py4dgeo/searchtree.hpp"

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
                              std::vector<double>& used_radii,
                              SearchTree tree)
{
  used_radii.resize(corepoints.rows());
  const auto orientation_vector = orientation.row(0).transpose();

  auto radius_search = get_radius_search_function(epoch, normal_radii, tree);

  // Instantiate a container for the first thrown exception in
  // the following parallel region.
  CallbackExceptionVault vault;
#ifdef PY4DGEO_WITH_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (IndexType i = 0; i < corepoints.rows(); ++i) {
    vault.run([&]() {
      double highest_planarity = 0.0;
      Eigen::Vector3d query_point = corepoints.row(i);
      for (size_t r = 0; r < normal_radii.size(); ++r) {

        std::vector<IndexType> points;
        radius_search(query_point, r, points);

        auto subset = epoch.cloud(points, Eigen::all);

        // Calculate covariance matrix
        auto centered = (subset.rowwise() - subset.colwise().mean()).eval();
        auto cov =
          ((centered.adjoint() * centered) / double(subset.rows() - 1)).eval();

        // Calculate Eigen vectors
        Eigen::SelfAdjointEigenSolver<decltype(cov)> solver(cov);
        const auto& evalues = solver.eigenvalues();
        const auto evec = solver.eigenvectors().col(0);

        // Calculate planarity
        double planarity = (evalues[1] - evalues[0]) / evalues[2];
        if (planarity > highest_planarity) {
          highest_planarity = planarity;

          double sign = (evec.dot(orientation_vector) < 0.0) ? -1.0 : 1.0;
          result.row(i) = sign * evec;
          used_radii[i] = normal_radii[r];
        }
      }
    });
  }

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
