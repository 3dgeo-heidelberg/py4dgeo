#include <py4dgeo/compute.hpp>

#include <py4dgeo/kdtree.hpp>
#include <py4dgeo/octree.hpp>
#include <py4dgeo/openmp.hpp>
#include <py4dgeo/py4dgeo.hpp>
#include <py4dgeo/searchtree.hpp>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

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
  const Eigen::Vector3d orientation_vector = orientation.row(0).transpose();

  auto radius_search = get_radius_search_function(epoch, normal_radii);

  // Instantiate a container for the first thrown exception in
  // the following parallel region.
  CallbackExceptionVault vault;
#ifdef PY4DGEO_WITH_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (IndexType i = 0; i < corepoints.rows(); ++i) {
    vault.run([&]() {
      double highest_planarity = 0.0;
      Eigen::Matrix3d cov;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver{};
      RadiusSearchResult points;
      for (std::size_t r = 0; r < normal_radii.size(); ++r) {

        radius_search(corepoints.row(i), r, points);

        EigenPointCloud subset = epoch.cloud(points, Eigen::indexing::all);

        // Calculate covariance matrix
        const Eigen::Vector3d mean = subset.colwise().mean();
        subset.rowwise() -= mean.transpose();
        // only need the lower-triangular elements of the covariance matrix
        cov.diagonal() = subset.colwise().squaredNorm();
        cov(1, 0) = subset.col(0).dot(subset.col(1));
        cov(2, 0) = subset.col(0).dot(subset.col(2));
        cov(2, 1) = subset.col(1).dot(subset.col(2));
        cov /= double(subset.rows() - 1);

        // Calculate Eigen vectors
        solver.computeDirect(cov);
        const Eigen::Vector3d& evalues = solver.eigenvalues();
        const Eigen::Vector3d evec = solver.eigenvectors().col(0);

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

  NearestNeighborsDistanceResult result;
  epoch.kdtree.nearest_neighbors_with_distances(transformated_pc, result, 1);
  std::vector<double> p2pdist(transformated_pc.rows());

#ifdef PY4DGEO_WITH_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (IndexType i = 0; i < transformated_pc.rows(); ++i) {
    if (epoch.cloud.rows() != check_size) {
      EigenPointCloud subset = corepoints[result[i].first[0]];

      // Calculate covariance matrix
      Eigen::Matrix3d cov;
      const Eigen::Vector3d mean = subset.colwise().mean();
      subset.rowwise() -= mean.transpose();
      // only need the lower-triangular elements of the covariance matrix
      cov.diagonal() = subset.colwise().squaredNorm();
      cov(1, 0) = subset.col(0).dot(subset.col(1));
      cov(2, 0) = subset.col(0).dot(subset.col(2));
      cov(2, 1) = subset.col(1).dot(subset.col(2));
      cov /= double(subset.rows() - 1);

      // Calculate eigenvectors using direct 3x3 solver
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
      solver.computeDirect(cov);

      // Calculate Eigen vectors
      Eigen::Vector3d normal_vector = solver.eigenvectors().col(0);
      // Calculate cor distance
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
