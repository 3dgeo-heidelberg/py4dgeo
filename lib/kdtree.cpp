#ifdef PY4DGEO_WITH_OPENMP
#include <omp.h>
#endif

#include "py4dgeo/kdtree.hpp"
#include "py4dgeo/py4dgeo.hpp"

namespace py4dgeo {

KDTree::KDTree(const EigenPointCloudRef& cloud)
  : adaptor{ cloud }
{}

KDTree
KDTree::create(const EigenPointCloudRef& cloud)
{
  return KDTree(cloud);
}

void
KDTree::build_tree(int leaf)
{
  search = std::make_shared<KDTreeImpl>(
    3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(leaf));
  search->buildIndex();
  leaf_parameter = leaf;
}

void
KDTree::precompute(EigenPointCloudRef querypoints,
                   double maxradius,
                   MemoryPolicy policy)
{
  precomputed_querypoints = querypoints;
  precomputed_policy = policy;
  if (policy < MemoryPolicy::COREPOINTS)
    return;

  // Resize the output data structures
  precomputed_indices.resize(querypoints.rows());
  precomputed_distances.resize(querypoints.rows());

  // Loop over query points and evaluate with maxradius
#ifdef PY4DGEO_WITH_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (IndexType i = 0; i < querypoints.rows(); ++i) {
    RadiusSearchDistanceResult result;
    radius_search_with_distances(&querypoints(i, 0), maxradius, result);

    precomputed_indices[i].resize(result.size());
    precomputed_distances[i].resize(result.size());

    for (std::size_t j = 0; j < result.size(); ++j) {
      precomputed_indices[i][j] = result[j].first;
      precomputed_distances[i][j] = result[j].second;
    }
  }
}

std::size_t
KDTree::radius_search(const double* query,
                      double radius,
                      RadiusSearchResult& result) const
{
  NoDistancesReturnSet set{ radius * radius, result };
  nanoflann::SearchParams params;
  params.sorted = false;
  return search->radiusSearchCustomCallback(query, set, params);
}

std::size_t
KDTree::radius_search_with_distances(const double* query,
                                     double radius,
                                     RadiusSearchDistanceResult& result) const
{
  nanoflann::SearchParams params;
  return search->radiusSearch(query, radius * radius, result, params);
}

std::size_t
KDTree::precomputed_radius_search(const IndexType idx,
                                  double radius,
                                  RadiusSearchResult& result) const
{
  // Check whether precomputation was no-op
  if (precomputed_policy < MemoryPolicy::COREPOINTS)
    return radius_search(&precomputed_querypoints(idx, 0), radius, result);

  // Access our precomputation
  result.clear();
  auto it = std::find_if(precomputed_distances[idx].begin(),
                         precomputed_distances[idx].end(),
                         [radius](auto d) { return d > radius * radius; });

  std::copy(precomputed_indices[idx].begin(),
            precomputed_indices[idx].begin() +
              (it - precomputed_distances[idx].begin()),
            std::back_inserter(result));
  return result.size();
}

} // namespace py4dgeo