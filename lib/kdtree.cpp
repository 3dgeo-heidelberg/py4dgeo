#ifdef PY4DGEO_WITH_OPENMP
#include <omp.h>
#endif

#include "py4dgeo/kdtree.hpp"
#include "py4dgeo/py4dgeo.hpp"

#include <vector>

namespace py4dgeo {

KDTree::KDTree(const EigenPointCloudRef& cloud)
  : adaptor{ cloud }
{
}

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

std::ostream&
KDTree::saveIndex(std::ostream& stream) const
{
  stream.write(reinterpret_cast<const char*>(&leaf_parameter), sizeof(int));

  if (leaf_parameter != 0)
    search->saveIndex(stream);

  return stream;
}

std::istream&
KDTree::loadIndex(std::istream& stream)
{
  // Read the leaf parameter
  stream.read(reinterpret_cast<char*>(&leaf_parameter), sizeof(int));

  if (leaf_parameter != 0) {
    search = std::make_shared<KDTree::KDTreeImpl>(
      3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_parameter));
    search->loadIndex(stream);
  }

  return stream;
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

void
KDTree::nearest_neighbors_with_distances(EigenPointCloudConstRef cloud,
                                         NearestNeighborsDistanceResult& result,
                                         int k) const
{
  result.resize(cloud.rows());
  nanoflann::SearchParams params;

#ifdef PY4DGEO_WITH_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (IndexType i = 0; i < cloud.rows(); ++i) {
    std::pair<std::vector<long unsigned int>, std::vector<double>> pointResult;

    std::vector<long unsigned int>& ret_indices = pointResult.first;
    std::vector<double>& out_dists_sqr = pointResult.second;
    ret_indices.resize(k);
    out_dists_sqr.resize(k);

    nanoflann::KNNResultSet<double, size_t, size_t> resultset(k);
    auto qp = cloud.row(i).eval();
    resultset.init(ret_indices.data(), out_dists_sqr.data());
    search->findNeighbors(resultset, &(qp(0, 0)), params);
    result[i] = pointResult;
  }
}

int
KDTree::get_leaf_parameter() const
{
  return leaf_parameter;
}

} // namespace py4dgeo
