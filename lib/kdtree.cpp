#ifdef PY4DGEO_WITH_OPENMP
#include <omp.h>
#endif

#include <py4dgeo/kdtree.hpp>
#include <py4dgeo/py4dgeo.hpp>

#include <cstddef>
#include <istream>
#include <memory>
#include <ostream>
#include <utility>
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

void
KDTree::invalidate()
{
  search = nullptr;
  leaf_parameter = 0;
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
  nanoflann::SearchParameters params;
  params.sorted = false;
  return search->radiusSearchCustomCallback(query, set, params);
}

std::size_t
KDTree::radius_search_with_distances(const double* query,
                                     double radius,
                                     RadiusSearchDistanceResult& result) const
{
  WithDistancesReturnSet set{ radius * radius, result };
  nanoflann::SearchParameters params;
  return search->radiusSearchCustomCallback(query, set, params);
}

void
KDTree::nearest_neighbors_with_distances(EigenPointCloudConstRef cloud,
                                         NearestNeighborsDistanceResult& result,
                                         int k) const
{
  result.resize(cloud.rows());
  nanoflann::SearchParameters params;

#ifdef PY4DGEO_WITH_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (IndexType i = 0; i < cloud.rows(); ++i) {
    std::pair<std::vector<IndexType>, std::vector<double>> pointResult;

    std::vector<IndexType>& ret_indices = pointResult.first;
    std::vector<double>& out_dists_sqr = pointResult.second;
    ret_indices.resize(k);
    out_dists_sqr.resize(k);

    nanoflann::KNNResultSet<double, IndexType> resultset(k);
    auto qp = cloud.row(i).eval();
    resultset.init(ret_indices.data(), out_dists_sqr.data());
    search->findNeighbors(resultset, &(qp(0, 0)), params);
    result[i] = pointResult;
  }
}

void
KDTree::nearest_neighbors(EigenPointCloudConstRef cloud,
                          NearestNeighborsResult& result,
                          int k) const
{
  result.resize(cloud.rows());
  nanoflann::SearchParameters params;

#ifdef PY4DGEO_WITH_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (IndexType i = 0; i < cloud.rows(); ++i) {
    std::vector<IndexType> pointResult;
    std::vector<double> dis_skip;

    std::vector<IndexType>& ret_indices = pointResult;
    std::vector<double>& out_dists_sqr = dis_skip;
    ret_indices.resize(k);
    out_dists_sqr.resize(k);

    nanoflann::KNNResultSet<double, IndexType> resultset(k);
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
