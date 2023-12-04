#ifdef PY4DGEO_WITH_OPENMP
#include <omp.h>
#endif

#include "py4dgeo/kdtree.hpp"
#include "py4dgeo/py4dgeo.hpp"

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
KDTree::nearest_neighbors_with_distances(
  EigenPointCloudConstRef cloud,
  NearestNeighborsDistanceResult& result) const
{
  // Resize the results container
  result.first.resize(cloud.rows());
  result.second.resize(cloud.rows());
  nanoflann::SearchParams params;

#ifdef PY4DGEO_WITH_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (IndexType i = 0; i < cloud.rows(); ++i) {
    nanoflann::KNNResultSet<double, IndexType> resultset(1);
    resultset.init(result.first.data() + i, result.second.data() + i);
    auto qp = cloud.row(i).eval();
    search->findNeighbors(resultset, &(qp(0, 0)), params);
  }
}

int
KDTree::get_leaf_parameter() const
{
  return leaf_parameter;
}

} // namespace py4dgeo
