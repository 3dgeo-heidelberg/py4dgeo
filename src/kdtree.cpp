#include "py4dgeo/py4dgeo.hpp"

#include <iostream>

namespace py4dgeo {

inline std::size_t
KDTree::Adaptor::kdtree_get_point_count() const
{
  return cloud.rows();
}

inline double
KDTree::Adaptor::kdtree_get_pt(const IndexType idx, const IndexType dim) const
{
  return cloud(idx, dim);
}

template<class BBOX>
bool
KDTree::Adaptor::kdtree_get_bbox(BBOX&) const
{
  return false;
}

KDTree::KDTree(const EigenPointCloudRef& cloud)
  : _adaptor{ cloud }
{}

KDTree
KDTree::create(const EigenPointCloudRef& cloud)
{
  return KDTree(cloud);
}

void
KDTree::build_tree(int leaf)
{
  _search = std::make_shared<KDTreeImpl>(
    3, _adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(leaf));
  _search->buildIndex();
}

std::size_t
KDTree::radius_search(const double* query,
                      const double& radius,
                      std::vector<std::pair<IndexType, double>>& result) const
{
  nanoflann::SearchParams params;
  return _search->radiusSearch(query, radius * radius, result, params);
}

CachedKDTree::CachedKDTree(const EigenPointCloudRef& cloud,
                           const EigenPointCloudRef& querypoints,
                           double maxradius)
  : kdtree(KDTree::create(cloud))
  , querypoints(querypoints)
  , maxradius(maxradius)
  , results(querypoints.rows())
{}

void
CachedKDTree::build_tree(int leaf)
{
  // Build the original KDTree
  kdtree.build_tree(leaf);

  // Do the evaluation
  for (IndexType i = 0; i < querypoints.rows(); ++i)
    kdtree.radius_search(&querypoints(i, 0), maxradius, results[i]);
}

std::size_t
CachedKDTree::fixed_radius_search(
  const IndexType& core_idx,
  const double& radius,
  std::vector<std::pair<IndexType, double>>& result) const
{
  result.clear();

  auto it =
    std::find_if(results[core_idx].begin(),
                 results[core_idx].end(),
                 [radius](auto p) { return p.second > radius * radius; });

  std::copy(results[core_idx].begin(), it, std::back_inserter(result));
  return result.size();
}

CachedKDTree
CachedKDTree::create(const EigenPointCloudRef& cloud,
                     const EigenPointCloudRef& querypoints,
                     double maxradius)
{
  return CachedKDTree(cloud, querypoints, maxradius);
}

} // namespace py4dgeo