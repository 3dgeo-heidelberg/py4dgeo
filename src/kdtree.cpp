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

} // namespace py4dgeo