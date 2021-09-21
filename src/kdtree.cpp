#include "py4dgeo/py4dgeo.hpp"

#include <iostream>

namespace py4dgeo {

KDTree::KDTree(double* ptr, std::size_t n)
  : _adaptor{ ptr, n }
{}

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
                      std::vector<std::pair<std::size_t, double>>& result) const
{
  nanoflann::SearchParams params;
  return _search->radiusSearch(query, radius * radius, result, params);
}

} // namespace py4dgeo