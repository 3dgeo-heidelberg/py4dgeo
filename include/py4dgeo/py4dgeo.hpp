#pragma once

#include <memory>
#include <vector>

#include "nanoflann.hpp"
namespace py4dgeo {

namespace impl {

struct NanoFLANNAdaptor
{
  inline std::size_t kdtree_get_point_count() const { return n; }

  inline double kdtree_get_pt(const size_t idx, const size_t dim) const
  {
    return ptr[3 * idx + dim];
  }

  template<class BBOX>
  bool kdtree_get_bbox(BBOX&) const
  {
    return false;
  }

  double* ptr;
  std::size_t n;
};

}

class KDTree
{
private:
  using KDTreeImpl = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, impl::NanoFLANNAdaptor>,
    impl::NanoFLANNAdaptor,
    3>;

public:
  KDTree(double* ptr, std::size_t);
  void build_tree(int);

  std::size_t radius_search(const double*,
                            const double&,
                            std::vector<std::pair<std::size_t, double>>&) const;

private:
  impl::NanoFLANNAdaptor _adaptor;
  std::shared_ptr<KDTreeImpl> _search;
};

} // namespace py4dgeo
