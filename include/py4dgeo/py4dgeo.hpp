#pragma once

#include <memory>
#include <vector>

#include "nanoflann.hpp"
namespace py4dgeo {

namespace impl {

struct NanoFLANNAdaptor
{
  // Constructors and destructors
  NanoFLANNAdaptor(std::size_t n)
    : ownership(true)
    , ptr(new double[n * 3])
    , n(n)
  {}

  NanoFLANNAdaptor(double* ptr, std::size_t n)
    : ownership(false)
    , ptr(ptr)
    , n(n)
  {}

  ~NanoFLANNAdaptor()
  {
    if (ownership)
      delete[] ptr;
  }

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

  bool ownership;
  double* ptr;
  std::size_t n;
};

}

class KDTree
{
public:
  using KDTreeImpl = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, impl::NanoFLANNAdaptor>,
    impl::NanoFLANNAdaptor,
    3>;

  KDTree(double* ptr, std::size_t);
  KDTree(std::size_t);
  void build_tree(int);

  std::size_t radius_search(const double*,
                            const double&,
                            std::vector<std::pair<std::size_t, double>>&) const;

  impl::NanoFLANNAdaptor _adaptor;
  std::shared_ptr<KDTreeImpl> _search;
};

} // namespace py4dgeo
