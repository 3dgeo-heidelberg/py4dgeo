#pragma once

#include <Eigen/Eigen>

#include <memory>
#include <vector>

#include "nanoflann.hpp"
namespace py4dgeo {

// The types we use for Point Clouds on the C++ side
using EigenPointCloud =
  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using EigenPointCloudRef = Eigen::Ref<EigenPointCloud>;
using IndexType = Eigen::Index;

class KDTree
{
private:
  // An adaptor between Eigen and our NanoFLANN data structure
  struct Adaptor
  {
    EigenPointCloudRef cloud;
    inline std::size_t kdtree_get_point_count() const;
    inline double kdtree_get_pt(const IndexType, const IndexType) const;
    template<class BBOX>
    bool kdtree_get_bbox(BBOX&) const;
  };

  // The NanoFLANN internal type we are using
  using KDTreeImpl = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, Adaptor>,
    Adaptor,
    3,
    IndexType>;

  // Private constructor - it is used through below static factory function
  // which is more suitable for construction through Python bindings
  KDTree(const EigenPointCloudRef&);

public:
  // Building the KDTree structure given a leaf threshold parameter
  void build_tree(int);

  // Perform a radius search
  std::size_t radius_search(const double*,
                            const double&,
                            std::vector<std::pair<IndexType, double>>&) const;

  // Static factory function
  static KDTree create(const EigenPointCloudRef&);

private:
  Adaptor _adaptor;
  std::shared_ptr<KDTreeImpl> _search;
};

class CachedKDTree
{
  // private constructor
  CachedKDTree(const EigenPointCloudRef&, const EigenPointCloudRef&, double);

public:
  void build_tree(int);
  std::size_t fixed_radius_search(
    const IndexType&,
    const double&,
    std::vector<std::pair<IndexType, double>>&) const;
  static CachedKDTree create(const EigenPointCloudRef&,
                             const EigenPointCloudRef&,
                             double);

private:
  KDTree kdtree;
  EigenPointCloudRef querypoints;
  double maxradius;
  std::vector<std::vector<std::pair<IndexType, double>>> results;
};

// Compute interfaces
void
compute_multiscale_directions(const EigenPointCloudRef&,
                              const std::vector<double>&,
                              const KDTree&,
                              EigenPointCloudRef&);

} // namespace py4dgeo
