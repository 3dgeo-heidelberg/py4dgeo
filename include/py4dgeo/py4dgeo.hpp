#pragma once

#include <Eigen/Eigen>

#include <memory>
#include <utility>
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
public:
  // The Types used for the results
  using RadiusSearchResult = std::vector<IndexType>;
  using RadiusSearchDistanceResult = std::vector<std::pair<IndexType, double>>;

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

  struct NoDistancesReturnSet
  {
    double radius;
    RadiusSearchResult& indices;

    inline std::size_t size() const;
    inline bool full() const;
    inline bool addPoint(double, IndexType);
    inline double worstDist() const;
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
  // Static factory function. These serve as de facto constructors, but they can
  // are much easier exposed in Python bindings than actual constructors.
  static KDTree create(const EigenPointCloudRef&);

  // Building the KDTree structure given a leaf threshold parameter
  void build_tree(int);

  // Precompute on a number of query points with a maximal radius
  void precompute(const EigenPointCloudRef&, double);

  // Normal radius search at an arbitrary query point
  std::size_t radius_search(const double*, double, RadiusSearchResult&) const;

  // A normal radius search that also returns the squared distances. The entries
  // are always sorted according to distance.
  std::size_t radius_search_with_distances(const double*,
                                           double,
                                           RadiusSearchDistanceResult&) const;

  // Radius search around a query point from the precomputation set
  std::size_t precomputed_radius_search(const IndexType,
                                        double,
                                        RadiusSearchResult&) const;

private:
  Adaptor adaptor;
  std::unique_ptr<KDTreeImpl> search;
  std::vector<std::vector<IndexType>> precomputed_indices;
  std::vector<std::vector<double>> precomputed_distances;
};

// Compute interfaces
void
compute_multiscale_directions(const EigenPointCloudRef&,
                              const std::vector<double>&,
                              const KDTree&,
                              EigenPointCloudRef&);

} // namespace py4dgeo
