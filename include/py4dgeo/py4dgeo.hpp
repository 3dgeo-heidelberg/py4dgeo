#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/search.h>

#include <memory>
#include <vector>

#include "nanoflann.hpp"
namespace py4dgeo {

enum class SearchStrategy
{
  kdtree,
  octree,
  bruteforce
};

struct PCLPointCloud
{
public:
  PCLPointCloud(const float* ptr, std::size_t);
  void build_tree(SearchStrategy strategy);
  int radius_search(const pcl::PointXYZ&,
                    double,
                    std::vector<int>&,
                    std::vector<float>&);

  pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud;
  pcl::search::Search<pcl::PointXYZ>::Ptr _search;
};

struct NFPointCloud2
{
  using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, NFPointCloud2>,
    NFPointCloud2,
    3>;

  NFPointCloud2(float* ptr, std::size_t);
  void build_tree();

  std::size_t radius_search(const float*,
                            const double&,
                            std::vector<std::pair<std::size_t, float>>&);

  inline std::size_t kdtree_get_point_count() const { return n; }

  inline float kdtree_get_pt(const size_t idx, const size_t dim) const
  {
    return data[3 * idx + dim];
  }

  template<class BBOX>
  bool kdtree_get_bbox(BBOX& /* bb */) const
  {
    return false;
  }

  float* data;
  std::size_t n;
  std::shared_ptr<KDTree> _search;
};

} // namespace py4dgeo
