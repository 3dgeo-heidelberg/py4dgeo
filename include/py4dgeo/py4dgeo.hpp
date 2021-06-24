#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/search.h>

#include <vector>

namespace py4dgeo {

enum class SearchStrategy
{
  kdtree,
  octree,
  bruteforce
};

class PCLPointCloud
{
public:
  PCLPointCloud(const float* ptr, std::size_t);
  void build_tree(SearchStrategy strategy);
  int radius_search(const pcl::PointXYZ&,
                    double,
                    std::vector<int>&,
                    std::vector<float>&);

private:
  pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud;
  pcl::search::Search<pcl::PointXYZ>::Ptr _search;
};

} // namespace py4dgeo
