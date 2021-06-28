#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/search.h>

#include <Eigen/Dense>

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

struct NFPointCloud
{
  using Matrix = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 3>>;
  using KDTree =
    nanoflann::KDTreeEigenMatrixAdaptor<Matrix, 3, nanoflann::metric_L2_Simple>;

  NFPointCloud(float* ptr, std::size_t);
  void build_tree();

  std::size_t radius_search(const float*,
                            const double&,
                            std::vector<std::pair<KDTree::IndexType, float>>&);

  Matrix _cloud;
  std::shared_ptr<KDTree> _search;
};

} // namespace py4dgeo
