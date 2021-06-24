#include "py4dgeo/py4dgeo.hpp"

#include <pcl/search/brute_force.h>
#include <pcl/search/flann_search.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/octree.h>

#include <iostream>

namespace py4dgeo {

PCLPointCloud::PCLPointCloud(const float* ptr, std::size_t n)
  : _cloud(new pcl::PointCloud<pcl::PointXYZ>())
{
  _cloud->resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    pcl::PointXYZ point(ptr[3 * i], ptr[3 * i + 1], ptr[3 * i + 2]);
    (*_cloud)[i] = point;
  }
}

void
PCLPointCloud::build_tree(SearchStrategy strategy)
{
  // We can expose a multitude of search algorithms here
  // https://pointclouds.org/documentation/group__search.html
  if (strategy == SearchStrategy::kdtree)
    _search.reset(new pcl::search::KdTree<pcl::PointXYZ>());
  if (strategy == SearchStrategy::octree)
    _search.reset(new pcl::search::Octree<pcl::PointXYZ>(128.0));
  if (strategy == SearchStrategy::bruteforce)
    _search.reset(new pcl::search::BruteForce<pcl::PointXYZ>());
  _search->setInputCloud(_cloud);
}

int
PCLPointCloud::radius_search(const pcl::PointXYZ& p,
                             double r,
                             std::vector<int>& indices,
                             std::vector<float>& dist)
{
  return _search->radiusSearch(p, r, indices, dist);
}

} // namespace py4dgeo