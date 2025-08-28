#include "testsetup.hpp"
#include <py4dgeo/compute.hpp>
#include <py4dgeo/epoch.hpp>
#include <py4dgeo/kdtree.hpp>
#include <py4dgeo/py4dgeo.hpp>
#include <py4dgeo/searchtree.hpp>

#include <Eigen/Core>
#include <mimalloc.h>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

using namespace py4dgeo;

void
benchmark(std::shared_ptr<EigenPointCloud> cloud,
          SearchTree searchtree = SearchTree::KDTree)
{
  Epoch epoch(*cloud);
  epoch.set_default_radius_search_tree(searchtree);
  epoch.kdtree.build_tree(10);
  epoch.octree.build_tree();

  const std::vector<double> normal_radii{ 1.0 };
  std::vector<double> used_radii;
  EigenNormalSet directions(cloud->rows(), 3);
  EigenNormalSet orientation(1, 3);
  orientation << 0, 0, 1;

  auto start = std::chrono::high_resolution_clock::now();
  compute_multiscale_directions(
    epoch, *cloud, normal_radii, orientation, directions, used_radii);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "compute_multiscale_directions executed in " << duration.count()
            << " seconds.\n";
  std::cout << directions.rows() << " normals computed.\n";
}

int
main(int argc, char** argv)
{
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path-to-pointcloud.xyz>\n";
    return 1;
  }

  std::cout << "mi_version: " << mi_version() << std::endl;
  void* p = mi_malloc(100);
  mi_free(p);

  // Load point cloud
  std::string filename = argv[1];
  auto cloud = benchcloud_from_file(filename);

  // Change default searchtree (default = KDTREE)
  SearchTree searchtree = SearchTree::KDTree;

  // Create epoch and run benchmark
  benchmark(cloud, searchtree);

  return 0;
}
