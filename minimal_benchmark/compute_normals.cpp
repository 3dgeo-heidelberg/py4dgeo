#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <vector>

#include "testsetup.hpp"
#include <py4dgeo/compute.hpp>
#include <py4dgeo/epoch.hpp>
#include <py4dgeo/kdtree.hpp>

using namespace py4dgeo;

void
benchmark(std::shared_ptr<EigenPointCloud> cloud)
{
  Epoch epoch(*cloud);
  epoch.kdtree.build_tree(10);
  std::vector<double> normal_radii{ 1.0 };
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

  std::string filename = argv[1];

  auto cloud = benchcloud_from_file(filename);
  auto cloud_old = benchcloud_from_file_old(filename);
  benchmark(cloud_old);
  benchmark(cloud);

  return 0;
}
