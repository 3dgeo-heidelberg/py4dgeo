#include "testsetup.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace py4dgeo;

std::shared_ptr<EigenPointCloud>
benchcloud_from_file(const std::string& filename)
{
  std::ifstream stream(filename);
  if (!stream) {
    std::cerr << "Was not successfully opened. Please check that the file "
                 "currently exists: "
              << filename << std::endl;
    std::exit(1);
  }

  std::vector<Eigen::Vector3d> points;
  Eigen::Vector3d mincoord =
    Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());

  std::string line;
  while (std::getline(stream, line)) {
    std::istringstream s(line);
    Eigen::Vector3d point;
    s >> point[0] >> point[1] >> point[2];

    if (!s)
      continue;

    mincoord = mincoord.cwiseMin(point);
    points.push_back(point);
  }

  auto cloud = std::make_shared<EigenPointCloud>(points.size(), 3);
  for (std::size_t i = 0; i < points.size(); ++i) {
    (*cloud).row(i) = points[i] - mincoord;
  }

  return cloud;
}

std::shared_ptr<EigenPointCloud>
slice_cloud(EigenPointCloudConstRef cloud, int sampling_factor)
{
  auto sliced =
    std::make_shared<EigenPointCloud>(cloud.rows() / sampling_factor, 3);
  for (IndexType i = 0; i < cloud.rows() / sampling_factor; ++i)
    (*sliced)(i, Eigen::indexing::all) =
      cloud(i * sampling_factor, Eigen::indexing::all);
  return sliced;
}

std::pair<std::shared_ptr<EigenPointCloud>, std::shared_ptr<EigenPointCloud>>
ahk_benchcloud()
{
  // auto cloud = benchcloud_from_file(DATAPATH(ahk_2017_small.xyz));
  auto cloud = benchcloud_from_file(DATAPATH(plane_horizontal_t1.xyz));
  return std::make_pair(cloud, slice_cloud(*cloud, 100));
}

std::pair<std::shared_ptr<EigenPointCloud>, std::shared_ptr<EigenPointCloud>>
testcloud()
{
  auto cloud = benchcloud_from_file(DATAPATH(plane_horizontal_t1.xyz));
  return std::make_pair(cloud, cloud);
}

std::pair<std::shared_ptr<EigenPointCloud>, std::shared_ptr<EigenPointCloud>>
testcloud_dif_files()
{
  auto cloud1 = benchcloud_from_file(DATAPATH(plane_horizontal_t1.xyz));
  auto cloud2 = benchcloud_from_file(DATAPATH(plane_horizontal_t2.xyz));
  return std::make_pair(cloud1, cloud2);
}
