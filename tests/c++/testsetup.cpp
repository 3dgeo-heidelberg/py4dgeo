#include "testsetup.hpp"

#include <fstream>
#include <limits>
#include <sstream>
#include <vector>

#include <iomanip>
#include <iostream>

using namespace py4dgeo;

std::shared_ptr<EigenPointCloud>
benchcloud_from_file(const std::string& filename)
{

  std::ifstream stream;

  stream.open(filename);
  if (stream.fail()) {
    std::cerr << filename
              << " Was not successfully opened. Please check that the file "
                 "currently exists. "
              << std::endl;
    exit(1);
  }
  std::string line;

  // Read the file once to determine size and lower left corner
  // Definitely not the best way to read this, but good enough.
  std::array<double, 3> mincoord{ std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity() };
  std::size_t points{ 0 };
  while (std::getline(stream, line)) {
    std::istringstream s(line);
    double x;
    for (int i = 0; i < 3; ++i) {
      s >> x;
      if (x < mincoord[i])
        mincoord[i] = x;
    }
    ++points;
  }
  stream.close();

  // Now read and shift the actual data
  auto cloud = std::make_shared<EigenPointCloud>(points, 3);
  stream.open(filename);
  points = 0;
  while (std::getline(stream, line)) {
    std::istringstream s(line);
    double x;
    for (int i = 0; i < 3; ++i) {
      s >> x;
      (*cloud)(points, i) = x - mincoord[i];
    }
    ++points;
  }

  return cloud;
}

std::shared_ptr<EigenPointCloud>
slice_cloud(EigenPointCloudConstRef cloud, int sampling_factor)
{
  auto sliced =
    std::make_shared<EigenPointCloud>(cloud.rows() / sampling_factor, 3);
  for (IndexType i = 0; i < cloud.rows() / sampling_factor; ++i)
    (*sliced)(i, Eigen::all) = cloud(i * sampling_factor, Eigen::all);
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
