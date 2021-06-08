#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <string>

namespace geolib4d {

class PointCloud : public pcl::PointCloud<pcl::PointXYZ> {
public:
  void from_file(const std::string &);
};

} // namespace geolib4d
