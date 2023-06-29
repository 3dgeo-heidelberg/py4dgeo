#include <py4dgeo/registration.hpp>

namespace py4dgeo {

void
transform_pointcloud_inplace(EigenPointCloudRef cloud,
                             const Transformation& trafo)
{
  cloud.transpose() = trafo * cloud.transpose();
}

} // namespace py4dgeo
