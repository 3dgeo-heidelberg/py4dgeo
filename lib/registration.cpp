#include <py4dgeo/registration.hpp>

namespace py4dgeo {

void
transform_pointcloud_inplace(EigenPointCloudRef cloud,
                             const Transformation& trafo,
                             EigenPointCloudConstRef reduction_point)
{
  cloud.transpose() =
    (trafo.linear() * (cloud.rowwise() - reduction_point.row(0)).transpose())
      .colwise() +
    (trafo.translation() + reduction_point.row(0).transpose());
}

} // namespace py4dgeo
