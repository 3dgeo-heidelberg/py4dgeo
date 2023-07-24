#include "catch2/catch.hpp"
#include "py4dgeo/registration.hpp"
#include "testsetup.hpp"

using namespace py4dgeo;

TEST_CASE("Affine Transformation", "[compute]")
{
  auto [cloud1, corepoints] = testcloud();
  auto [cloud2, corepoints2] = testcloud();

  // Define a transformation
  Transformation t(Transformation::Identity());
  t(0, 3) = 1;

  EigenPointCloud ref(1, 3);
  ref << 1, 2, 3;

  // Apply the transformation
  transform_pointcloud_inplace(*cloud1, t, ref);

  for (IndexType i = 0; i < cloud1->rows(); ++i) {
    if (std::abs((*cloud1)(i, 0) - (*cloud2)(i, 0) - 1.0) >= 1e-8) {
      CAPTURE((*cloud1)(i, 0));
      CAPTURE((*cloud2)(i, 0));
    }
    REQUIRE(std::abs((*cloud1)(i, 0) - (*cloud2)(i, 0) - 1.0) < 1e-8);
    REQUIRE(std::abs((*cloud1)(i, 1) - (*cloud2)(i, 1)) < 1e-8);
    REQUIRE(std::abs((*cloud1)(i, 2) - (*cloud2)(i, 2)) < 1e-8);
  }
}
