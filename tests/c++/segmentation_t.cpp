#include "catch2/catch.hpp"
#include "py4dgeo/segmentation.hpp"
#include "testsetup.hpp"

using namespace py4dgeo;

TEST_CASE("DTW distance calculation", "[segmentation]")
{
  // Wikipedia test case: https://de.wikipedia.org/wiki/Dynamic-Time-Warping
  EigenSpatiotemporalArray arr(2, 4);
  arr << 1, 5, 4, 2, 1, 2, 4, 1;

  auto dist = dtw_distance(arr.row(0), arr.row(1));

  REQUIRE(dist > 0);
  REQUIRE(std::abs(dist - 3) < 1e-8);
}