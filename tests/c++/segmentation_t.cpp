#include "catch2/catch.hpp"
#include "py4dgeo/segmentation.hpp"
#include "testsetup.hpp"

#include <limits>

using namespace py4dgeo;

TEST_CASE("DTW distance calculation", "[segmentation]")
{
  // Wikipedia test case: https://de.wikipedia.org/wiki/Dynamic-Time-Warping
  EigenSpatiotemporalArray arr(2, 4);
  arr << 1, 5, 4, 2, 1, 2, 4, 1;

  TimeseriesDistanceFunctionData data{ arr.row(0), arr.row(1), 0.0, 0.0 };
  auto dist = dtw_distance(data);

  REQUIRE(dist > 0);
  REQUIRE(std::abs(dist - 3) < 1e-8);
}

TEST_CASE("DTW distance with NaN Values", "[segmentation]")
{
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  EigenSpatiotemporalArray arr(2, 6);
  arr << nan, 1, 42, 5, 4, 2, 42, 1, nan, 2, 4, 1;
  TimeseriesDistanceFunctionData data{ arr.row(0), arr.row(1), 0.0, 0.0 };
  auto dist = dtw_distance(data);
  REQUIRE(std::abs(dist - 3) < 1e-8);
}

TEST_CASE("DTW distance with all NaN Values", "[segmentation]")
{
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  EigenSpatiotemporalArray arr(2, 2);
  arr << nan, nan, nan, nan;
  TimeseriesDistanceFunctionData data{ arr.row(0), arr.row(1), 0.0, 0.0 };
  auto dist = dtw_distance(data);
  REQUIRE(std::isnan(dist));
}

TEST_CASE("Normalized DTW Distances", "[segmentation]")
{
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  EigenSpatiotemporalArray arr(2, 6);
  arr << nan, 1, 42, 5, 4, 2, 42, 1, nan, 2, 4, 1;
  TimeseriesDistanceFunctionData data{ arr.row(0), arr.row(1), 0.0, 0.0 };
  auto dist = normalized_dtw_distance(data);
  REQUIRE(dist >= 0.0);
  REQUIRE(dist <= 1.0);
}
