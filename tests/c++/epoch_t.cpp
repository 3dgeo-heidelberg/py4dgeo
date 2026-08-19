#include "testsetup.hpp"
#include <py4dgeo/epoch.hpp>

#include <catch2/catch_test_macros.hpp>

#include <sstream>

using namespace py4dgeo;

TEST_CASE("Epoch is working correctly", "[epoch]")
{
  // Instantiate a test epoch
  auto [cloud, corepoints] = testcloud();
  Epoch epoch(*cloud);

  SECTION("Serialize + deserialize")
  {
    std::stringstream buf;
    epoch.to_stream(buf);
    auto deserialized = Epoch::from_stream(buf);

    REQUIRE(epoch.cloud.rows() == deserialized->cloud.rows());
  }
}
