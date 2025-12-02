#include "testsetup.hpp"
#include <py4dgeo/epoch.hpp>
#include <py4dgeo/kdtree.hpp>
#include <py4dgeo/octree.hpp>
#include <py4dgeo/py4dgeo.hpp>
#include <py4dgeo/searchtree.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>

using namespace py4dgeo;

TEST_CASE("KDTree and Octree are correctly built")
{
  // Get a test epoch
  auto [cloud, corepoints] = testcloud();
  Epoch epoch(*cloud);

  // Construct the KDTree
  auto kdtree = KDTree::create(epoch.cloud);
  kdtree.build_tree(10);

  // Construct the Octree
  auto octree = Octree::create(epoch.cloud);
  octree.build_tree();

  std::array<IndexType, 4> query_point_indices = {
    0, cloud->rows() / 3, 2 * cloud->rows() / 3, cloud->rows() - 1
  };

  std::array<double, 3> radii{ 1.5, 2.0, 5.0 };

  for (const IndexType idx : query_point_indices) {
    const Eigen::Vector3d& point = epoch.cloud.row(idx);
    for (double radius : radii) {
      std::string tag = "Query at (" + std::to_string(point.x()) + ", " +
                        std::to_string(point.y()) + ", " +
                        std::to_string(point.z()) +
                        "), radius = " + std::to_string(radius);

      SECTION("Index-only " + tag)
      {
        RadiusSearchResult kd_result, oct_result;

        kdtree.radius_search(point.data(), radius, kd_result);

        unsigned int level =
          epoch.octree.find_appropriate_level_for_radius_search(radius);
        octree.radius_search(point, radius, level, oct_result);

        std::sort(kd_result.begin(), kd_result.end());
        std::sort(oct_result.begin(), oct_result.end());

        REQUIRE(kd_result == oct_result);
      }

      SECTION("With distances " + tag)
      {
        RadiusSearchDistanceResult kd_result, oct_result;

        kdtree.radius_search_with_distances(point.data(), radius, kd_result);

        unsigned int level =
          epoch.octree.find_appropriate_level_for_radius_search(radius);
        octree.radius_search_with_distances(point, radius, level, oct_result);

        REQUIRE(kd_result.size() == oct_result.size());

        for (std::size_t i = 0; i < kd_result.size(); ++i) {
          REQUIRE(kd_result[i].first == oct_result[i].first);
          REQUIRE_THAT(kd_result[i].second,
                       Catch::Matchers::WithinAbs(oct_result[i].second, 1e-7));
        }
      }
    }
  }
}
