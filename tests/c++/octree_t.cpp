#include "testsetup.hpp"
#include <py4dgeo/epoch.hpp>
#include <py4dgeo/octree.hpp>
#include <py4dgeo/searchtree.hpp>

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

using namespace py4dgeo;

namespace py4dgeo {
struct OctreeTestAccess
{
  static Octree::SpatialKey compute_spatial_key(const Octree& t,
                                                Octree::OctreeCoordinate c,
                                                unsigned int level)
  {
    return t.compute_spatial_key(c, level);
  }

  static Octree::OctreeCoordinate decode_spatial_key_at_level(
    const Octree& t,
    Octree::SpatialKey k,
    unsigned int level)
  {
    return t.decode_spatial_key_at_level(k, level);
  }
};
} // namespace py4dgeo

TEST_CASE("Octree is correctly build", "[octree]")
{
  // Get a test epoch
  auto [cloud, corepoints] = testcloud();
  Epoch epoch(*cloud);

  constexpr double tol = 1.e-10;

  SECTION("Build cubic, no corner")
  {
    // Construct the Octree
    auto tree = Octree::create(epoch.cloud);
    tree.build_tree(true);

    Eigen::Vector3d extent = tree.get_max_point() - tree.get_min_point();

    REQUIRE_THAT(extent.x(), Catch::Matchers::WithinRel(extent.y(), tol));
    REQUIRE_THAT(extent.x(), Catch::Matchers::WithinRel(extent.z(), tol));
    REQUIRE_THAT(extent.y(), Catch::Matchers::WithinRel(extent.z(), tol));
  }
  SECTION("Build cubic, min corner")
  {
    // Construct the Octree
    auto tree = Octree::create(epoch.cloud);
    tree.build_tree(true, cloud->colwise().minCoeff());

    Eigen::Vector3d extent = tree.get_max_point() - tree.get_min_point();

    REQUIRE_THAT(extent.x(), Catch::Matchers::WithinRel(extent.y(), tol));
    REQUIRE_THAT(extent.x(), Catch::Matchers::WithinRel(extent.z(), tol));
    REQUIRE_THAT(extent.y(), Catch::Matchers::WithinRel(extent.z(), tol));
  }
  SECTION("Build cubic, max corner")
  {
    // Construct the Octree
    auto tree = Octree::create(epoch.cloud);
    tree.build_tree(true, std::nullopt, cloud->colwise().maxCoeff());

    Eigen::Vector3d extent = tree.get_max_point() - tree.get_min_point();

    REQUIRE_THAT(extent.x(), Catch::Matchers::WithinRel(extent.y(), tol));
    REQUIRE_THAT(extent.x(), Catch::Matchers::WithinRel(extent.z(), tol));
    REQUIRE_THAT(extent.y(), Catch::Matchers::WithinRel(extent.z(), tol));
  }

  // Construct the Octree
  auto tree = Octree::create(epoch.cloud);
  tree.build_tree();

  // Do radius search with radius wide enough to cover the entire cloud
  constexpr double radius = 100.;

  SECTION("Perform radius search")
  {
    // Find all nodes with a radius search
    Eigen::Vector3d query_point{ 0.0, 0.0, 0.0 };
    RadiusSearchResult result;

    for (unsigned int level = 0; level < 7; ++level) {
      auto num = tree.radius_search(query_point, radius, level, result);
      REQUIRE(num == epoch.cloud.rows());
      REQUIRE(result.size() == epoch.cloud.rows());
    }
  }

  SECTION("Perform radius search with distances")
  {
    // Find all nodes with a radius search
    Eigen::Vector3d query_point{ 0.0, 0.0, 0.0 };
    RadiusSearchDistanceResult result;

    unsigned int level =
      epoch.octree.find_appropriate_level_for_radius_search(radius);
    auto num =
      tree.radius_search_with_distances(query_point, radius, level, result);
    REQUIRE(num == epoch.cloud.rows());
    REQUIRE(result.size() == epoch.cloud.rows());
    REQUIRE(std::is_sorted(result.begin(), result.end(), [](auto a, auto b) {
      return a.second < b.second;
    }));
  }

  SECTION("Spatial key encode/decode round-trip")
  {
    constexpr unsigned int level = 4;
    constexpr unsigned int max_coord = 1u << level;

    for (unsigned int x = 0; x < max_coord; ++x) {
      for (unsigned int y = 0; y < max_coord; ++y) {
        for (unsigned int z = 0; z < max_coord; ++z) {

          Octree::OctreeCoordinate c{ x, y, z };
          Octree::SpatialKey key =
            OctreeTestAccess::compute_spatial_key(tree, c, level);
          Octree::OctreeCoordinate decoded =
            OctreeTestAccess::decode_spatial_key_at_level(tree, key, level);

          REQUIRE(decoded[0] == x);
          REQUIRE(decoded[1] == y);
          REQUIRE(decoded[2] == z);
        }
      }
    }
  }

  SECTION("Cross check get_unique_cells against get_number_of_occupied_cells")
  {
    for (unsigned int level = 0; level <= tree.get_max_depth(); ++level) {
      Octree::KeyContainer unique_cells = tree.get_unique_cells(level);

      // Number of unique occupied cells matches the value stored in the octree
      REQUIRE(unique_cells.size() == tree.get_number_of_occupied_cells(level));
    }
  }

  SECTION("Cross check statistics derived from get_unique_cells and "
          "get_cell_population with pre-computed stats")
  {
    for (unsigned int level = 0; level <= tree.get_max_depth(); ++level) {
      Octree::KeyContainer unique_cells = tree.get_unique_cells(level);
      REQUIRE(unique_cells.size() == tree.get_number_of_occupied_cells(level));

      std::vector<std::size_t> populations =
        tree.get_cell_population(unique_cells, level);

      auto [total_sum, sum2, computed_max_population] =
        std::accumulate(populations.begin(),
                        populations.end(),
                        std::make_tuple(0.0, 0.0, std::size_t(0)),
                        [](auto acc, std::size_t pop) {
                          auto [sum, sum2, max_val] = acc;
                          sum += pop;
                          sum2 += pop * pop;
                          max_val = std::max(max_val, pop);
                          return std::make_tuple(sum, sum2, max_val);
                        });

      double computed_average_population = total_sum / populations.size();
      double computed_std_population =
        std::sqrt(sum2 / populations.size() -
                  computed_average_population * computed_average_population);

      REQUIRE_THAT(computed_average_population,
                   Catch::Matchers::WithinRel(
                     tree.get_average_cell_population(level), tol));
      REQUIRE_THAT(
        computed_std_population,
        Catch::Matchers::WithinRel(tree.get_std_cell_population(level), tol));
      REQUIRE(computed_max_population == tree.get_max_cell_population(level));
    }
  }
}
