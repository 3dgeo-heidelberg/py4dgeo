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
#include <optional>
#include <string>
#include <vector>

using namespace py4dgeo;

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

  SECTION("Cross check statistics derived from get_cell_population with "
          "pre-computed stats")
  {
    constexpr unsigned int level = 2;

    // Get per-point cell coordinates at this level
    const auto coords = tree.get_coordinates(level);

    // Collect all (x,y,z) triples into a vector and unique them
    std::vector<std::array<unsigned int, 3>> cells;
    cells.reserve(static_cast<std::size_t>(coords.rows()));

    for (IndexType i = 0; i < coords.rows(); ++i) {
      cells.push_back({ static_cast<unsigned int>(coords(i, 0)),
                        static_cast<unsigned int>(coords(i, 1)),
                        static_cast<unsigned int>(coords(i, 2)) });
    }

    auto cell_less = [](const auto& a, const auto& b) {
      if (a[0] != b[0])
        return a[0] < b[0];
      if (a[1] != b[1])
        return a[1] < b[1];
      return a[2] < b[2];
    };
    auto cell_equal = [](const auto& a, const auto& b) {
      return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
    };

    std::sort(cells.begin(), cells.end(), cell_less);
    cells.erase(std::unique(cells.begin(), cells.end(), cell_equal),
                cells.end());

    // Number of unique occupied cells matches the value stored in the octree
    REQUIRE(cells.size() == tree.get_occupied_cells_per_level(level));

    // Query population for each occupied cell and recompute stats
    std::vector<unsigned int> pops;
    pops.reserve(cells.size());

    unsigned int sum_pop = 0;
    unsigned int max_pop = 0;

    for (const auto& c : cells) {
      Octree::OctreeCoordinate coord = { c[0], c[1], c[2] };
      const unsigned int p = tree.get_cell_population(coord, level);
      pops.push_back(p);

      sum_pop += p;
      max_pop = std::max(max_pop, p);
    }

    // Populations over all occupied cells must sum to the number of points
    REQUIRE(sum_pop == epoch.cloud.rows());
    REQUIRE(sum_pop == tree.get_number_of_points());

    // Recompute mean and stddev
    const double mean =
      static_cast<double>(sum_pop) / static_cast<double>(pops.size());

    double sq_sum = 0.0;
    for (auto p : pops) {
      const double d = static_cast<double>(p) - mean;
      sq_sum += d * d;
    }
    const double stddev = std::sqrt(sq_sum / static_cast<double>(pops.size()));

    // Compare with Octree's cached per-level stats
    REQUIRE(max_pop == tree.get_max_cell_population_per_level(level));
    REQUIRE_THAT(mean,
                 Catch::Matchers::WithinRel(
                   tree.get_average_cell_population_per_level(level), 1e-12));
    REQUIRE_THAT(stddev,
                 Catch::Matchers::WithinRel(
                   tree.get_std_cell_population_per_level(level), 1e-12));
  }
}
