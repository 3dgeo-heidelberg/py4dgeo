#include "testsetup.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace py4dgeo;

std::string
datapath(const char* filename)
{
  const std::filesystem::path base(PY4DGEO_TEST_DATA_DIRECTORY);

  auto find_in = [&](const std::filesystem::path& root) -> std::string {
    if (!std::filesystem::exists(root))
      return {};
    if (std::filesystem::is_regular_file(root) && root.filename() == filename)
      return root.string();
    if (std::filesystem::is_directory(root)) {
      for (const auto& entry :
           std::filesystem::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().filename() == filename) {
          return entry.path().string();
        }
      }
    }
    return {};
  };

  // Direct and extracted under the configured test directory
  for (const std::filesystem::path& root : { base, base / "extracted" }) {
    if (auto hit = find_in(root); !hit.empty())
      return hit;
  }

  // Common pooch cache locations (OS-dependent)
  std::vector<std::filesystem::path> pooch_roots;
  if (const char* xdg = std::getenv("XDG_CACHE_HOME"))
    pooch_roots.emplace_back(xdg);
  if (const char* home = std::getenv("HOME")) {
    pooch_roots.emplace_back(std::filesystem::path(home) / ".cache");
    pooch_roots.emplace_back(std::filesystem::path(home) / "Library" /
                             "Caches"); // macOS
  }
#ifdef _WIN32
  if (const char* local = std::getenv("LOCALAPPDATA"))
    pooch_roots.emplace_back(local);
  if (const char* appdata = std::getenv("APPDATA"))
    pooch_roots.emplace_back(appdata);
#endif

  std::vector<std::filesystem::path> pooch_candidates;
  for (const auto& root : pooch_roots) {
    pooch_candidates.push_back(root / "pooch");
    pooch_candidates.push_back(root / "pooch" / "py4dgeo");
    pooch_candidates.push_back(root / "py4dgeo");
  }

  for (const auto& root : pooch_candidates) {
    if (auto hit = find_in(root); !hit.empty())
      return hit;
  }

  std::cerr << "Searching for test data in:\n"
            << "  - " << base << "\n"
            << "  - " << base / "extracted"
            << "\n";
  for (const auto& root : pooch_candidates) {
    std::cerr << "  - " << root << "\n";
  }
  std::cerr << "Test data file not found: " << filename << "\n";
  std::exit(1);
}

std::shared_ptr<EigenPointCloud>
benchcloud_from_file(const std::string& filename)
{
  std::ifstream stream(filename);
  if (!stream) {
    std::cerr << "Was not successfully opened. Please check that the file "
                 "currently exists: "
              << filename << std::endl;
    std::exit(1);
  }

  std::vector<Eigen::Vector3d> points;
  Eigen::Vector3d mincoord =
    Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());

  std::string line;
  while (std::getline(stream, line)) {
    std::istringstream s(line);
    Eigen::Vector3d point;
    s >> point[0] >> point[1] >> point[2];

    if (!s)
      continue;

    mincoord = mincoord.cwiseMin(point);
    points.push_back(point);
  }

  auto cloud = std::make_shared<EigenPointCloud>(points.size(), 3);
  for (std::size_t i = 0; i < points.size(); ++i) {
    (*cloud).row(i) = points[i] - mincoord;
  }

  return cloud;
}

std::shared_ptr<EigenPointCloud>
slice_cloud(EigenPointCloudConstRef cloud, int sampling_factor)
{
  auto sliced =
    std::make_shared<EigenPointCloud>(cloud.rows() / sampling_factor, 3);
  for (IndexType i = 0; i < cloud.rows() / sampling_factor; ++i)
    (*sliced)(i, Eigen::indexing::all) =
      cloud(i * sampling_factor, Eigen::indexing::all);
  return sliced;
}

std::pair<std::shared_ptr<EigenPointCloud>, std::shared_ptr<EigenPointCloud>>
ahk_benchcloud()
{
  auto cloud = benchcloud_from_file(datapath("plane_horizontal_t1.xyz"));
  return std::make_pair(cloud, slice_cloud(*cloud, 100));
}

std::pair<std::shared_ptr<EigenPointCloud>, std::shared_ptr<EigenPointCloud>>
testcloud()
{
  auto cloud = benchcloud_from_file(datapath("plane_horizontal_t1.xyz"));
  return std::make_pair(cloud, cloud);
}

std::pair<std::shared_ptr<EigenPointCloud>, std::shared_ptr<EigenPointCloud>>
testcloud_dif_files()
{
  auto cloud1 = benchcloud_from_file(datapath("plane_horizontal_t1.xyz"));
  auto cloud2 = benchcloud_from_file(datapath("plane_horizontal_t2.xyz"));
  return std::make_pair(cloud1, cloud2);
}
