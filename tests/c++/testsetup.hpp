#include <py4dgeo/py4dgeo.hpp>

#include <memory>
#include <string>
#include <tuple>
#include <utility>

#ifndef PY4DGEO_TEST_DATA_DIRECTORY
#error Test data directory needs to be set from CMake
#endif

#define DATAPATH(filename) PY4DGEO_TEST_DATA_DIRECTORY "/" #filename

std::shared_ptr<py4dgeo::EigenPointCloud>
benchcloud_from_file(const std::string& filename);

std::pair<std::shared_ptr<py4dgeo::EigenPointCloud>,
          std::shared_ptr<py4dgeo::EigenPointCloud>>
ahk_benchcloud();

std::pair<std::shared_ptr<py4dgeo::EigenPointCloud>,
          std::shared_ptr<py4dgeo::EigenPointCloud>>
testcloud();

std::pair<std::shared_ptr<py4dgeo::EigenPointCloud>,
          std::shared_ptr<py4dgeo::EigenPointCloud>>
testcloud_dif_files();
