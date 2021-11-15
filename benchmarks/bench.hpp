#include <benchmark/benchmark.h>
#include <py4dgeo/py4dgeo.hpp>

#include <memory>
#include <string>

using namespace py4dgeo;

std::shared_ptr<EigenPointCloud>
benchcloud_from_file(const std::string& filename);
std::pair<std::shared_ptr<EigenPointCloud>, std::shared_ptr<EigenPointCloud>>
ahk_benchcloud();
