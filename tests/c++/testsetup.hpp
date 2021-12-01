#include "py4dgeo/py4dgeo.hpp"

#ifndef PY4DGEO_TEST_DATA_DIRECTORY
#error Test data directory needs to be set from CMake
#endif

#define DATAPATH(filename) PY4DGEO_TEST_DATA_DIRECTORY "/" #filename

py4dgeo::EigenPointCloud
testcloud();
