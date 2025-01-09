cmake_minimum_required(VERSION 3.9)

# Set a name and a version number for your project:
project(
  py4dgeo
  VERSION 0.0.1
  LANGUAGES CXX)

# Take into account <Package>_ROOT environment variable (used in packaging
# process)
cmake_policy(SET CMP0074 NEW)

# Initialize some default paths
include(GNUInstallDirs)

# Define the minimum C++ standard that is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable PIC for Python bindings
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compilation options
set(BUILD_PYTHON
    ON
    CACHE BOOL "Enable building of Python bindings")
set(BUILD_DOCS
    ON
    CACHE BOOL "Enable building of documentation")
set(BUILD_BENCHMARKS
    OFF
    CACHE BOOL "Enable building of benchmark applications")
set(PY4DGEO_WITH_OPENMP
    ON
    CACHE BOOL "Enable OpenMP parallelization")

# Check that the repository was clone recursively
if(NOT EXISTS ${PROJECT_SOURCE_DIR}/ext/Catch2/CMakeLists.txt)
  message(
    FATAL_ERROR
      "Submodules not found. py4dgeo needs to be either cloned with the"
      "'--recursive' flag or 'git submodule update --init' needs to be called")
endif()

# Compile the library
add_library(py4dgeo lib/directions.cpp lib/distances.cpp lib/epoch.cpp
                    lib/kdtree.cpp lib/registration.cpp lib/segmentation.cpp)
target_include_directories(
  py4dgeo PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/ext/eigen
                 ${CMAKE_SOURCE_DIR}/ext/nanoflann/include)
if(PY4DGEO_WITH_OPENMP)
  find_package(OpenMP)
  if(OpenMP_FOUND)
    target_link_libraries(py4dgeo PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(py4dgeo PUBLIC PY4DGEO_WITH_OPENMP
                                              EIGEN_DONT_PARALLELIZE)
  endif()
endif()

# Compile the tests and benchmarks
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(ext/Catch2)
  include(./ext/Catch2/contrib/Catch.cmake)
  add_subdirectory(tests)
endif()

if(BUILD_BENCHMARKS)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
  set(BENCHMARK_ENABLE_TESTING OFF)
  add_subdirectory(./ext/benchmark)
  add_subdirectory(benchmarks)
endif()

# Add the documentation
if(BUILD_DOCS)
  add_subdirectory(doc)
endif()

if(BUILD_PYTHON)
  # Add Python bindings
  find_package(pybind11 REQUIRED)
  pybind11_add_module(_py4dgeo MODULE src/py4dgeo/py4dgeo_python.cpp)
  target_link_libraries(_py4dgeo PUBLIC py4dgeo)

  # Install th Python module library target
  install(TARGETS _py4dgeo DESTINATION .)
endif()

# This prints a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)
