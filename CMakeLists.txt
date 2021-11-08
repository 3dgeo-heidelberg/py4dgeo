cmake_minimum_required(VERSION 3.9)

# Set a name and a version number for your project:
project(
  py4dgeo
  VERSION 0.0.1
  LANGUAGES CXX)

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

# Check that the repository was clone recursively
if(NOT EXISTS ${PROJECT_SOURCE_DIR}/ext/Catch2/CMakeLists.txt)
  message(
    FATAL_ERROR
      "Submodules not found. py4dgeo needs to be either cloned with the"
      "'--recursive' flag or 'git submodule update --init' needs to be called")
endif()

# Enable CTest early to set the BUILD_TESTING option
include(CTest)

# Handle library dependencies
set(BKP_BUILD_TESTING ${BUILD_TESTING})
set(BUILD_TESTING OFF)
add_subdirectory(ext/eigen EXCLUDE_FROM_ALL)
set(BUILD_TESTING ${BKP_BUILD_TESTING})

# compile the library
add_subdirectory(lib)

# compile the application
add_subdirectory(app)

# compile the tests
if(BUILD_TESTING)
  add_subdirectory(ext/Catch2)
  include(./ext/Catch2/contrib/Catch.cmake)
  add_subdirectory(tests)
endif()

# Add the documentation
if(BUILD_DOCS)
  add_subdirectory(doc)
endif()

if(BUILD_PYTHON)
  # Add Python bindings
  add_subdirectory(ext/pybind11)
  pybind11_add_module(_py4dgeo MODULE src/py4dgeo/py4dgeo_python.cpp)
  target_link_libraries(_py4dgeo PUBLIC py4dgeo)
endif()

# Install library targets
install(TARGETS _py4dgeo DESTINATION .)

# This prints a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)
