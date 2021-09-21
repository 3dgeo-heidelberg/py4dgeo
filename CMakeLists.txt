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
add_subdirectory(ext/eigen)
set(BUILD_TESTING ${BKP_BUILD_TESTING})

# compile the library
add_subdirectory(src)

# compile the application
add_subdirectory(app)

# compile the tests
if(BUILD_TESTING)
  add_subdirectory(ext/Catch2)
  include(./ext/Catch2/contrib/Catch.cmake)
  add_subdirectory(tests)
endif()

if(BUILD_DOCS)
  # Add the documentation
  add_subdirectory(doc)
endif()

if(BUILD_PYTHON)
  # Add Python bindings
  add_subdirectory(ext/pybind11)
  add_subdirectory(py4dgeo)

  # Generate a file that sets the PYTHONPATH during development This is
  # necessary to have access to both the pure Python part and the compiled
  # module. To use it, source the generated file.
  file(
    GENERATE
    OUTPUT ${CMAKE_BINARY_DIR}/setup-pythonpath.sh
    CONTENT
      "
export PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/py4dgeo\
:${PROJECT_SOURCE_DIR}/py4dgeo:$PYTHONPATH
      ")
endif()

# Add an alias target for use if this project is included as a subproject in
# another project
add_library(py4dgeo::py4dgeo ALIAS py4dgeo)

# I would really love to export my targets here, but the fact that PCL does not
# properly do it with their libraries prevents it.

# This prints a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)
