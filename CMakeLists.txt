cmake_minimum_required(VERSION 3.9)

# Set a name and a version number for your project:
project(geolib4d VERSION 0.0.1 LANGUAGES CXX)

# Initialize some default paths
include(GNUInstallDirs)

# Define the minimum C++ standard that is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable PIC for Python bindings
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compilation options
set(BUILD_PYTHON ON CACHE BOOL "Enable building of Python bindings")
set(BUILD_DOCS ON CACHE BOOL "Enable building of documentation")

# compile the library
add_subdirectory(src)

# compile the application
add_subdirectory(app)

# compile the tests
include(CTest)
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
  add_subdirectory(python)
endif()

# Build the bundled external libraries
add_subdirectory(ext)

# Add an alias target for use if this project is included as a subproject in another project
add_library(geolib4d::geolib4d ALIAS geolib4d)

# I would really love to export my targets here, but the fact that PCL
# does not properly do it with their libraries prevents it.

# This prints a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)
