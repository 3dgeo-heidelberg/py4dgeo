cmake_minimum_required(VERSION 3.9)

# Set a name and a version number for your project:
project(py4dgeo VERSION 0.0.1 LANGUAGES CXX)

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

# Add an alias target for use if this project is included as a subproject in another project
add_library(py4dgeo::py4dgeo ALIAS py4dgeo)

# Install targets and configuration
install(
  TARGETS py4dgeo
  EXPORT py4dgeo-config
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
  EXPORT py4dgeo-config
  NAMESPACE py4dgeo::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/py4dgeo
)

install(
  DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# This prints a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)
