# ------------------------------------------------------------------------------
# Enable CMake's FetchContent module, which allows downloading dependencies like
# Catch2 automatically at configure time.
# ------------------------------------------------------------------------------
include(FetchContent)

# ------------------------------------------------------------------------------
# Declare a dependency on Catch2 via FetchContent. This will clone the Catch2
# GitHub repository during configuration.
# ------------------------------------------------------------------------------
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.10.0)

# ------------------------------------------------------------------------------
# Make Catch2 available in this project. This defines the CMake targets like
# Catch2::Catch2 and Catch2::Catch2WithMain.
# ------------------------------------------------------------------------------
FetchContent_MakeAvailable(Catch2)

# ------------------------------------------------------------------------------
# Make Catch2 available in this project. This defines the CMake targets like
# Catch2::Catch2 and Catch2::Catch2WithMain.
# ------------------------------------------------------------------------------
include(Catch)

# ------------------------------------------------------------------------------
# Build the py4dgeo_test library (a helper for all test files). This provides
# access to shared setup/utility functions and test data.
# ------------------------------------------------------------------------------
add_library(py4dgeo_test testsetup.cpp)

# Link it to the main py4dgeo library under test.
target_link_libraries(py4dgeo_test PUBLIC py4dgeo)

# Define a macro to pass the test data path into the tests.
target_compile_definitions(
  py4dgeo_test
  PUBLIC PY4DGEO_TEST_DATA_DIRECTORY="${CMAKE_SOURCE_DIR}/tests/data")

# Also expose the test setup includes (e.g., testsetup.hpp)
target_include_directories(py4dgeo_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ------------------------------------------------------------------------------
# Create a single test binary that compiles all unit test files. Each file
# should use TEST_CASE(...) macros from Catch2.
# ------------------------------------------------------------------------------
add_executable(
  tests
  directions_t.cpp
  distances_t.cpp
  epoch_t.cpp
  kdtree_t.cpp
  octree_t.cpp
  registration_t.cpp
  segmentation_t.cpp
  searchtrees_t.cpp)

# Link the test binary against: - the main library under test (py4dgeo) - the
# test helper library (py4dgeo_test) - the Catch2 main entry point
# (Catch2::Catch2WithMain)
target_link_libraries(tests PUBLIC py4dgeo py4dgeo_test Catch2::Catch2WithMain)

# ------------------------------------------------------------------------------
# Register the tests with CTest so that `ctest` or `make test` will run them.
# This parses all TEST_CASE(...) macros in the 'tests' binary.
# ------------------------------------------------------------------------------
catch_discover_tests(tests)
