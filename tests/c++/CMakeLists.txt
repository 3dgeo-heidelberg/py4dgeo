# Test support library that implements e.g. easy access to data files
add_library(py4dgeo_test testsetup.cpp)
target_link_libraries(py4dgeo_test PUBLIC py4dgeo)
target_compile_definitions(
  py4dgeo_test
  PUBLIC PY4DGEO_TEST_DATA_DIRECTORY="${CMAKE_SOURCE_DIR}/tests/data")
target_include_directories(py4dgeo_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# The Catch test executable
add_executable(
  tests
  tests.cpp
  directions_t.cpp
  distances_t.cpp
  epoch_t.cpp
  kdtree_t.cpp
  octree_t.cpp
  registration_t.cpp
  segmentation_t.cpp
  searchtrees_t.cpp)
target_link_libraries(tests PUBLIC py4dgeo py4dgeo_test Catch2::Catch2)

# allow user to run tests with `make test` or `ctest`
catch_discover_tests(tests)
