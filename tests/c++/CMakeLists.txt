add_executable(tests tests.cpp testsetup.cpp directions_t.cpp distances_t.cpp
                     epoch_t.cpp kdtree_t.cpp)

target_link_libraries(tests PUBLIC py4dgeo Catch2::Catch2)
target_compile_definitions(
  tests PUBLIC PY4DGEO_TEST_DATA_DIRECTORY="${CMAKE_SOURCE_DIR}/tests/data")

# allow user to run tests with `make test` or `ctest`
catch_discover_tests(tests)
