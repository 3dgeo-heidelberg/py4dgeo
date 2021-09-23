add_executable(tests tests.cpp testsetup.cpp bulkcompute_t.cpp kdtree_t.cpp)
target_link_libraries(tests PUBLIC py4dgeo Catch2::Catch2)

# allow user to run tests with `make test` or `ctest`
catch_discover_tests(tests)
