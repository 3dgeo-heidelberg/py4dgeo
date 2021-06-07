add_executable(tests tests.cpp geolib4d_t.cpp)
target_link_libraries(tests PUBLIC geolib4d Catch2::Catch2)

# allow user to run tests with `make test` or `ctest`
catch_discover_tests(tests)
