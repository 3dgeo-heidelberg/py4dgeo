add_executable(bench bench.cpp distances_b.cpp)
target_link_libraries(bench PUBLIC py4dgeo benchmark::benchmark)
