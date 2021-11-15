add_executable(bench bench.cc distances_b.cc)
target_link_libraries(bench PUBLIC py4dgeo benchmark::benchmark)
