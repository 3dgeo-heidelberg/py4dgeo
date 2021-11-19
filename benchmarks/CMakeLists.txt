add_executable(bench bench.cpp distances_b.cpp scaling.cpp)
target_link_libraries(bench PUBLIC py4dgeo benchmark::benchmark)

# Copy the scalability notebook into the build for ease of use
file(COPY scalability.ipynb DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
