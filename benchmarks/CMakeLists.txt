add_library(bench bench.cpp)
target_link_libraries(bench PUBLIC py4dgeo benchmark::benchmark)

add_executable(distances_b distances_b.cpp)
target_link_libraries(distances_b PUBLIC bench)

add_executable(scaling scaling.cpp)
target_link_libraries(scaling PUBLIC bench)

# Copy the scalability notebook into the build for ease of use
file(COPY scalability.ipynb DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
