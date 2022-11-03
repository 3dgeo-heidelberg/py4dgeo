# Individual benchmark programs
add_executable(changepoint_detection_b changepoint_detection_b.cpp)
target_link_libraries(changepoint_detection_b PUBLIC benchmark::benchmark
                                                     py4dgeo_test)

add_executable(cylindersearch_b cylindersearch_b.cpp)
target_link_libraries(cylindersearch_b PUBLIC benchmark::benchmark py4dgeo_test)

add_executable(distances_b distances_b.cpp)
target_link_libraries(distances_b PUBLIC benchmark::benchmark py4dgeo_test)

add_executable(scaling scaling.cpp)
target_link_libraries(scaling PUBLIC benchmark::benchmark py4dgeo_test)

# Copy the scalability notebook into the build for ease of use
file(COPY scalability.ipynb DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(multiscale_normal multiscale_normal.cpp)
target_link_libraries(multiscale_normal PUBLIC benchmark::benchmark
                                               py4dgeo_test)
