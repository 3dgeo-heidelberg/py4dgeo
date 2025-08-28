# Individual benchmark programs
add_executable(changepoint_detection_b changepoint_detection_b.cpp)
target_link_libraries(changepoint_detection_b
                      PUBLIC benchmark::benchmark py4dgeo_test Threads::Threads)
add_test(NAME changepoint_detection_b COMMAND changepoint_detection_b)

add_executable(cylindersearch_b cylindersearch_b.cpp)
target_link_libraries(cylindersearch_b PUBLIC benchmark::benchmark py4dgeo_test
                                              Threads::Threads)
add_test(NAME cylindersearch_b COMMAND cylindersearch_b)

add_executable(distances_b distances_b.cpp)
target_link_libraries(distances_b PUBLIC benchmark::benchmark py4dgeo_test
                                         Threads::Threads)
add_test(NAME distances_b COMMAND distances_b)

add_executable(scaling scaling.cpp)
target_link_libraries(scaling PUBLIC benchmark::benchmark py4dgeo_test
                                     Threads::Threads)
add_test(NAME scaling COMMAND scaling)

# Copy the scalability notebook into the build for ease of use
file(COPY scalability.ipynb DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(multiscale_normal multiscale_normal.cpp)
target_link_libraries(multiscale_normal PUBLIC benchmark::benchmark
                                               py4dgeo_test Threads::Threads)
add_test(NAME multiscale_normal COMMAND multiscale_normal)

add_executable(multiscale_normal_debug multiscale_normal_debug.cpp)
set(MIMALLOC_DIR "C:/Users/Thomas/Projects/mimalloc/out/release/Release")
set(MIMALLOC_INCLUDE_DIR "C:/Users/Thomas/Projects/mimalloc/include")
target_include_directories(multiscale_normal_debug
                           PRIVATE ${CMAKE_SOURCE_DIR}/../mimalloc/include)
target_link_libraries(multiscale_normal_debug PUBLIC py4dgeo py4dgeo_test
                                                     Threads::Threads)
if(MSVC)
  set(MIMALLOC_DIR "C:/Users/Thomas/Projects/mimalloc/out/release/Release")
  target_link_libraries(multiscale_normal_debug
                        PRIVATE "${MIMALLOC_DIR}/mimalloc.lib")
  target_compile_definitions(multiscale_normal_debug
                             PRIVATE MIMALLOC_DYNAMIC_LINK=1)
endif()
