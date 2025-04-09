# Create the minimal benchmark target
add_executable(minimal_benchmark compute_normals.cpp)

target_link_libraries(minimal_benchmark PUBLIC py4dgeo py4dgeo_test)
