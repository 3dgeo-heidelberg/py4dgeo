# Configure which PCL modules we are interested in
set(BUILD_2d                         OFF )
set(BUILD_common                     ON )
set(BUILD_features                   OFF)
set(BUILD_filters                    OFF)
set(BUILD_geometry                   OFF)
set(BUILD_global_tests               OFF)
set(BUILD_io                         ON )
set(BUILD_kdtree                     ON )
set(BUILD_keypoints                  OFF)
set(BUILD_ml                         OFF)
set(BUILD_octree                     ON )
set(BUILD_range_image                OFF)
set(BUILD_recognition                OFF)
set(BUILD_registration               OFF)
set(BUILD_sample_consensus           OFF)
set(BUILD_search                     OFF)
set(BUILD_segmentation               OFF)
set(BUILD_stereo                     OFF)
set(BUILD_surface                    OFF)
set(BUILD_tools                      ON)
set(BUILD_tracking                   OFF)
set(BUILD_visualization              OFF)

# Add the Point Cloud Library
add_subdirectory(pcl)

find_package(Eigen3 REQUIRED)

# Add a convenience target that we can link against to link against PCL
add_library(pcl INTERFACE)
target_link_libraries(pcl INTERFACE pcl_common pcl_io Eigen3::Eigen)
target_include_directories(
    pcl
    INTERFACE
    ${CMAKE_CURRENT_BINARY_DIR}/pcl/include
    pcl/common/include
    pcl/io/include
    )