add_library(py4dgeo directions.cpp distances.cpp epoch.cpp kdtree.cpp)

target_include_directories(
  py4dgeo PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
                 $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(py4dgeo PUBLIC Eigen3::Eigen)
