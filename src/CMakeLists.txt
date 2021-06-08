add_library(geolib4d geolib4d.cpp)
target_include_directories(geolib4d PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(geolib4d PUBLIC pcl)
