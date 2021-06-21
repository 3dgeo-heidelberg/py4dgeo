add_library(py4dgeo py4dgeo.cpp)
target_include_directories(
  py4dgeo PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
                 $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
