add_library(py4dgeo directions.cpp distances.cpp epoch.cpp kdtree.cpp)

target_include_directories(
  py4dgeo PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/ext/eigen
                 ${CMAKE_SOURCE_DIR}/ext/nanoflann/include)
