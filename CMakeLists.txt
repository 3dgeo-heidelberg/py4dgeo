cmake_minimum_required(VERSION 3.30)

# ------------------------------------------------
# Set a name and a version number for the project:
# ------------------------------------------------
project(
  py4dgeo
  VERSION 1.0.0
  LANGUAGES CXX)

# Take into account <Package>_ROOT environment variable (used in packaging
# process)
cmake_policy(SET CMP0074 NEW)

# Initialize some default paths
include(GNUInstallDirs)

# Enable PIC for Python bindings
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -------------------
# Compilation options
# -------------------
set(BUILD_PYTHON
    ON
    CACHE BOOL "Enable building of Python bindings")
set(BUILD_DOCS
    ON
    CACHE BOOL "Enable building of documentation")
set(BUILD_BENCHMARKS
    OFF
    CACHE BOOL "Enable building of benchmark applications")
set(PY4DGEO_WITH_OPENMP
    ON
    CACHE BOOL "Enable OpenMP parallelization")

# ---------------------------
# Add the main library target
# ---------------------------
add_library(py4dgeo)

target_sources(py4dgeo
  PRIVATE
    lib/directions.cpp
    lib/distances.cpp
    lib/epoch.cpp
    lib/kdtree.cpp
    lib/octree.cpp
    lib/registration.cpp
    lib/segmentation.cpp
    lib/searchtree.cpp

    PUBLIC
      FILE_SET HEADERS
      BASE_DIRS
        include
)

target_compile_features(py4dgeo
  PRIVATE
    cxx_std_17
)

# --------------------------------------
# Add nanoflann interface library target
# --------------------------------------
add_library(nanoflann INTERFACE)

target_sources(nanoflann
  INTERFACE
    FILE_SET nanoflann_headers
    TYPE HEADERS
    BASE_DIRS
      ext/nanoflann/include
    FILES
      ext/nanoflann/include/nanoflann.hpp
)

target_compile_features(nanoflann
  INTERFACE
    cxx_std_11
)

# -----------------
# Eigen3 dependency
# -----------------
add_subdirectory(ext/eigen EXCLUDE_FROM_ALL)

# -----------------------------------
# Link nanoflann and eigen to py4dgeo
# -----------------------------------
target_link_libraries(py4dgeo
  PUBLIC
    nanoflann
    Eigen3::Eigen
)

# ------------------------------------------
# Enable OpenMP (if supported and requested)
# ------------------------------------------
if(PY4DGEO_WITH_OPENMP)
  if(MSVC)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("/openmp:llvm" MSVC_SUPPORTS_OPENMP_LLVM)
    if(MSVC_SUPPORTS_OPENMP_LLVM)
      set(OpenMP_RUNTIME_MSVC "llvm")
    endif()
  endif()

  find_package(OpenMP)
  if(OpenMP_FOUND)
    target_link_libraries(py4dgeo PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(py4dgeo
      PUBLIC
        PY4DGEO_WITH_OPENMP
        EIGEN_DONT_PARALLELIZE
    )
  endif()
endif()

# -------------------------------------
# Optionally build tests and benchmarks
# -------------------------------------
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

if(BUILD_BENCHMARKS)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
  set(BENCHMARK_ENABLE_TESTING OFF)
  add_subdirectory(./ext/benchmark)
  add_subdirectory(benchmarks)
endif()

# ---------------------
# Optionally build docs
# ---------------------
if(BUILD_DOCS)
  add_subdirectory(doc)
endif()

# ------------------------------
# Optionally build Python module
# ------------------------------
if(BUILD_PYTHON)
  # Add Python bindings
  find_package(pybind11 CONFIG REQUIRED)

  # Define Python extension module
  pybind11_add_module(_py4dgeo MODULE src/py4dgeo/py4dgeo_python.cpp)
  target_link_libraries(_py4dgeo PUBLIC py4dgeo)

  # Install th Python module library target
  install(TARGETS _py4dgeo DESTINATION .)
endif()

# -----------------------
# Show dependency summary
# -----------------------
include(FeatureSummary)
feature_summary(WHAT ALL)
