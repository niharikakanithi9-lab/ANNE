cmake_minimum_required(VERSION 3.20)
project(ann_engine CXX)
 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
 
# ---------- Build type defaults ----------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
 
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -fsanitize=address,undefined")
 
# ---------- Detect AVX2 ----------
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
  message(STATUS "AVX2 supported — SIMD kernels enabled")
  add_compile_definitions(HAVE_AVX2)
else()
  message(STATUS "AVX2 not supported — scalar fallback only")
endif()
 
# ---------- Dependencies via FetchContent ----------
include(FetchContent)
 
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Declare(
  benchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Declare(
  cli11
  URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.4.1.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
 
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
 
FetchContent_MakeAvailable(googletest benchmark cli11)
 
# ---------- Core library ----------
add_library(ann_engine_lib
  src/hnsw.cpp
  src/distance.cpp
  src/io.cpp
)
target_include_directories(ann_engine_lib PUBLIC include/)
 
# ---------- CLI executable ----------
add_executable(ann-engine src/main.cpp)
target_link_libraries(ann-engine PRIVATE ann_engine_lib CLI11::CLI11)
 
# ---------- Benchmarks ----------
add_executable(bench_distance bench/bench_distance.cpp)
target_link_libraries(bench_distance PRIVATE ann_engine_lib benchmark::benchmark)
 
add_executable(bench_query bench/bench_query.cpp)
target_link_libraries(bench_query PRIVATE ann_engine_lib benchmark::benchmark)
 
# ---------- Tests ----------
enable_testing()
 
add_executable(tests
  tests/test_distance.cpp
  tests/test_hnsw.cpp
  tests/test_io.cpp
  tests/test_recall.cpp
)
target_link_libraries(tests PRIVATE ann_engine_lib GTest::gtest_main)
add_test(NAME all_tests COMMAND tests)
 
# ---------- Optional pybind11 ----------
find_package(Python3 COMPONENTS Interpreter Development QUIET)
if(Python3_FOUND)
  FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.12.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(pybind11)
  pybind11_add_module(ann_engine_py python/bindings.cpp)
  target_link_libraries(ann_engine_py PRIVATE ann_engine_lib)
  message(STATUS "Python ${Python3_VERSION} found — building pybind11 bindings")
else()
  message(STATUS "Python not found — skipping pybind11 bindings")
endif()