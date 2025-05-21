cmake_minimum_required(VERSION 3.14)
project(BlackScholes)

find_package(OpenMP REQUIRED)
find_package(benchmark REQUIRED)

add_executable(bs src/main.cc src/compute.cc)
target_link_libraries(bs llvm-openmp::llvm-openmp benchmark::benchmark)
target_include_directories(bs PRIVATE include)
target_compile_options(bs PRIVATE "-march=native" "-ffast-math")
