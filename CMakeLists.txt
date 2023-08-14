set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

cmake_minimum_required(VERSION 3.10)
project(cuda-playground LANGUAGES CUDA CXX)
include_directories($ENV{HOME}/cutlass/include)
add_executable(cuda-playground cutlass_print_fp16.cu)

