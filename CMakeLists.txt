cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ResNet-LibTorch)

set(CMAKE_CXX_STANDARD 14)

# Boost
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

# LibTorch
# You may set `CMAKE_PREFIX_PATH` into a path with LibTorch in
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})

include_directories(include)

add_subdirectory(src)
add_subdirectory(examples)
