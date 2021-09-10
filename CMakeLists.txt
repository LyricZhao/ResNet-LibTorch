cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ResNet-LibTorch)

set(CMAKE_CXX_STANDARD 14)

# You may set `CMAKE_PREFIX_PATH` into a path with LibTorch in
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(inference inference.cpp)
target_link_libraries(inference "${TORCH_LIBRARIES}")

add_executable(train train.cpp)
target_link_libraries(train "${TORCH_LIBRARIES}")
