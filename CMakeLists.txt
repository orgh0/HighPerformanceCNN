#Things to do
# Load Cuda
# Include project includes from ./includes
# Include cuda from /usr/local/cuda-10.0/include
# Set c++ to 14
# Make exec for main.cpp as main


cmake_minimum_required (VERSION 3.8)

# project name
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.0/bin/nvcc")
project(HighPerformanceCNN LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})


if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++14")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/includes)

file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/includes/*.hpp)
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${HEADERS} PROPERTIES LANGUAGE CUDA)

add_executable(run ${SOURCES})