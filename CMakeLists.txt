cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_COMPILER "g++-11")
set(CMAKE_CUDA_HOST_COMPILER "g++-11")

project(ParallelInpainting LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61)

if(APPLE)
    set(OpenCV_DIR "/opt/homebrew/opt/opencv/lib/cmake/opencv4")
elseif(UNIX)
    set(OpenCV_DIR "$ENV{HOME}/local/lib/cmake/opencv4")
endif()

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable(inpaint src/main.cpp src/inpaint.cpp src/masked_image.cpp src/nnf.cpp src/nnf.cu src/cuda.cpp src/cuda_helpers.cu)

target_link_libraries(inpaint ${OpenCV_LIBS})
